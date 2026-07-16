"""
端到端集成测试 —— 验证整个 Agent 系统从输入到报告的完整流程。

这是最关键的测试，证明所有组件（roofline/correctness/benchmark/cost_model/
memory/orchestrator/log）能协同工作。

策略：
- 用 mock 的 Coder（生成候选）+ mock 的 kernel（返回参考输出）
- 跑完整 orchestrator 闭环
- 验证：日志生成、记忆更新、报告输出、加速比计算
- 真实 GPU baseline 单独标记 slow

运行：pytest tests/test_e2e.py -v
"""
import json
import pytest
import torch
from pathlib import Path

from agent_system.roofline_engine import KernelConfig
from agent_system.correctness import make_test_inputs, generate_reference
from agent_system.benchmark_engine import BenchResult, benchmark_official_flash_attn
from agent_system.llm_cost_model import Candidate
from agent_system.domain_memory import DomainMemory
from agent_system.optimization_log import OptimizationLog
from agent_system.orchestrator_loop import run_iteration


@pytest.fixture
def setup(tmp_path):
    """每轮测试的完整环境。"""
    cfg = KernelConfig(batch_size=1, seqlen_kv=256, headdim=32, num_heads=4)
    memory = DomainMemory(base_dir=tmp_path / "memory")
    log = OptimizationLog(log_path=tmp_path / "opt_log.md")
    # baseline 设得较慢，让 mock kernel 容易超越
    baseline = BenchResult(
        cfg=cfg, time_ms=10.0, time_ms_median=10, time_ms_min=10, time_ms_std=0,
        repeats=10, achievable_bw_gb_s=5, bandwidth_utilization=0.003,
        gap_to_roofline=300, bound_type="memory-bound",
    )
    # 预计算参考输出
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = make_test_inputs(cfg, device=device)
    ref_output = generate_reference(*inputs, cfg=cfg)
    return cfg, memory, log, baseline, inputs, ref_output


class TestEndToEndPipeline:
    """端到端：完整迭代闭环。"""

    def test_full_keep_iteration(self, setup):
        """成功路径：Analyst→Coder→Filter→Profiler→Judge(KEEP)→Reflector→Logger"""
        cfg, memory, log, baseline, inputs, ref_output = setup

        def mock_generate(c, bottleneck, mem):
            return [
                Candidate(candidate_id="c1", description="split_k=4,tile_n=32",
                          params={"split_k": 4, "tile_n": 32}, confidence=0.85),
                Candidate(candidate_id="c2", description="split_k=8",
                          params={"split_k": 8, "tile_n": 16}, confidence=0.6),
            ]

        def mock_run_kernel(cand, inp):
            return ref_output  # 正确且极快

        result = run_iteration(
            iteration=1, cfg=cfg, baseline_result=baseline,
            generate_candidates_fn=mock_generate,
            run_kernel_fn=mock_run_kernel,
            memory=memory, log=log, max_survivors=2,
            warmup=1, repeats=3,
        )

        # 验证完整闭环产出
        assert result.verdict == "KEEP"
        assert result.best_candidate is not None
        assert result.best_speedup > 1.0
        assert result.correctness.passed
        assert result.gap_to_roofline is not None

        # 日志已生成（md + jsonl）
        assert log.log_path.exists()
        assert log.jsonl_path.exists()
        assert len(log.entries) == 1
        entry = log.entries[0]
        assert entry.verdict == "KEEP"
        assert entry.speedup > 1.0

        # 记忆库已更新（成功 → 信念）
        assert len(memory.belief.entries) >= 1

        # 日志可读回
        log2 = OptimizationLog(log_path=log.log_path)
        assert len(log2.entries) == 1

    def test_full_rollback_no_candidates(self, setup):
        """失败路径：Coder 生成 0 候选 → ROLLBACK"""
        cfg, memory, log, baseline, inputs, ref_output = setup

        result = run_iteration(
            iteration=1, cfg=cfg, baseline_result=baseline,
            generate_candidates_fn=lambda c, b, m: [],
            run_kernel_fn=lambda cand, inp: ref_output,
            memory=memory, log=log, warmup=1, repeats=2,
        )
        assert result.verdict == "ROLLBACK"
        assert log.entries[0].verdict == "ROLLBACK"

    def test_multi_iteration_accumulation(self, setup):
        """多轮迭代：日志和记忆应累积"""
        cfg, memory, log, baseline, inputs, ref_output = setup

        def gen(c, b, m):
            return [Candidate(candidate_id="c", description="opt",
                              params={"split_k": 4}, confidence=0.8)]

        for i in range(3):
            run_iteration(
                iteration=i+1, cfg=cfg, baseline_result=baseline,
                generate_candidates_fn=gen,
                run_kernel_fn=lambda cand, inp: ref_output,
                memory=memory, log=log, warmup=1, repeats=2,
            )

        assert len(log.entries) == 3
        # 每轮都 KEEP，信念应累积
        assert len(memory.belief.entries) >= 3

    def test_context_injection_works(self, setup):
        """领域记忆的上下文注入应能被 Coder 使用"""
        cfg, memory, log, baseline, inputs, ref_output = setup
        # 预先注入一条失败案例
        memory.record_failure("api_misuse", "InstructionShape 用了 16x16x8",
                              "bf16 必须 16x16x16", "改为 16x16x16")

        ctx = memory.build_context("InstructionShape")
        assert "InstructionShape" in ctx
        assert "已知陷阱" in ctx

    def test_log_markdown_human_readable(self, setup):
        """日志 markdown 应人类可读（赛题可复现性物料）"""
        cfg, memory, log, baseline, inputs, ref_output = setup

        run_iteration(
            iteration=1, cfg=cfg, baseline_result=baseline,
            generate_candidates_fn=lambda c, b, m: [
                Candidate(candidate_id="c1", description="split_k=4",
                          params={"split_k": 4}, confidence=0.8)
            ],
            run_kernel_fn=lambda cand, inp: ref_output,
            memory=memory, log=log, warmup=1, repeats=2,
        )
        content = log.log_path.read_text()
        assert "优化日志" in content
        assert "| 轮次 |" in content
        assert "KEEP" in content
        assert "split_k=4" in content

    def test_cost_model_filter_integrated(self, setup):
        """创新点 C：双层过滤应在闭环中生效"""
        cfg, memory, log, baseline, inputs, ref_output = setup
        # 用更真实的 baseline utilization（45%），让 100x 加速违反物理
        baseline.bandwidth_utilization = 0.45

        # 一个物理可行，一个违反 roofline
        def gen(c, b, m):
            return [
                Candidate(candidate_id="feasible", description="split_k=4",
                          params={"split_k": 4}, predicted_speedup=2.0, confidence=0.8),
                Candidate(candidate_id="infeasible", description="split_k=99",
                          params={"split_k": 99}, predicted_speedup=100.0, confidence=0.9),
            ]

        result = run_iteration(
            iteration=1, cfg=cfg, baseline_result=baseline,
            generate_candidates_fn=gen,
            run_kernel_fn=lambda cand, inp: ref_output,
            memory=memory, log=log, warmup=1, repeats=2,
        )
        # infeasible(100x, baseline 45% → 上限~2.4x) 应被物理过滤剔除
        assert result.filter_stats.rejected_by_physics >= 1


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
class TestRealBaseline:
    """慢测试：真实官方 flash_attn baseline（手动 --runslow 触发）。"""

    def test_official_baseline_small(self):
        """跑一个小配置的真实官方 baseline，验证 benchmark 引擎在真实环境工作"""
        cfg = KernelConfig(batch_size=1, seqlen_kv=1024, headdim=128)
        result = benchmark_official_flash_attn(cfg, warmup=3, repeats=10)
        assert result.time_ms > 0
        assert result.bandwidth_utilization < 1.0
        print(f"\n[E2E 真实baseline] {result.summary()}")
        print(f"  gap_to_roofline = {result.gap_to_roofline:.2f}x（优化空间）")
