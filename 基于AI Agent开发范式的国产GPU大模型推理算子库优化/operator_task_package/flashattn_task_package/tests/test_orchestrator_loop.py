"""
orchestrator_loop 的单元测试。

测试策略：
- 用 mock 的 generate_candidates_fn / run_kernel_fn 模拟 Coder/Profiler
- 验证闭环数据流：Analyst→Coder→过滤→Judge→Logger
- 验证成功(KEEP)/失败(ROLLBACK)两种路径
- 验证记忆库与日志的更新
- 真实 kernel 的端到端测试在 test_e2e.py

运行：pytest tests/test_orchestrator_loop.py -v
"""
import pytest
import torch

from agent_system.orchestrator_loop import (
    analyst_analyze, judge_verdict, run_iteration, IterationResult,
)
from agent_system.roofline_engine import KernelConfig
from agent_system.llm_cost_model import Candidate
from agent_system.domain_memory import DomainMemory
from agent_system.optimization_log import OptimizationLog
from agent_system.benchmark_engine import BenchResult
from agent_system.correctness import CorrectnessResult


@pytest.fixture
def cfg():
    return KernelConfig(batch_size=1, seqlen_kv=256, headdim=32, num_heads=4)


@pytest.fixture
def memory(tmp_path):
    return DomainMemory(base_dir=tmp_path)


@pytest.fixture
def log(tmp_path):
    return OptimizationLog(log_path=tmp_path / "opt_log.md")


def make_baseline(cfg):
    """构造一个假 baseline BenchResult。"""
    return BenchResult(
        cfg=cfg, time_ms=1.0, time_ms_median=1.0, time_ms_min=1.0, time_ms_std=0,
        repeats=10, achievable_bw_gb_s=100, bandwidth_utilization=0.1,
        gap_to_roofline=10, bound_type="memory-bound",
    )


class TestAnalystAnalyze:
    def test_memory_bound_direction(self, cfg):
        result = analyst_analyze(cfg)
        assert "memory-bound" in result

    def test_includes_splitk_suggestion(self, cfg):
        result = analyst_analyze(cfg)
        # batch=1*heads=4 < SM 数，应建议 split
        assert "Split-K" in result or "split" in result.lower()

    def test_with_baseline_time(self, cfg):
        result = analyst_analyze(cfg, baseline_time_s=0.001)
        assert "gap_to_roofline" in result


class TestJudgeVerdict:
    def test_keep_on_speedup(self, cfg):
        baseline = make_baseline(cfg)
        faster = BenchResult(
            cfg=cfg, time_ms=0.5, time_ms_median=0.5, time_ms_min=0.5, time_ms_std=0,
            repeats=10, achievable_bw_gb_s=200, bandwidth_utilization=0.2,
            gap_to_roofline=5, bound_type="memory-bound",
        )
        corr = CorrectnessResult(
            passed=True, max_abs_diff=0.001, max_rel_diff=0.01,
            mean_abs_diff=0.001, num_elements=128, rtol=1e-2, atol=1e-2,
        )
        cand = Candidate(candidate_id="c1", description="test", params={})
        verdict, reason = judge_verdict(cand, baseline, faster, corr)
        assert verdict == "KEEP"

    def test_reject_on_correctness_fail(self, cfg):
        baseline = make_baseline(cfg)
        candidate = make_baseline(cfg)  # 相同时间
        corr = CorrectnessResult(
            passed=False, max_abs_diff=1.0, max_rel_diff=1.0,
            mean_abs_diff=1.0, num_elements=128, rtol=1e-2, atol=1e-2,
        )
        cand = Candidate(candidate_id="c1", description="test", params={})
        verdict, _ = judge_verdict(cand, baseline, candidate, corr)
        assert verdict == "REJECT"

    def test_rollback_on_no_speedup(self, cfg):
        baseline = make_baseline(cfg)
        slower = BenchResult(
            cfg=cfg, time_ms=2.0, time_ms_median=2, time_ms_min=2, time_ms_std=0,
            repeats=10, achievable_bw_gb_s=50, bandwidth_utilization=0.05,
            gap_to_roofline=20, bound_type="memory-bound",
        )
        corr = CorrectnessResult(
            passed=True, max_abs_diff=0.001, max_rel_diff=0.01,
            mean_abs_diff=0.001, num_elements=128, rtol=1e-2, atol=1e-2,
        )
        cand = Candidate(candidate_id="c1", description="test", params={})
        verdict, _ = judge_verdict(cand, baseline, slower, corr)
        assert verdict == "ROLLBACK"


class TestRunIteration:
    def test_successful_iteration(self, cfg, memory, log):
        """模拟一次成功迭代：Coder 生成候选，Profiler 返回正确+更快的输出"""
        # baseline 设得较慢（10ms），让 mock kernel 容易超越
        baseline = BenchResult(
            cfg=cfg, time_ms=10.0, time_ms_median=10, time_ms_min=10, time_ms_std=0,
            repeats=10, achievable_bw_gb_s=10, bandwidth_utilization=0.01,
            gap_to_roofline=100, bound_type="memory-bound",
        )

        # 预计算参考输出，mock kernel 直接返回（极快）
        from agent_system.correctness import make_test_inputs, generate_reference
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = make_test_inputs(cfg, device=device)
        q, k, v, lens, bt = inputs
        ref_output = generate_reference(q, k, v, lens, bt, cfg)

        def mock_generate(c, bottleneck, mem):
            return [Candidate(
                candidate_id="c1", description="split_k=4",
                params={"split_k": 4, "tile_n": 32}, confidence=0.8,
            )]

        def mock_run_kernel(cand, inp):
            return ref_output  # 直接返回预算好的正确输出（极快）

        result = run_iteration(
            iteration=1, cfg=cfg, baseline_result=baseline,
            generate_candidates_fn=mock_generate,
            run_kernel_fn=mock_run_kernel,
            memory=memory, log=log,
            warmup=1, repeats=2,
        )
        assert result.verdict == "KEEP"
        assert result.best_candidate is not None
        # 日志已记录
        assert len(log.entries) == 1
        assert log.entries[0].verdict == "KEEP"
        # 记忆已更新
        assert len(memory.belief.entries) >= 1

    def test_failed_iteration_no_candidates(self, cfg, memory, log):
        """Coder 生成 0 个候选 → ROLLBACK"""
        baseline = make_baseline(cfg)

        def mock_generate(c, b, m):
            return []

        def mock_run_kernel(cand, inputs):
            return torch.zeros(1)

        result = run_iteration(
            iteration=1, cfg=cfg, baseline_result=baseline,
            generate_candidates_fn=mock_generate,
            run_kernel_fn=mock_run_kernel,
            memory=memory, log=log, warmup=1, repeats=2,
        )
        assert result.verdict == "ROLLBACK"
        assert result.candidates_total == 0

    def test_runtime_error_recorded(self, cfg, memory, log):
        """候选运行报错 → 记录到 failure_cases"""
        baseline = make_baseline(cfg)

        def mock_generate(c, b, m):
            return [Candidate(candidate_id="c1", description="buggy", params={"split_k":2}, confidence=0.8)]

        def mock_run_kernel(cand, inputs):
            raise RuntimeError("kernel crashed")

        result = run_iteration(
            iteration=1, cfg=cfg, baseline_result=baseline,
            generate_candidates_fn=mock_generate,
            run_kernel_fn=mock_run_kernel,
            memory=memory, log=log, warmup=1, repeats=2,
        )
        assert result.verdict == "ROLLBACK"
        # 错误已记录到记忆库
        assert len(memory.failures) >= 1

    def test_filter_stats_populated(self, cfg, memory, log):
        """过滤统计应正确填充"""
        baseline = make_baseline(cfg)

        from agent_system.correctness import make_test_inputs, generate_reference
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = make_test_inputs(cfg, device=device)
        q, k, v, lens, bt = inputs
        ref_output = generate_reference(q, k, v, lens, bt, cfg)

        def mock_generate(c, b, m):
            return [
                Candidate(candidate_id=f"c{i}", description=f"sk={i}",
                          params={"split_k": i+1}, confidence=0.3+i*0.1)
                for i in range(5)
            ]

        def mock_run_kernel(cand, inp):
            return ref_output

        result = run_iteration(
            iteration=1, cfg=cfg, baseline_result=baseline,
            generate_candidates_fn=mock_generate,
            run_kernel_fn=mock_run_kernel,
            memory=memory, log=log, max_survivors=2,
            warmup=1, repeats=2,
        )
        assert result.filter_stats is not None
        assert result.filter_stats.total == 5
        assert result.filter_stats.survivors <= 2
