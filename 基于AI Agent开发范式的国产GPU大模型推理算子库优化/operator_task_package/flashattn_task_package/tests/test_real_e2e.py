"""
真实优化闭环的端到端测试。

这个测试消耗真实 LLM token + 真实 GPU，是最完整的验证。
默认跳过（slow），手动 --runslow 触发。

验证完整链路：
LLM生成候选 → 双层过滤 → mxcc编译 → GPU运行 → 正确性校验 → benchmark → 裁决 → 记忆/日志

运行：pytest tests/test_real_e2e.py -v --runslow
"""
import pytest
import torch
from pathlib import Path

from agent_system.roofline_engine import KernelConfig
from agent_system.domain_memory import DomainMemory
from agent_system.optimization_log import OptimizationLog
from agent_system.real_orchestrator import run_real_iteration, run_optimization_loop
from agent_system.real_coder import generate_candidates
from agent_system.real_cost_model import make_predict_fn
from agent_system.llm_client import is_available


pytestmark = pytest.mark.slow


@pytest.fixture
def workdir(tmp_path):
    return tmp_path


@pytest.fixture
def small_cfg():
    """小配置（快速测试，headdim=4 适配单warp baseline kernel）"""
    return KernelConfig(batch_size=1, seqlen_kv=16, headdim=4, num_heads=1)


@pytest.fixture
def memory(tmp_path):
    return DomainMemory(base_dir=tmp_path / "mem")


@pytest.fixture
def log(tmp_path):
    return OptimizationLog(log_path=tmp_path / "opt.md")


class TestRealCoder:
    """真实 LLM Coder 测试。"""

    @pytest.mark.skipif(not is_available(), reason="无 MOARK_API_KEY")
    def test_generate_candidates_returns_list(self, small_cfg, memory):
        """LLM 应返回候选列表"""
        candidates = generate_candidates(
            small_cfg, "memory-bound，建议 split_k", memory, num_candidates=2,
        )
        assert isinstance(candidates, list)
        # LLM 至少应返回 1 个候选（可能不完美，但应有输出）
        if candidates:
            assert hasattr(candidates[0], "description")
            assert hasattr(candidates[0], "params")


class TestRealCostModel:
    """真实 LLM cost model 测试。"""

    @pytest.mark.skipif(not is_available(), reason="无 MOARK_API_KEY")
    def test_predict_returns_tuple(self, small_cfg):
        from agent_system.real_cost_model import predict
        from agent_system.llm_cost_model import Candidate
        cand = Candidate("c1", "split_k=4", {"split_k": 4})
        speedup, conf = predict(cand, baseline_util=0.45, cfg=small_cfg)
        assert isinstance(speedup, float)
        assert isinstance(conf, float)
        assert speedup >= 0
        assert 0 <= conf <= 1

    @pytest.mark.skipif(not is_available(), reason="无 MOARK_API_KEY")
    def test_physical_clip(self, small_cfg):
        """LLM 预测的超大加速比应被 roofline clip"""
        from agent_system.real_cost_model import predict
        from agent_system.llm_cost_model import Candidate
        cand = Candidate("c1", "不可能的100x加速", {"split_k": 999})
        speedup, conf = predict(cand, baseline_util=0.45, cfg=small_cfg)
        # 不应超过物理上限太多
        assert speedup <= 1.0 / 0.45 * 1.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
@pytest.mark.skipif(not is_available(), reason="无 MOARK_API_KEY")
class TestRealIteration:
    """真实迭代闭环（LLM+编译+GPU）。"""

    def test_single_iteration_runs(self, small_cfg, memory, log, workdir):
        """单轮真实迭代应完整跑完（不崩溃）"""
        # 用 baseline kernel 代码作为起点
        baseline_code = (Path(__file__).resolve().parent.parent / "kernel" / "baseline_kernel.cu").read_text()

        result = run_real_iteration(
            iteration=1, cfg=small_cfg,
            baseline_time_ms=1.0,  # 假设 baseline 1ms
            baseline_util=0.45,
            generate_fn=generate_candidates,
            current_code=baseline_code,
            memory=memory, log=log, workdir=workdir,
            max_survivors=2, warmup=1, repeats=3,
        )
        # 验证完整跑完
        assert result.iteration == 1
        assert isinstance(result.verdict, str)
        assert result.verdict in ("KEEP", "ROLLBACK")
        # 日志已记录
        assert len(log.entries) == 1
        # 无论成败，都应有输出
        print(f"\n[真实迭代] 生成{result.candidates_generated} "
              f"编译OK{result.candidates_compiled_ok} "
              f"正确{result.candidates_correct} "
              f"裁决{result.verdict}")
        if result.errors:
            print(f"  错误: {'; '.join(result.errors[:3])}")

    def test_iteration_records_failures(self, small_cfg, memory, log, workdir):
        """迭代中的失败应记录到记忆库"""
        baseline_code = (Path(__file__).resolve().parent.parent / "kernel" / "baseline_kernel.cu").read_text()
        run_real_iteration(
            iteration=1, cfg=small_cfg,
            baseline_time_ms=1.0, baseline_util=0.45,
            generate_fn=generate_candidates, current_code=baseline_code,
            memory=memory, log=log, workdir=workdir,
            max_survivors=2, warmup=1, repeats=2,
        )
        # 如果有失败，应记录（取决于 LLM 生成质量）
        # 这里只验证不崩溃 + 记忆库可读写
        mem_ctx = memory.build_context("mctlass")
        assert isinstance(mem_ctx, str)
