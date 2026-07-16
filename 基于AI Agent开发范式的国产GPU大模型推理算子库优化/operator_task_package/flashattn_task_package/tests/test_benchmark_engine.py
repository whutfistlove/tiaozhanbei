"""
benchmark_engine 的单元测试。

由于真实 GPU benchmark 较慢，测试策略：
- 核心逻辑（计时、stats、对比）用 CPU 模拟函数快速验证
- GPU 真实计时用一个最小配置验证不崩溃
- official flash_attn 单独标记 slow

运行：pytest tests/test_benchmark_engine.py -v
"""
import time
import pytest
import torch

from agent_system.benchmark_engine import (
    BenchResult,
    benchmark_fn,
    benchmark_config,
    compare_results,
)
from agent_system.roofline_engine import KernelConfig


class TestBenchmarkFn:
    def test_callable_check(self):
        with pytest.raises(TypeError):
            benchmark_fn("not callable")

    def test_cpu_timing_returns_times(self):
        """CPU 模拟函数应返回 repeats 个时间"""
        def sleep_fn():
            time.sleep(0.001)
        times = benchmark_fn(sleep_fn, warmup=2, repeats=5, use_cuda_event=False)
        assert len(times) == 5
        assert all(t > 0 for t in times)

    def test_repeats_count(self):
        counter = [0]
        def fn():
            counter[0] += 1
        times = benchmark_fn(fn, warmup=1, repeats=10, use_cuda_event=False)
        assert len(times) == 10
        # warmup(1) + repeats(10)
        assert counter[0] == 11

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
    def test_cuda_event_timing(self):
        """GPU 上跑一个空操作，CUDA event 计时应合理"""
        def fn():
            x = torch.randn(64, device="cuda")
            x.sum()
        times = benchmark_fn(fn, warmup=3, repeats=5, use_cuda_event=True)
        assert len(times) == 5
        assert all(t >= 0 for t in times)


class TestBenchmarkConfig:
    def test_cpu_config(self):
        """CPU 上跑配置 benchmark（模拟函数）"""
        cfg = KernelConfig(batch_size=1, seqlen_kv=1024)
        def fn():
            time.sleep(0.001)
        result = benchmark_config(fn, cfg, warmup=1, repeats=3, use_cuda_event=False)
        assert isinstance(result, BenchResult)
        assert result.cfg.batch_size == 1
        assert result.time_ms > 0
        assert result.repeats == 3
        assert result.achievable_bw_gb_s >= 0
        assert result.gap_to_roofline > 0
        assert result.bound_type == "memory-bound"

    def test_summary_string(self):
        cfg = KernelConfig(batch_size=1, seqlen_kv=1024)
        def fn():
            pass
        result = benchmark_config(fn, cfg, warmup=1, repeats=2, use_cuda_event=False)
        s = result.summary()
        assert "batch=1" in s
        assert "GB/s" in s
        assert "memory-bound" in s

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
    def test_gpu_real_benchmark(self):
        """GPU 真实小配置 benchmark"""
        cfg = KernelConfig(batch_size=1, seqlen_kv=512, headdim=128)
        x = torch.randn(1024, device="cuda")
        def fn():
            torch.matmul(x, x)
        result = benchmark_config(fn, cfg, warmup=3, repeats=5)
        assert result.time_ms > 0
        assert result.bandwidth_utilization >= 0


class TestCompareResults:
    def test_faster_candidate(self):
        cfg = KernelConfig(batch_size=1, seqlen_kv=1024)
        slow = BenchResult(cfg=cfg, time_ms=2.0, time_ms_median=2.0, time_ms_min=2.0,
                           time_ms_std=0, repeats=10, achievable_bw_gb_s=100,
                           bandwidth_utilization=0.05, gap_to_roofline=20, bound_type="memory-bound")
        fast = BenchResult(cfg=cfg, time_ms=1.0, time_ms_median=1.0, time_ms_min=1.0,
                          time_ms_std=0, repeats=10, achievable_bw_gb_s=200,
                          bandwidth_utilization=0.1, gap_to_roofline=10, bound_type="memory-bound")
        cmp = compare_results(slow, fast)
        assert cmp["speedup"] == pytest.approx(2.0)
        assert cmp["candidate_faster"]
        assert cmp["bw_gain"] == pytest.approx(2.0)

    def test_slower_candidate(self):
        cfg = KernelConfig(batch_size=1, seqlen_kv=1024)
        fast = BenchResult(cfg=cfg, time_ms=1.0, time_ms_median=1, time_ms_min=1,
                           time_ms_std=0, repeats=1, achievable_bw_gb_s=100,
                           bandwidth_utilization=0.05, gap_to_roofline=20, bound_type="memory-bound")
        slow = BenchResult(cfg=cfg, time_ms=3.0, time_ms_median=3, time_ms_min=3,
                           time_ms_std=0, repeats=1, achievable_bw_gb_s=33,
                           bandwidth_utilization=0.02, gap_to_roofline=60, bound_type="memory-bound")
        cmp = compare_results(fast, slow)
        assert not cmp["candidate_faster"]
        assert cmp["speedup"] == pytest.approx(1/3)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
class TestOfficialFlashAttn:
    """慢测试：真实官方 flash_attn baseline（默认不跑，手动触发）"""

    def test_official_baseline_runs(self):
        from agent_system.benchmark_engine import benchmark_official_flash_attn
        cfg = KernelConfig(batch_size=1, seqlen_kv=512, headdim=128)
        result = benchmark_official_flash_attn(cfg, warmup=3, repeats=10)
        assert result.time_ms > 0
        assert result.bandwidth_utilization < 1.0
        print(f"\n[Official baseline] {result.summary()}")
