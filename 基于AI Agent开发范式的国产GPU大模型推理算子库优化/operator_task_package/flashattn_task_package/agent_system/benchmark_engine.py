"""
Benchmark 引擎（Profiler 角色的核心工具）。

封装：
- GPU 计时（warmup + repeat，CUDA event 精确计时）
- 有效带宽计算
- 与官方 flash_attn_with_kvcache baseline 的对比
- 针对任意 run_kernel 可调用对象

设计为可测试：核心计时逻辑可在 CPU 模拟函数上验证。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from agent_system.roofline_engine import (
    KernelConfig,
    achievable_bandwidth_gb_s,
    bandwidth_utilization,
    gap_to_roofline,
    analyze,
)


@dataclass
class BenchResult:
    """单次 benchmark 结果。"""
    cfg: KernelConfig
    time_ms: float                  # 平均 kernel 时间（毫秒）
    time_ms_median: float           # 中位数
    time_ms_min: float              # 最小值
    time_ms_std: float              # 标准差
    repeats: int
    achievable_bw_gb_s: float       # 有效带宽
    bandwidth_utilization: float    # 带宽利用率（0~1）
    gap_to_roofline: float          # 离物理极限的距离（1.0=已达极限）
    bound_type: str                 # memory/compute-bound

    def summary(self) -> str:
        return (
            f"batch={self.cfg.batch_size} seq_kv={self.cfg.seqlen_kv} | "
            f"{self.time_ms:.4f}ms (min {self.time_ms_min:.4f}) | "
            f"bw={self.achievable_bw_gb_s:.1f} GB/s ({self.bandwidth_utilization*100:.1f}%) | "
            f"gap={self.gap_to_roofline:.2f}x [{self.bound_type}]"
        )


def benchmark_fn(
    fn: Callable[[], None],
    warmup: int = 10,
    repeats: int = 100,
    use_cuda_event: bool = True,
) -> List[float]:
    """
    通用 GPU 计时：返回每次调用的耗时（毫秒）列表。

    - use_cuda_event=True：用 CUDA event 精确计时（推荐，GPU 端）
    - use_cuda_event=False：用 CPU 时间（仅用于无 GPU 环境的测试）
    """
    if not callable(fn):
        raise TypeError("fn 必须是可调用对象")

    # warmup
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times_ms: List[float] = []

    if use_cuda_event and torch.cuda.is_available():
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
    else:
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1e3)

    return times_ms


def _stats(times_ms: List[float]) -> dict:
    t = torch.tensor(times_ms)
    return {
        "mean": t.float().mean().item(),
        "median": t.float().median().item(),
        "min": t.float().min().item(),
        "std": t.float().std(unbiased=False).item() if len(times_ms) > 1 else 0.0,
    }


def benchmark_config(
    fn: Callable[[], None],
    cfg: KernelConfig,
    warmup: int = 10,
    repeats: int = 100,
    use_cuda_event: bool = True,
) -> BenchResult:
    """
    对一个配置跑完整 benchmark，返回含 roofline 分析的完整结果。

    这是创新点 A 的运行时：benchmark 后立即用 roofline 量化优化空间。
    """
    times = benchmark_fn(fn, warmup=warmup, repeats=repeats, use_cuda_event=use_cuda_event)
    s = _stats(times)
    time_s = s["mean"] / 1e3

    r = analyze(cfg)
    bw = achievable_bandwidth_gb_s(cfg, time_s)
    util = bandwidth_utilization(cfg, time_s)
    gap = gap_to_roofline(cfg, time_s)

    return BenchResult(
        cfg=cfg,
        time_ms=s["mean"],
        time_ms_median=s["median"],
        time_ms_min=s["min"],
        time_ms_std=s["std"],
        repeats=repeats,
        achievable_bw_gb_s=bw,
        bandwidth_utilization=util,
        gap_to_roofline=gap,
        bound_type=r.bound_type,
    )


def compare_results(baseline: BenchResult, candidate: BenchResult) -> dict:
    """对比 baseline 和优化候选，输出加速比等指标（Judge 用）。"""
    speedup = baseline.time_ms / candidate.time_ms if candidate.time_ms > 0 else float("inf")
    bw_gain = candidate.achievable_bw_gb_s / baseline.achievable_bw_gb_s if baseline.achievable_bw_gb_s > 0 else float("inf")
    return {
        "speedup": speedup,
        "bw_gain": bw_gain,
        "time_baseline_ms": baseline.time_ms,
        "time_candidate_ms": candidate.time_ms,
        "util_baseline": baseline.bandwidth_utilization,
        "util_candidate": candidate.bandwidth_utilization,
        "candidate_faster": candidate.time_ms < baseline.time_ms,
    }


def benchmark_official_flash_attn(
    cfg: KernelConfig,
    warmup: int = 10,
    repeats: int = 100,
) -> BenchResult:
    """
    用官方 flash_attn_with_kvcache 跑 baseline（与 benchmark/benchmark_kvcache.py 一致）。

    这是建立性能基线 $T_b$（50 分对应）的工具。
    """
    from flash_attn.flash_attn_interface import flash_attn_with_kvcache
    from einops import rearrange

    num_blocks = max(1024, math.ceil(cfg.seqlen_kv / cfg.page_block_size) * cfg.batch_size * 3)
    paged_bs = cfg.page_block_size

    torch.manual_seed(0)
    q = torch.randn(cfg.batch_size, cfg.seqlen_q, cfg.num_heads, cfg.headdim,
                    device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(num_blocks, paged_bs, cfg.num_heads_k, cfg.headdim,
                          device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(num_blocks, paged_bs, cfg.num_heads_k, cfg.headdim,
                          device="cuda", dtype=torch.bfloat16)
    block_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device="cuda"),
        "(b n) -> b n", b=cfg.batch_size,
    )
    cache_seqlens = torch.full((cfg.batch_size,), cfg.seqlen_kv, dtype=torch.int32, device="cuda")

    def run():
        flash_attn_with_kvcache(
            q, k_cache, v_cache, None, None,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=None,
            block_table=block_table,
            causal=False,
            window_size=(-1, -1),
            rotary_interleaved=False,
            alibi_slopes=None,
            num_splits=1,
        )

    return benchmark_config(run, cfg, warmup=warmup, repeats=repeats)
