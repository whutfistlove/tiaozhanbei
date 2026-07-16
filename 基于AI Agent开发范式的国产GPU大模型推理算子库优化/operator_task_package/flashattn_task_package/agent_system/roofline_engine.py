"""
Roofline Engine — 创新点 A 的物理基础。

MXC500 缺少 per-kernel profiler（实测 mx-smi 仅功耗监视器，无 NCU），
因此用 Roofline 物理模型作为零成本、恒定可靠的硬件反馈源，
替代 CudaForge/KernelAgent 所依赖的 NCU 实测指标。

提供：
- 理论下限耗时 T_h = max(FLOPs/peak_TFLOPS, bytes/peak_BW)
- 算术强度（arithmetic intensity）→ memory-bound vs compute-bound 判定
- 有效带宽利用率（实测带宽 / 理论峰值）
- gap_to_roofline（离物理上限的距离，优化空间量化）

参考：
- Roofline Model (Williams et al., CACM 2009)
- CudaForge (arXiv:2511.01884) 的硬件反馈闭环（本题因无NCU而改用roofline）
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ──────────────────────────────────────────────────────────────
# 沐曦 C500 / MXC500 硬件规格（来源：沐曦官网 metax-tech.com）
# ──────────────────────────────────────────────────────────────
# 显存：64 GB HBM2e，带宽 1.8 TB/s
# BF16 Tensor 算力：280 TFLOPS
# INT8：560 TOPS
C500_PEAK_BW_GB_S = 1800.0          # GB/s
C500_PEAK_BF16_TFLOPS = 280.0       # TFLOPS

# BF16 每元素字节数
BF16_BYTES_PER_ELEM = 2
FP32_BYTES_PER_ELEM = 4


@dataclass
class KernelConfig:
    """单个 run_kernel 调用的配置参数。"""
    batch_size: int
    seqlen_kv: int
    seqlen_q: int = 1
    num_heads: int = 8
    num_heads_k: int = 8
    headdim: int = 128
    page_block_size: int = 16
    dtype_bytes: int = BF16_BYTES_PER_ELEM

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.seqlen_kv <= 0:
            raise ValueError(f"seqlen_kv must be positive, got {self.seqlen_kv}")
        if self.headdim <= 0:
            raise ValueError(f"headdim must be positive, got {self.headdim}")


@dataclass
class RooflineResult:
    """Roofline 分析结果。"""
    flops: float                  # 总计算量（FLOP）
    bytes: float                  # 总访存量（Byte）
    arithmetic_intensity: float   # FLOP/Byte
    t_compute_s: float            # 算力下限（秒）
    t_memory_s: float             # 带宽下限（秒）
    t_lower_bound_s: float        # Roofline 理论下限 = max(t_compute, t_memory)
    bound_type: str               # "memory-bound" | "compute-bound" | "balanced"
    peak_bw_gb_s: float           # 理论峰值带宽
    peak_tflops: float            # 理论峰值算力


def calc_flops(cfg: KernelConfig) -> float:
    """
    FlashAttention decode（seqlen_q=1）的总 FLOPs。

    数学：O = softmax(Q @ K^T / sqrt(d)) @ V
    - Q@K^T: 一次 (1×d)·(d×seq_kv) 内积 → 2*d*seq_kv FLOPs/（head,batch）
    - P@V:   一次 (1×seq_kv)·(seq_kv×d) → 2*seq_kv*d FLOPs/（head,batch）
    - softmax 的 exp/sum/max 相对 GEMM 可忽略（O(seq_kv)，无 d 维）

    总计 = batch * heads * (QK + PV) = batch * heads * 2 * (2*d*seq_kv)
         = batch * heads * 4 * d * seq_kv
    """
    # 每段 GEMM 的 MAC 数 = d * seq_kv，每个 MAC = 2 FLOPs（乘+加）
    qk_flops = 2 * cfg.headdim * cfg.seqlen_kv
    pv_flops = 2 * cfg.seqlen_kv * cfg.headdim
    per_head = qk_flops + pv_flops
    total = cfg.batch_size * cfg.num_heads * per_head
    return float(total)


def calc_bytes(cfg: KernelConfig) -> float:
    """
    FlashAttention decode 的最小必须访存量（Byte）。

    Q 只读一次（seqlen_q=1，每 head 一条 d 向量）；
    K 和 V 各需读完整 cache（seq_kv 条向量/（head,batch））。
    这是 memory-bound 算子的理论下限（假设完美复用、无中间落盘）。
    """
    elem = cfg.dtype_bytes
    # Q: batch * seqlen_q * num_heads * headdim
    q_bytes = cfg.batch_size * cfg.seqlen_q * cfg.num_heads * cfg.headdim * elem
    # K+V: batch * seqlen_kv * num_heads_k * headdim * 2(K和V)
    kv_bytes = (
        cfg.batch_size * cfg.seqlen_kv * cfg.num_heads_k * cfg.headdim * elem * 2
    )
    # 输出 O: batch * seqlen_q * num_heads * headdim（写一次）
    o_bytes = cfg.batch_size * cfg.seqlen_q * cfg.num_heads * cfg.headdim * elem
    return float(q_bytes + kv_bytes + o_bytes)


def analyze(cfg: KernelConfig,
            peak_bw_gb_s: float = C500_PEAK_BW_GB_S,
            peak_tflops: float = C500_PEAK_BF16_TFLOPS) -> RooflineResult:
    """
    执行 Roofline 分析。

    返回理论下限耗时、算术强度、瓶颈类型。
    这是创新点 A 的核心：用物理先验替代缺失的 NCU profiler。
    """
    flops = calc_flops(cfg)
    bytes_total = calc_bytes(cfg)

    arithmetic_intensity = flops / bytes_total if bytes_total > 0 else float("inf")

    # 算力下限：FLOPs / (TFLOPS * 1e12)
    t_compute_s = flops / (peak_tflops * 1e12)
    # 带宽下限：Bytes / (GB/s * 1e9)
    t_memory_s = bytes_total / (peak_bw_gb_s * 1e9)

    t_lower_bound_s = max(t_compute_s, t_memory_s)

    # 瓶颈判定（C500 平衡点 = peak_tflops/peak_bw，单位 FLOP/Byte 需统一量纲）
    # 平衡点算术强度 = (TFLOPS*1e12) / (GB/s*1e9) = peak_tflops/peak_bw * 1e3
    balance_point = (peak_tflops * 1e12) / (peak_bw_gb_s * 1e9)
    if arithmetic_intensity < 0.5 * balance_point:
        bound_type = "memory-bound"
    elif arithmetic_intensity > 2.0 * balance_point:
        bound_type = "compute-bound"
    else:
        bound_type = "balanced"

    return RooflineResult(
        flops=flops,
        bytes=bytes_total,
        arithmetic_intensity=arithmetic_intensity,
        t_compute_s=t_compute_s,
        t_memory_s=t_memory_s,
        t_lower_bound_s=t_lower_bound_s,
        bound_type=bound_type,
        peak_bw_gb_s=peak_bw_gb_s,
        peak_tflops=peak_tflops,
    )


def achievable_bandwidth_gb_s(cfg: KernelConfig, measured_time_s: float) -> float:
    """由实测耗时反算有效带宽利用率对应的带宽值（GB/s）。"""
    bytes_total = calc_bytes(cfg)
    return (bytes_total / 1e9) / measured_time_s if measured_time_s > 0 else 0.0


def bandwidth_utilization(cfg: KernelConfig, measured_time_s: float,
                          peak_bw_gb_s: float = C500_PEAK_BW_GB_S) -> float:
    """
    有效带宽利用率 = 实测有效带宽 / 理论峰值。
    0.0~1.0+，>1.0 表示可能 bytes 估算偏低或测量异常。
    """
    eff_bw = achievable_bandwidth_gb_s(cfg, measured_time_s)
    return eff_bw / peak_bw_gb_s if peak_bw_gb_s > 0 else 0.0


def gap_to_roofline(cfg: KernelConfig, measured_time_s: float) -> float:
    """
    实测耗时 / Roofline 下限。1.0 = 已达物理极限。
    越大说明优化空间越大。这是 Judge 判定优化潜力的关键指标。
    """
    r = analyze(cfg)
    return measured_time_s / r.t_lower_bound_s if r.t_lower_bound_s > 0 else float("inf")


def is_physically_feasible(predicted_speedup: float,
                            baseline_utilization: float) -> bool:
    """
    创新点 A 的"物理刹车"：判断 LLM 预测的加速比是否物理可行。

    若 baseline 已达 X% 带宽利用率，则理论最大加速比 ≈ 1/X。
    超过此值则 LLM 预测可能幻觉（越过 roofline）。
    """
    if baseline_utilization <= 0:
        return predicted_speedup <= 1000.0  # 宽松兜底
    max_feasible = 1.0 / baseline_utilization * 1.1  # 允许 10% 容差
    return predicted_speedup <= max_feasible


def suggest_split_k(cfg: KernelConfig, num_sms: int = 96) -> int:
    """
    基于硬件 SM 数建议 Split-K 数（启发式，来自 FlashInfer/FlashDecoding）。

    目标：让总 threadblock 数 batch*heads*split ≈ num_sms，填满 GPU。
    """
    base_blocks = cfg.batch_size * cfg.num_heads
    if base_blocks >= num_sms:
        return 1
    ideal_split = max(1, num_sms // base_blocks)
    # 与 seq_kv 对齐（每 split 至少处理几个 page）
    max_useful = max(1, cfg.seqlen_kv // (cfg.page_block_size * 2))
    return min(ideal_split, max_useful)
