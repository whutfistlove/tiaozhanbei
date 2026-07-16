"""
roofline_engine 的单元测试。

测试覆盖：
- FLOPs / Bytes 计算正确性
- 算术强度与 memory-bound 判定（核心：decode 必须 memory-bound）
- Roofline 下限计算
- 带宽利用率 / gap_to_roofline
- 物理可行性（创新点 A 的"物理刹车"）
- split-k 启发式
- 边界条件与异常

运行：pytest tests/test_roofline_engine.py -v
"""
import math
import pytest

from agent_system.roofline_engine import (
    KernelConfig,
    RooflineResult,
    calc_flops,
    calc_bytes,
    analyze,
    achievable_bandwidth_gb_s,
    bandwidth_utilization,
    gap_to_roofline,
    is_physically_feasible,
    suggest_split_k,
    C500_PEAK_BW_GB_S,
    C500_PEAK_BF16_TFLOPS,
)


# ──────────────────────────────────────────────────────────────
# FLOPs / Bytes 计算
# ──────────────────────────────────────────────────────────────

class TestFlopsAndBytes:
    def test_flops_basic_decode(self):
        """decode：FLOPs = batch * heads * 4 * d * seq_kv"""
        cfg = KernelConfig(batch_size=1, seqlen_kv=1024, headdim=128, num_heads=8)
        expected = 1 * 8 * 4 * 128 * 1024
        assert calc_flops(cfg) == expected

    def test_flops_scales_with_batch(self):
        """FLOPs 与 batch 线性相关"""
        base = calc_flops(KernelConfig(batch_size=1, seqlen_kv=4096))
        b16 = calc_flops(KernelConfig(batch_size=16, seqlen_kv=4096))
        assert b16 == pytest.approx(16 * base)

    def test_flops_scales_with_seqkv(self):
        base = calc_flops(KernelConfig(batch_size=4, seqlen_kv=1024))
        big = calc_flops(KernelConfig(batch_size=4, seqlen_kv=8192))
        assert big == pytest.approx(8 * base)

    def test_bytes_qkv_partition(self):
        """KV cache 占主导（decode 特征）"""
        cfg = KernelConfig(batch_size=1, seqlen_kv=1024, headdim=128, num_heads=8)
        b = calc_bytes(cfg)
        # Q 很小（1 token），KV 大（1024 token × 2）
        kv_bytes = 1 * 1024 * 8 * 128 * 2 * 2  # batch*seq*heads*d*2(KV)*bf16
        assert b > kv_bytes  # 还含 Q 和 O
        assert kv_bytes / b > 0.9  # KV 占绝大多数

    def test_bytes_bf16_default(self):
        cfg = KernelConfig(batch_size=2, seqlen_kv=2048, headdim=64)
        assert cfg.dtype_bytes == 2  # bf16


# ──────────────────────────────────────────────────────────────
# Roofline 分析（核心：decode 必须 memory-bound）
# ──────────────────────────────────────────────────────────────

class TestRooflineAnalysis:
    def test_decode_is_memory_bound(self):
        """关键断言：所有 OJ 评测配置都是 memory-bound"""
        for batch in [1, 4, 16]:
            for seq in [1024, 4096, 8192, 16384]:
                cfg = KernelConfig(batch_size=batch, seqlen_kv=seq)
                r = analyze(cfg)
                assert r.bound_type == "memory-bound", (
                    f"batch={batch},seq={seq} 应 memory-bound，实际 {r.bound_type}"
                )

    def test_arithmetic_intensity_low(self):
        """decode 算术强度远低于 C500 平衡点（~155 FLOP/Byte）"""
        cfg = KernelConfig(batch_size=1, seqlen_kv=4096)
        r = analyze(cfg)
        balance = (C500_PEAK_BF16_TFLOPS * 1e12) / (C500_PEAK_BW_GB_S * 1e9)
        assert r.arithmetic_intensity < 0.5 * balance
        # decode 算术强度应该在个位数（约 2 FLOP/Byte 量级）
        assert r.arithmetic_intensity < 10

    def test_lower_bound_is_memory(self):
        """memory-bound 时下限由带宽决定"""
        cfg = KernelConfig(batch_size=1, seqlen_kv=8192)
        r = analyze(cfg)
        assert r.t_memory_s > r.t_compute_s
        assert r.t_lower_bound_s == pytest.approx(r.t_memory_s)

    def test_lower_bound_positive(self):
        for batch in [1, 16]:
            for seq in [1024, 16384]:
                cfg = KernelConfig(batch_size=batch, seqlen_kv=seq)
                r = analyze(cfg)
                assert r.t_lower_bound_s > 0

    def test_result_fields_complete(self):
        cfg = KernelConfig(batch_size=1, seqlen_kv=1024)
        r = analyze(cfg)
        assert r.flops > 0
        assert r.bytes > 0
        assert r.arithmetic_intensity > 0
        assert r.bound_type in ("memory-bound", "compute-bound", "balanced")
        assert r.peak_bw_gb_s == C500_PEAK_BW_GB_S
        assert r.peak_tflops == C500_PEAK_BF16_TFLOPS


# ──────────────────────────────────────────────────────────────
# 带宽利用率与 gap_to_roofline
# ──────────────────────────────────────────────────────────────

class TestBandwidthMetrics:
    def test_bandwidth_from_time(self):
        cfg = KernelConfig(batch_size=1, seqlen_kv=4096)
        # 假设耗时恰好等于带宽下限 → 利用率应为 1.0
        r = analyze(cfg)
        util = bandwidth_utilization(cfg, r.t_memory_s)
        assert util == pytest.approx(1.0, abs=0.02)

    def test_achievable_bandwidth_positive(self):
        cfg = KernelConfig(batch_size=1, seqlen_kv=4096)
        bw = achievable_bandwidth_gb_s(cfg, measured_time_s=0.001)
        assert bw > 0

    def test_gap_one_at_roofline(self):
        """实测=下限时 gap=1.0"""
        cfg = KernelConfig(batch_size=1, seqlen_kv=8192)
        r = analyze(cfg)
        gap = gap_to_roofline(cfg, r.t_lower_bound_s)
        assert gap == pytest.approx(1.0, abs=0.02)

    def test_gap_large_when_slow(self):
        """实测远大于下限时 gap 很大（优化空间大）"""
        cfg = KernelConfig(batch_size=1, seqlen_kv=8192)
        r = analyze(cfg)
        slow_time = r.t_lower_bound_s * 10  # 慢 10 倍
        gap = gap_to_roofline(cfg, slow_time)
        assert gap == pytest.approx(10.0, abs=0.1)

    def test_zero_time_safe(self):
        """耗时为 0 不应崩溃"""
        cfg = KernelConfig(batch_size=1, seqlen_kv=1024)
        assert achievable_bandwidth_gb_s(cfg, 0.0) == 0.0


# ──────────────────────────────────────────────────────────────
# 物理可行性（创新点 A 的"物理刹车"）
# ──────────────────────────────────────────────────────────────

class TestPhysicalFeasibility:
    def test_feasible_speedup(self):
        """baseline 用 45% 带宽，预测 2x 加速（→90%）物理可行"""
        assert is_physically_feasible(predicted_speedup=2.0, baseline_utilization=0.45)

    def test_infeasible_speedup(self):
        """预测 10x 但 baseline 已用 45% → 物理不可行（超 roofline）"""
        assert not is_physically_feasible(predicted_speedup=10.0, baseline_utilization=0.45)

    def test_at_boundary(self):
        """刚好 1/0.45≈2.22x 在容差内可行"""
        assert is_physically_feasible(predicted_speedup=2.3, baseline_utilization=0.45)

    def test_zero_utilization_safe(self):
        assert is_physically_feasible(predicted_speedup=5.0, baseline_utilization=0.0)


# ──────────────────────────────────────────────────────────────
# Split-K 启发式
# ──────────────────────────────────────────────────────────────

class TestSplitKSuggestion:
    def test_small_batch_suggests_split(self):
        """batch=1 时应建议 split（SM 欠载）"""
        cfg = KernelConfig(batch_size=1, seqlen_kv=8192)
        s = suggest_split_k(cfg, num_sms=96)
        assert s > 1  # batch=1*heads=8=8 blocks，96/8=12

    def test_large_batch_no_split(self):
        """batch 足够大时不 split"""
        cfg = KernelConfig(batch_size=128, seqlen_kv=1024)
        s = suggest_split_k(cfg, num_sms=96)
        assert s == 1  # 128*8 >> 96

    def test_split_bounded_by_seqkv(self):
        """split 数不超过 seq_kv 能支撑的范围"""
        cfg = KernelConfig(batch_size=1, seqlen_kv=64)  # 很短
        s = suggest_split_k(cfg, num_sms=96)
        assert s >= 1


# ──────────────────────────────────────────────────────────────
# 边界与异常
# ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_invalid_batch_raises(self):
        with pytest.raises(ValueError):
            KernelConfig(batch_size=0, seqlen_kv=1024)

    def test_invalid_seqkv_raises(self):
        with pytest.raises(ValueError):
            KernelConfig(batch_size=1, seqlen_kv=0)

    def test_invalid_headdim_raises(self):
        with pytest.raises(ValueError):
            KernelConfig(batch_size=1, seqlen_kv=1024, headdim=0)

    def test_negative_batch_raises(self):
        with pytest.raises(ValueError):
            KernelConfig(batch_size=-1, seqlen_kv=1024)
