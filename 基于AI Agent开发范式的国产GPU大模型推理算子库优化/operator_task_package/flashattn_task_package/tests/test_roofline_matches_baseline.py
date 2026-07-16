"""
Roofline 与实测 baseline 的吻合度验证（创新点 A 的闭环验证）。

读取已有的 benchmark CSV（官方 flash_attn_with_kvcache 跑出的真实数据），
对每个配置计算 roofline 理论值，验证：
1. 所有配置确实 memory-bound（理论判定）
2. 官方 baseline 带宽利用率在合理范围（应 < 100%，远未榨干）
3. gap_to_roofline 量化了优化空间

这一步把"理论 roofline"与"实测 baseline"对齐，是创新点 A 立得住的关键。
"""
import csv
import os
import sys
from pathlib import Path

import pytest

# 让测试能 import agent_system
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_system.roofline_engine import (
    KernelConfig,
    analyze,
    bandwidth_utilization,
    gap_to_roofline,
    C500_PEAK_BW_GB_S,
)

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmark"


def _load_csv():
    """加载最新一份 benchmark CSV。"""
    csvs = sorted(BENCH_DIR.glob("benchmark_kvcache_*.csv"))
    if not csvs:
        return []
    rows = []
    with open(csvs[-1]) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "batch_size": int(r["batch_size"]),
                    "seq_len_kv": int(r["seq_len_kv"]),
                    "heads": int(r["heads"]),
                    "headdim": int(r["headdim"]),
                    "time_ms": float(r["time_ms"]),
                    "bandwidth_GB_s": float(r["bandwidth_GB_s"]),
                })
            except (ValueError, KeyError):
                continue  # 跳过 OOM 行
    return rows


@pytest.fixture(scope="module")
def bench_rows():
    rows = _load_csv()
    if not rows:
        pytest.skip("无 benchmark CSV 数据，跳过 roofline 吻合度验证")
    return rows


class TestRooflineMatchesBaseline:
    """验证 roofline 理论与官方 baseline 实测吻合。"""

    def test_all_configs_memory_bound(self, bench_rows):
        """所有实测配置理论上都应是 memory-bound。"""
        non_mb = []
        for r in bench_rows:
            cfg = KernelConfig(
                batch_size=r["batch_size"],
                seqlen_kv=r["seq_len_kv"],
                headdim=r["headdim"],
                num_heads=r["heads"],
                num_heads_k=r["heads"],
            )
            res = analyze(cfg)
            if res.bound_type != "memory-bound":
                non_mb.append((r, res.bound_type))
        assert not non_mb, f"这些配置理论判定非 memory-bound：{non_mb}"

    def test_baseline_below_peak(self, bench_rows):
        """官方 baseline 带宽必须低于理论峰值（< 100%）。"""
        for r in bench_rows:
            assert r["bandwidth_GB_s"] < C500_PEAK_BW_GB_S, (
                f"batch={r['batch_size']},seq={r['seq_len_kv']} "
                f"实测带宽 {r['bandwidth_GB_s']} 超过峰值 {C500_PEAK_BW_GB_S}，测量异常"
            )

    def test_gap_to_roofline_quantifies_headroom(self, bench_rows):
        """gap_to_roofline > 1 表明有优化空间（baseline 未达物理极限）。"""
        headroom_configs = []
        for r in bench_rows:
            cfg = KernelConfig(
                batch_size=r["batch_size"],
                seqlen_kv=r["seq_len_kv"],
                headdim=r["headdim"],
                num_heads=r["heads"],
                num_heads_k=r["heads"],
            )
            gap = gap_to_roofline(cfg, r["time_ms"] / 1e3)
            # gap > 1.5 表示至少有 1.5x 优化空间
            if gap > 1.5:
                headroom_configs.append({
                    "batch": r["batch_size"], "seq": r["seq_len_kv"],
                    "gap": round(gap, 2),
                })
        # 至少应有一些配置有优化空间（除非 baseline 已完美）
        assert len(headroom_configs) > 0, "没有配置有优化空间，baseline 可能已接近极限"
        print(f"\n[Roofline验证] {len(headroom_configs)}/{len(bench_rows)} 个配置 gap>1.5（有优化空间）")

    def test_small_batch_has_largest_headroom(self, bench_rows):
        """小 batch 应有最大优化空间（SM 欠载，Split-K 收益最大）。"""
        seq_fixed = 8192
        gaps_by_batch = {}
        for r in bench_rows:
            if r["seq_len_kv"] != seq_fixed:
                continue
            cfg = KernelConfig(
                batch_size=r["batch_size"], seqlen_kv=r["seq_len_kv"],
                headdim=r["headdim"], num_heads=r["heads"], num_heads_k=r["heads"],
            )
            gaps_by_batch[r["batch_size"]] = gap_to_roofline(cfg, r["time_ms"] / 1e3)
        if len(gaps_by_batch) < 2:
            pytest.skip("seq=8192 的配置不足")
        # batch=1 的 gap 应大于 batch=128
        b1 = gaps_by_batch.get(1)
        b128 = gaps_by_batch.get(128)
        if b1 and b128:
            assert b1 > b128, (
                f"batch=1 gap={b1:.1f} 应大于 batch=128 gap={b128:.1f}（小 batch SM 更欠载）"
            )
