"""
LLM Cost Model（创新点 C）—— 双层候选过滤，降低 GPU 评测成本。

灵感：GPU Forecasters (arXiv:2605.31464) 的 selective surrogate + CompilerDream 的廉价 world model。
国产 GPU benchmark 昂贵，每个候选都跑真实 profiler 成本太高。

机制：
- 层1（物理过滤）：用 roofline 计算每个候选的理论可行性，剔除违反物理的
- 层2（LLM 预测）：让 LLM 预测候选相对 baseline 的预期加速比 + 置信度，roofline 上界 clip 防幻觉
- 仅对通过两层的少数候选（高置信 + 物理可行）跑真实 GPU benchmark

本模块实现层1（确定性，可单测）+ 层2 的调度框架。
LLM 的实际预测通过可注入的 predict_fn，测试时用 mock。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from agent_system.roofline_engine import (
    KernelConfig, analyze, is_physically_feasible,
)


@dataclass
class Candidate:
    """一个优化候选 kernel 描述。"""
    candidate_id: str
    description: str             # 自然语言描述（如 "split_k=4, tile_n=32"）
    params: dict                 # 结构化参数
    # LLM 预测（层2 填充）
    predicted_speedup: Optional[float] = None
    confidence: Optional[float] = None
    # 物理校验（层1 填充）
    physically_feasible: Optional[bool] = None
    feasibility_reason: str = ""
    # 过滤决策
    passed_filter: bool = False
    filter_reason: str = ""


@dataclass
class FilterStats:
    """双层过滤的统计（用于校准 cost model 准确度）。"""
    total: int = 0
    rejected_by_physics: int = 0
    rejected_by_confidence: int = 0
    survivors: int = 0
    survival_rate: float = 0.0


# LLM 预测函数签名：(candidate, baseline_util) -> (predicted_speedup, confidence)
PredictFn = Callable[[Candidate, float], tuple]


def physics_filter(candidate: Candidate, baseline_util: float) -> tuple[bool, str]:
    """
    层1：物理过滤（创新点 A 的物理刹车）。

    用 roofline 上界判断候选描述里的预期加速比是否物理可行。
    若候选没给 predicted_speedup，则检查其参数是否明显违反约束。
    """
    # 若已有 LLM 预测的加速比，用 roofline 校验
    if candidate.predicted_speedup is not None:
        if not is_physically_feasible(candidate.predicted_speedup, baseline_util):
            return False, (
                f"预测加速比 {candidate.predicted_speedup:.1f}x 违反 roofline "
                f"（baseline 已用 {baseline_util:.0%} 带宽，理论上限 ~{1/baseline_util:.1f}x）"
            )
    # 检查参数基本约束
    params = candidate.params
    split_k = params.get("split_k", 1)
    if split_k < 1:
        return False, "split_k 必须 >= 1"
    tile_n = params.get("tile_n", 16)
    if tile_n <= 0 or tile_n > 256:
        return False, f"tile_n={tile_n} 越界（应在 1~256）"
    return True, "物理可行"


def two_stage_filter(
    candidates: List[Candidate],
    baseline_util: float,
    predict_fn: Optional[PredictFn] = None,
    min_confidence: float = 0.5,
    max_survivors: Optional[int] = None,
) -> tuple[List[Candidate], FilterStats]:
    """
    双层过滤主入口（创新点 C）。

    1. 若提供 predict_fn：先让 LLM 预测每个候选（层2预判）
    2. 物理过滤：用 roofline 校验（层1）
    3. 置信度过滤：保留 >= min_confidence 的
    4. 限流：最多保留 max_survivors 个（按置信度排序）

    返回 (幸存候选列表, 统计)。
    """
    stats = FilterStats(total=len(candidates))

    # 层2 预判（若提供 LLM）
    if predict_fn is not None:
        for c in candidates:
            speedup, conf = predict_fn(c, baseline_util)
            c.predicted_speedup = speedup
            c.confidence = conf

    # 层1 物理过滤
    for c in candidates:
        ok, reason = physics_filter(c, baseline_util)
        c.physically_feasible = ok
        c.feasibility_reason = reason
        if not ok:
            stats.rejected_by_physics += 1

    # 置信度过滤
    survivors = []
    for c in candidates:
        if not c.physically_feasible:
            c.passed_filter = False
            c.filter_reason = "物理不可行: " + c.feasibility_reason
            continue
        if c.confidence is not None and c.confidence < min_confidence:
            stats.rejected_by_confidence += 1
            c.passed_filter = False
            c.filter_reason = f"置信度 {c.confidence:.2f} < {min_confidence}"
            continue
        c.passed_filter = True
        c.filter_reason = "通过双层过滤"
        survivors.append(c)

    # 限流（按置信度降序）
    if max_survivors is not None and len(survivors) > max_survivors:
        survivors.sort(key=lambda c: c.confidence or 0, reverse=True)
        for c in survivors[max_survivors:]:
            c.passed_filter = False
            c.filter_reason = f"置信度排名超出 top-{max_survivors}"
        survivors = survivors[:max_survivors]

    stats.survivors = len(survivors)
    stats.survival_rate = len(survivors) / len(candidates) if candidates else 0.0
    return survivors, stats


# ── 校准机制 ──
@dataclass
class PredictionRecord:
    """单条预测记录，用于校准 cost model。"""
    candidate_id: str
    predicted_speedup: float
    actual_speedup: float
    error: float                # |predicted - actual|
    correct_direction: bool     # 预测方向（>1/<1）是否正确


def calibrate(records: List[PredictionRecord]) -> dict:
    """
    校准 LLM cost model 的准确度（GPU Forecasters 的 selective surrogate 思想）。

    返回：平均误差、方向准确率、是否建议退化到全量评测。
    """
    if not records:
        return {"mae": 0, "direction_accuracy": 0, "degrade_to_full": False}
    errors = [r.error for r in records]
    directions = [r.correct_direction for r in records]
    mae = sum(errors) / len(errors)
    dir_acc = sum(directions) / len(directions)
    # 若方向准确率 < 50%（比随机还差），建议退化
    degrade = dir_acc < 0.5
    return {"mae": mae, "direction_accuracy": dir_acc, "degrade_to_full": degrade}
