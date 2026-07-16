"""
Real LLM Cost Model —— 用真实 LLM 做性能预测（创新点 C 真实实现）。

替代 mock predict_fn。
让 MiniMax-M2.7 扮演 cost model，预测候选的相对性能 + 置信度，
配合 roofline 物理校验（创新点 A 的物理刹车）。
"""
from __future__ import annotations

from typing import List

from agent_system.llm_client import chat, extract_json
from agent_system.llm_cost_model import Candidate
from agent_system.roofline_engine import is_physically_feasible


COST_MODEL_SYSTEM = """你是 GPU kernel 性能预测器（cost model）。
给定候选优化方案和当前性能，预测其相对加速比和你的置信度。

规则：
1. 只考虑 memory-bound 场景（attention decode）
2. split_k 提升并行度（小 batch 有效），但过多会增加 reduce 开销
3. 加速比不能超过物理上限（baseline 带宽利用率的倒数）

只输出JSON，格式：{"speedup": 数字, "confidence": 0到1, "reason": "一句话"}
"""


def predict(candidate: Candidate, baseline_util: float,
            cfg=None, model: str = "MiniMax-M2.7") -> tuple:
    """
    真实 LLM cost model：预测单个候选的 (speedup, confidence)。

    用 roofline 物理刹车 clip 防幻觉（创新点 A+C 融合）。
    """
    # 构造预测 prompt
    cfg_str = ""
    if cfg:
        cfg_str = f"配置: batch={cfg.batch_size}, seq_kv={cfg.seqlen_kv}, headdim={cfg.headdim}"

    prompt = f"""候选优化: {candidate.description}
参数: {candidate.params}
{cfg_str}
当前 baseline 带宽利用率: {baseline_util:.1%}
（物理上限约 {1/baseline_util:.1f}x 加速）

预测这个候选相对 baseline 的加速比。只输出JSON。"""

    try:
        resp = chat(
            [{"role": "system", "content": COST_MODEL_SYSTEM},
             {"role": "user", "content": prompt}],
            model=model, temperature=0.3, max_tokens=400,
        )
        result = extract_json(resp.content)
        speedup = float(result.get("speedup", 1.0))
        confidence = float(result.get("confidence", 0.5))
    except Exception:
        # LLM 失败 → 保守预测
        speedup, confidence = 1.0, 0.3

    # 创新点 A 物理刹车：roofline clip
    if not is_physically_feasible(speedup, baseline_util):
        speedup = min(speedup, 1.0 / baseline_util * 0.9)
        confidence *= 0.5  # 降低置信度（被物理修正）

    return speedup, confidence


def make_predict_fn(cfg=None, model: str = "MiniMax-M2.7"):
    """
    创建符合 two_stage_filter 签名的 predict_fn 闭包。

    用法：
        from agent_system.llm_cost_model import two_stage_filter
        survivors, stats = two_stage_filter(candidates, util, predict_fn=make_predict_fn(cfg))
    """
    def fn(candidate: Candidate, baseline_util: float) -> tuple:
        return predict(candidate, baseline_util, cfg=cfg, model=model)
    return fn
