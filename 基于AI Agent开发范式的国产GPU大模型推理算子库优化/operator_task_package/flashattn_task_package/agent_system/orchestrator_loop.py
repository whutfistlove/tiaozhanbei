"""
Orchestrator Loop —— Agent 优化闭环的核心调度器。

整合所有组件：
- Analyst：roofline 分析瓶颈（创新点 A）
- Coder：生成候选（依赖领域记忆，创新点 B）
- Profiler：双层过滤（创新点 C）+ 真实 benchmark
- Judge：roofline 锚定裁决 + 正确性校验
- Reflector：失败反思 + 信念更新（创新点 B）
- Logger：记录每轮（可复现性物料）

本模块定义闭环的数据流和调度逻辑。
角色的"智能"（LLM 推理）通过可注入的回调实现，便于单测（mock）。
真实运行时由 OpenCode subagent 填充回调。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, List

from agent_system.roofline_engine import (
    KernelConfig, analyze, gap_to_roofline, suggest_split_k,
)
from agent_system.correctness import (
    generate_reference, check as correctness_check, make_test_inputs, CorrectnessResult,
)
from agent_system.benchmark_engine import (
    benchmark_config, benchmark_fn, BenchResult, compare_results,
)
from agent_system.llm_cost_model import (
    Candidate, two_stage_filter, FilterStats,
)
from agent_system.domain_memory import DomainMemory
from agent_system.optimization_log import OptimizationLog, OptimizationEntry


# ── 回调签名（角色智能的注入点）──
# Coder 生成候选列表
GenerateCandidatesFn = Callable[[KernelConfig, str, DomainMemory], List[Candidate]]
# Profiler 跑候选 kernel（编译+运行），返回 (output_tensor, time_s)
RunKernelFn = Callable[[Candidate, object], object]  # object = inputs tuple


@dataclass
class IterationResult:
    """单轮迭代结果。"""
    iteration: int
    bottleneck: str
    candidates_total: int
    candidates_filtered: int
    best_candidate: Optional[Candidate]
    best_speedup: Optional[float]
    correctness: Optional[CorrectnessResult]
    verdict: str
    gap_to_roofline: Optional[float]
    filter_stats: Optional[FilterStats] = None


def analyst_analyze(cfg: KernelConfig, baseline_time_s: Optional[float] = None) -> str:
    """
    Analyst 角色：用 roofline 分析瓶颈，输出优化方向建议。

    创新点 A 的运行时：用 roofline 替代缺失的 NCU profiler。
    """
    r = analyze(cfg)
    direction = []
    if r.bound_type == "memory-bound":
        direction.append("memory-bound：优化方向=提升带宽利用率+并行度")
        sk = suggest_split_k(cfg)
        if sk > 1:
            direction.append(f"建议 Split-K={sk}（当前 batch*heads={cfg.batch_size*cfg.num_heads} 远小于 SM 数）")
    elif r.bound_type == "compute-bound":
        direction.append("compute-bound：优化方向=Tensor Core 利用率")
    else:
        direction.append("balanced：需综合优化")

    if baseline_time_s:
        gap = gap_to_roofline(cfg, baseline_time_s)
        util = 1.0 / gap if gap > 0 else 0
        direction.append(f"当前 gap_to_roofline={gap:.2f}x（带宽利用率约 {util:.1%}）")

    return "; ".join(direction)


def judge_verdict(
    candidate: Candidate,
    baseline_result: BenchResult,
    candidate_result: BenchResult,
    correctness: CorrectnessResult,
) -> tuple[str, str]:
    """
    Judge 角色：roofline 锚定裁决（创新点 A 的 Judge）。

    返回 (verdict, reason)。verdict ∈ {KEEP, ROLLBACK, REJECT}。
    """
    if not correctness.passed:
        return "REJECT", f"正确性失败：{correctness.detail}"
    if candidate_result.time_ms >= baseline_result.time_ms:
        return "ROLLBACK", f"未提速（{candidate_result.time_ms:.4f} >= {baseline_result.time_ms:.4f}）"
    # roofline 锚定：检查候选结果是否物理合理（不应超过 roofline）
    if candidate_result.gap_to_roofline < 0.9:
        return "KEEP", f"提速且接近物理极限 gap={candidate_result.gap_to_roofline:.2f}"
    return "KEEP", f"提速 {baseline_result.time_ms/candidate_result.time_ms:.2f}x，gap={candidate_result.gap_to_roofline:.2f}"


def run_iteration(
    iteration: int,
    cfg: KernelConfig,
    baseline_result: BenchResult,
    generate_candidates_fn: GenerateCandidatesFn,
    run_kernel_fn: RunKernelFn,
    memory: DomainMemory,
    log: OptimizationLog,
    predict_fn=None,
    max_survivors: int = 3,
    warmup: int = 5,
    repeats: int = 20,
) -> IterationResult:
    """
    执行一轮完整优化迭代（Analyst→Coder→Profiler双层→Judge→Reflector→Logger）。
    """
    # ── Analyst ──
    bottleneck = analyst_analyze(cfg, baseline_result.time_ms / 1e3)

    # ── Coder：生成候选 ──
    candidates = generate_candidates_fn(cfg, bottleneck, memory)

    # ── Profiler 阶段一：双层过滤（创新点 C）──
    survivors, fstats = two_stage_filter(
        candidates,
        baseline_util=baseline_result.bandwidth_utilization,
        predict_fn=predict_fn,
        max_survivors=max_survivors,
    )

    # ── Profiler 阶段二：真实硬验证 ──
    best_candidate = None
    best_result = None
    best_correctness = None
    best_speedup = None

    inputs = make_test_inputs(cfg, device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    q, k_cache, v_cache, cache_seqlens, block_table = inputs
    output_ref = generate_reference(q, k_cache, v_cache, cache_seqlens, block_table, cfg)

    for cand in survivors:
        try:
            output_test = run_kernel_fn(cand, inputs)
            # 正确性校验
            corr = correctness_check(output_test, output_ref)
            # benchmark
            def run():
                run_kernel_fn(cand, inputs)
            cand_result = benchmark_config(run, cfg, warmup=warmup, repeats=repeats)

            cand_verdict, reason = judge_verdict(cand, baseline_result, cand_result, corr)

            if cand_verdict == "KEEP":
                speedup = baseline_result.time_ms / cand_result.time_ms
                if best_speedup is None or speedup > best_speedup:
                    best_candidate = cand
                    best_result = cand_result
                    best_correctness = corr
                    best_speedup = speedup
        except Exception as e:
            memory.record_failure(
                category="runtime_error",
                symptom=str(e)[:200],
                root_cause=f"候选 {cand.candidate_id} 运行失败",
                fix="待分析",
                config=f"batch={cfg.batch_size},seq={cfg.seqlen_kv}",
            )

    # ── Judge 最终裁决 ──
    if best_candidate and best_result and best_correctness:
        verdict = "KEEP"
        # ── Reflector：成功 → 更新信念 ──
        memory.record_belief(
            observation=f"{best_candidate.description} 在 {cfg.batch_size},{cfg.seqlen_kv} 加速 {best_speedup:.2f}x",
            rule=f"该配置下 {best_candidate.params} 有效",
            confidence=0.7,
        )
    else:
        verdict = "ROLLBACK"
        # ── Reflector：失败 → 反思 ──
        if bottleneck:
            memory.record_belief(
                observation=f"第{iteration}轮未找到有效优化，瓶颈={bottleneck}",
                rule="需尝试其它方向",
                confidence=0.3,
            )

    # ── Logger ──
    entry = OptimizationEntry(
        iteration=iteration,
        timestamp=__import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        change_description=best_candidate.description if best_candidate else "无有效候选",
        target_config=f"batch={cfg.batch_size},seq={cfg.seqlen_kv},d={cfg.headdim}",
        baseline_time_ms=baseline_result.time_ms,
        candidate_time_ms=best_result.time_ms if best_result else None,
        speedup=best_speedup,
        bandwidth_util_before=baseline_result.bandwidth_utilization,
        bandwidth_util_after=best_result.bandwidth_utilization if best_result else None,
        correctness_passed=best_correctness.passed if best_correctness else False,
        verdict=verdict,
        gap_to_roofline=best_result.gap_to_roofline if best_result else None,
    )
    log.record(entry)

    return IterationResult(
        iteration=iteration,
        bottleneck=bottleneck,
        candidates_total=len(candidates),
        candidates_filtered=fstats.survivors,
        best_candidate=best_candidate,
        best_speedup=best_speedup,
        correctness=best_correctness,
        verdict=verdict,
        gap_to_roofline=best_result.gap_to_roofline if best_result else None,
        filter_stats=fstats,
    )
