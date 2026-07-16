"""
Real Orchestrator —— 真实优化迭代闭环（完整实现）。

替代 mock 的 orchestrator_loop，用真实的：
- LLM Coder（MiniMax-M2.7 生成 kernel 代码）
- mxcc 编译（kernel_compiler）
- ctypes 运行（kernel_loader）
- 正确性校验（correctness）
- GPU benchmark（benchmark_engine）
- LLM cost model（real_cost_model）
- 领域记忆（domain_memory）
- 优化日志（optimization_log）

完整流程：
LLM生成候选 → 双层过滤 → mxcc编译 → GPU运行 → 正确性校验 → benchmark → Judge裁决 → 记忆/日志更新

这是"原型"到"完整实现"的最终形态。
"""
from __future__ import annotations

import json
import inspect
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional

import torch

from agent_system.roofline_engine import KernelConfig, analyze, suggest_split_k
from agent_system.correctness import make_test_inputs, generate_reference, check as correctness_check
from agent_system.benchmark_engine import benchmark_config, BenchResult
from agent_system.llm_cost_model import Candidate, two_stage_filter
from agent_system.domain_memory import DomainMemory
from agent_system.optimization_log import OptimizationLog, OptimizationEntry
from agent_system.kernel_compiler import compile_source, classify_error
from agent_system.kernel_loader import load_kernel, call_run_kernel, make_output_tensor
from agent_system.kernel_version_store import KernelVersionStore


@dataclass
class RealIterationResult:
    """真实迭代结果。"""
    iteration: int
    bottleneck: str
    candidates_generated: int
    candidates_compiled_ok: int
    candidates_correct: int
    candidates_faster: int
    best_speedup: Optional[float]
    verdict: str
    best_candidate_desc: str
    errors: list
    best_code: str = ""
    best_version_id: Optional[str] = None


def run_real_iteration(
    iteration: int,
    cfg: KernelConfig,
    baseline_time_ms: float,
    baseline_util: float,
    generate_fn: Callable,
    current_code: str,
    memory: DomainMemory,
    log: OptimizationLog,
    workdir: Path,
    predict_fn: Optional[Callable] = None,
    max_survivors: int = 3,
    warmup: int = 3,
    repeats: int = 10,
) -> RealIterationResult:
    """
    执行一轮真实优化迭代。

    generate_fn: 真实 LLM Coder 回调 (cfg, bottleneck, memory) -> List[Candidate]
    predict_fn: 真实 LLM cost model (可选)
    """
    errors = []
    t_start = time.time()

    # ── Analyst：roofline 分析 ──
    r = analyze(cfg)
    sk = suggest_split_k(cfg)
    bottleneck = (
        f"{r.bound_type}，带宽利用率 {baseline_util:.1%}，"
        f"建议 Split-K={sk}（batch*heads={cfg.batch_size*cfg.num_heads}）"
    )

    # ── Coder：LLM 生成候选 ──
    candidates = generate_fn(cfg, bottleneck, memory)

    # ── Profiler 阶段一：双层过滤 ──
    survivors, fstats = two_stage_filter(
        candidates, baseline_util, predict_fn=predict_fn, max_survivors=max_survivors,
    )

    # ── Profiler 阶段二：真实编译 + 运行 + 校验 ──
    best_speedup = None
    best_desc = ""
    best_code = current_code
    best_version_id = None
    compiled_ok = 0
    correct_count = 0
    faster_count = 0

    # 准备测试输入
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = make_test_inputs(cfg, device=device, seed=42)
    q, k_cache, v_cache, cache_seqlens, block_table = inputs
    num_blocks = k_cache.shape[0]
    output_ref = generate_reference(q, k_cache, v_cache, cache_seqlens, block_table, cfg)

    for cand in survivors:
        code = getattr(cand, "_code", None)
        if not code:
            errors.append(f"{cand.candidate_id}: 无代码")
            continue

        # 编译
        so_path = str(workdir / f"cand_{cand.candidate_id}.so")
        cres = compile_source(code, so_path, timeout=60)
        if not cres.success:
            category = classify_error(cres.stderr)
            memory.record_failure(
                category=category, symptom=cres.error_msg[:150],
                root_cause=f"候选 {cand.candidate_id} 编译失败",
                fix="待分析", config=f"b={cfg.batch_size},seq={cfg.seqlen_kv}",
            )
            errors.append(f"{cand.candidate_id}: 编译失败({category})")
            continue
        compiled_ok += 1

        # 加载 + 运行
        lres = load_kernel(so_path)
        if not lres.success:
            errors.append(f"{cand.candidate_id}: 加载失败 {lres.error_msg}")
            continue

        try:
            output = make_output_tensor(cfg, device=device)
            call_run_kernel(lres.run_kernel_fn, q, k_cache, v_cache, output,
                            cache_seqlens, block_table, cfg, num_blocks)
        except Exception as e:
            memory.record_failure(
                category="runtime_error", symptom=str(e)[:150],
                root_cause=f"候选 {cand.candidate_id} 运行崩溃",
                fix="待分析", config=f"b={cfg.batch_size},seq={cfg.seqlen_kv}",
            )
            errors.append(f"{cand.candidate_id}: 运行崩溃")
            continue

        # 正确性校验
        corr = correctness_check(output, output_ref)
        if not corr.passed:
            memory.record_failure(
                category="correctness_error",
                symptom=f"max_abs={corr.max_abs_diff:.4f}",
                root_cause=f"候选 {cand.candidate_id} 数值错误",
                fix="检查寻址/归约/bf16转换",
                config=f"b={cfg.batch_size},seq={cfg.seqlen_kv}",
            )
            errors.append(f"{cand.candidate_id}: 正确性失败 {corr.detail[:50]}")
            continue
        correct_count += 1

        # benchmark
        def run():
            call_run_kernel(lres.run_kernel_fn, q, k_cache, v_cache,
                            make_output_tensor(cfg, device), cache_seqlens, block_table,
                            cfg, num_blocks)
        bench = benchmark_config(run, cfg, warmup=warmup, repeats=repeats)

        speedup = baseline_time_ms / bench.time_ms
        if speedup > 1.0:
            faster_count += 1
            if best_speedup is None or speedup > best_speedup:
                best_speedup = speedup
                best_desc = cand.description
                best_code = code

    # ── Judge 裁决 ──
    if best_speedup is not None:
        verdict = "KEEP"
        memory.record_belief(
            observation=f"第{iteration}轮 {best_desc} 加速 {best_speedup:.2f}x",
            rule=f"{cfg.batch_size},{cfg.seqlen_kv}: 该优化有效",
            confidence=0.7,
        )
        # Persist the promoted code so the next iteration can build on it.
        try:
            store = KernelVersionStore(Path(__file__).resolve().parent / "kernel_versions")
            version = store.add_source(
                operator_id="flashattention_kvcache_decode",
                source_code=best_code,
                description=best_desc or f"iteration {iteration} promoted candidate",
                verdict="KEEP",
                metrics={
                    "iteration": iteration,
                    "speedup": best_speedup,
                    "compiled_ok": compiled_ok,
                    "correct_count": correct_count,
                    "faster_count": faster_count,
                    "config": f"b={cfg.batch_size},seq={cfg.seqlen_kv},d={cfg.headdim}",
                },
            )
            store.promote(version.version_id)
            best_version_id = version.version_id
        except Exception as e:
            errors.append(f"版本持久化失败: {e}")
    else:
        verdict = "ROLLBACK"
        if errors:
            memory.record_belief(
                observation=f"第{iteration}轮全部失败: {';'.join(errors[:2])}",
                rule="需调整优化方向",
                confidence=0.3,
            )

    # ── Logger ──
    log.record(OptimizationEntry(
        iteration=iteration,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        change_description=best_desc if best_desc else "无有效候选",
        target_config=f"batch={cfg.batch_size},seq={cfg.seqlen_kv},d={cfg.headdim}",
        baseline_time_ms=baseline_time_ms,
        candidate_time_ms=baseline_time_ms / best_speedup if best_speedup else None,
        speedup=best_speedup,
        bandwidth_util_before=baseline_util,
        bandwidth_util_after=None,
        correctness_passed=(verdict == "KEEP"),
        verdict=verdict,
        gap_to_roofline=None,
        notes=f"生成{len(candidates)} 编译OK{compiled_ok} 正确{correct_count} 更快{faster_count}",
    ))

    return RealIterationResult(
        iteration=iteration, bottleneck=bottleneck,
        candidates_generated=len(candidates), candidates_compiled_ok=compiled_ok,
        candidates_correct=correct_count, candidates_faster=faster_count,
        best_speedup=best_speedup, verdict=verdict,
        best_candidate_desc=best_desc, errors=errors,
        best_code=best_code if verdict == "KEEP" else current_code,
        best_version_id=best_version_id,
    )


def run_optimization_loop(
    cfg: KernelConfig,
    baseline_time_ms: float,
    baseline_util: float,
    initial_code: str,
    generate_fn: Callable,
    memory: DomainMemory,
    log: OptimizationLog,
    workdir: Path,
    max_iterations: int = 5,
    predict_fn: Optional[Callable] = None,
) -> List[RealIterationResult]:
    """
    运行完整优化循环（多轮迭代）。

    每轮用当前最优代码作为起点，让 LLM 继续优化。
    """
    results = []
    current_code = initial_code
    for i in range(max_iterations):
        print(f"\n{'='*50}\n  迭代 {i+1}/{max_iterations}\n{'='*50}")
        result = run_real_iteration(
            iteration=i+1, cfg=cfg,
            baseline_time_ms=baseline_time_ms,
            baseline_util=baseline_util,
            generate_fn=generate_fn, current_code=current_code,
            memory=memory, log=log, workdir=workdir,
            predict_fn=predict_fn,
        )
        results.append(result)
        print(f"  生成:{result.candidates_generated} 编译OK:{result.candidates_compiled_ok} "
              f"正确:{result.candidates_correct} 更快:{result.candidates_faster}")
        print(f"  裁决: {result.verdict} 加速: {result.best_speedup}")
        if result.best_version_id:
            print(f"  版本: {result.best_version_id}")
        if result.errors:
            print(f"  错误: {'; '.join(result.errors[:2])}")

        # 如果 KEEP，更新当前代码为最优
        if result.verdict == "KEEP":
            current_code = result.best_code
        # 连续多轮无改进可提前停止
        if i >= 1 and all(r.verdict == "ROLLBACK" for r in results[-2:]):
            print("  连续2轮无改进，提前停止")
            break
    return results


# ════════════════════════════════════════════════════════════════════
# 严格的滚动 A/B 实验编排器（单轮单点改动 + 噪声容限 KEEP）
# ════════════════════════════════════════════════════════════════════
# 设计要点（详见 docs/MCTLASS_GEMM_EXPERIMENT.md 与计划）：
# 1. 每轮 A = 当前已 KEEP 的最优版本（滚动基线）；KEEP 后版本链推进
# 2. 每轮只接受一个 ChangeProposal（单点改动），由 validate_single_change 硬约束
# 3. KEEP 判定带噪声容限：B 的中位数必须 < A 的中位数 * (1 - noise_margin)
# 4. 编译失败/运行崩溃 → ERROR；正确性失败 → REJECT；提速不足 → NOCHANGE
# 5. 每一步按 category 写入分类归档日志

from agent_system.strategy_schema import (
    ChangeProposal,
    PatchApplyError,
    apply_change_proposal,
    validate_single_change,
)
from agent_system.benchmark_engine import benchmark_fn


@dataclass
class ABResult:
    """单轮 A/B 实验结果。"""
    iteration: int
    proposal: Optional[ChangeProposal]
    baseline_version: str            # A: 当前最优版本 id
    baseline_time_ms: float          # A 的基准时间（中位数）
    candidate_time_ms: Optional[float]   # B 的时间（中位数）
    speedup: Optional[float]         # A/B（A.median / B.median）
    correctness_passed: bool
    verdict: str                     # KEEP | REJECT | ERROR | NOCHANGE | SKIP
    reject_reason: str = ""          # KEEP 以外的原因
    noise_margin: float = 0.03
    promoted_version: Optional[str] = None  # KEEP 时新版本 id
    errors: list = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def _safe_artifact_name(value: str, default: str = "candidate") -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in value)
    return (safe or default)[:80]


def _call_generate_proposal(
    fn: Callable,
    cfg: KernelConfig,
    bottleneck: str,
    memory: DomainMemory,
    current_code: str,
) -> Optional[ChangeProposal]:
    try:
        if len(inspect.signature(fn).parameters) >= 4:
            return fn(cfg, bottleneck, memory, current_code)
    except (TypeError, ValueError):
        pass
    return fn(cfg, bottleneck, memory)


def _bench_median_ms(fn: Callable, cfg: KernelConfig, warmup: int, repeats: int) -> float:
    """benchmark 取中位数（比均值更抗 outlier/抖动，配合噪声容限判定）。"""
    times = benchmark_fn(fn, warmup=warmup, repeats=repeats,
                         use_cuda_event=torch.cuda.is_available())
    t = torch.tensor(times)
    return float(t.float().median().item())


def run_ab_iteration(
    iteration: int,
    cfg: KernelConfig,
    current_code: str,
    current_version_id: str,
    generate_proposal_fn: Callable[[KernelConfig, str, DomainMemory], Optional[ChangeProposal]],
    memory: DomainMemory,
    log: OptimizationLog,
    version_store: KernelVersionStore,
    workdir: Path,
    noise_margin: float = 0.03,
    warmup: int = 5,
    repeats: int = 30,
) -> ABResult:
    """执行一轮严格的滚动 A/B 实验。

    generate_proposal_fn 返回 *单个* ChangeProposal（其 .after 是改动后的完整 kernel 源码）。
    返回 ABResult，并已写入分类日志、更新记忆库、（KEEP 时）推进版本链。
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "baseline_a.cu").write_text(current_code, encoding="utf-8")
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Analyst：roofline 分析 ──
    r = analyze(cfg)
    bottleneck = (
        f"{r.bound_type}，建议 Split-K={suggest_split_k(cfg)} "
        f"(batch*heads={cfg.batch_size*cfg.num_heads})"
    )
    (workdir / "analysis.json").write_text(json.dumps({
        "iteration": iteration,
        "baseline_version": current_version_id,
        "config": asdict(cfg),
        "bottleneck": bottleneck,
        "roofline": asdict(r),
        "noise_margin": noise_margin,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    def _record(verdict, category, candidate_ms, speedup, correctness, reason,
                proposal=None, promoted=None):
        log.record(OptimizationEntry(
            iteration=iteration, timestamp=ts,
            change_description=(proposal.one_line_summary if proposal else "无提案"),
            target_config=f"batch={cfg.batch_size},seq={cfg.seqlen_kv},d={cfg.headdim}",
            baseline_time_ms=baseline_ms, candidate_time_ms=candidate_ms,
            speedup=speedup, bandwidth_util_before=0.0,
            bandwidth_util_after=None, correctness_passed=correctness,
            verdict=verdict, gap_to_roofline=None,
            category=category,
            proposal_target=(proposal.target if proposal else ""),
            diff_summary=(proposal.to_diff_summary() if proposal else ""),
            baseline_version=current_version_id, promoted_version=promoted or "",
            noise_margin=noise_margin, reject_reason=reason, used_median=True,
        ))

    # ── Step 1: benchmark A（当前最优），取中位数 ──
    inputs = make_test_inputs(cfg, device=device, seed=42)
    q, k_cache, v_cache, cache_seqlens, block_table = inputs
    num_blocks = k_cache.shape[0]

    # 编译当前代码为 A 的 .so
    a_so = str(workdir / "baseline_a.so")
    a_cres = compile_source(current_code, a_so, timeout=120)
    if not a_cres.success:
        # 极端情况：当前最优代码都编不过（不应发生），直接 ERROR
        log.record(OptimizationEntry(
            iteration=iteration, timestamp=ts,
            change_description="baseline compile failed",
            target_config=f"batch={cfg.batch_size},seq={cfg.seqlen_kv},d={cfg.headdim}",
            baseline_time_ms=0.0, candidate_time_ms=None, speedup=None,
            bandwidth_util_before=0.0, bandwidth_util_after=None,
            correctness_passed=False, verdict="ERROR", gap_to_roofline=None,
            category="error_compile", proposal_target="baseline",
            diff_summary="", baseline_version=current_version_id,
            noise_margin=noise_margin,
            reject_reason=f"baseline 编译失败: {a_cres.error_msg[:120]}",
            used_median=False,
        ))
        return ABResult(iteration=iteration, proposal=None, baseline_version=current_version_id,
                        baseline_time_ms=0.0, candidate_time_ms=None, speedup=None,
                        correctness_passed=False, verdict="ERROR",
                        reject_reason=f"baseline 编译失败: {a_cres.error_msg[:80]}",
                        noise_margin=noise_margin)
    a_loaded = load_kernel(a_so)
    a_fn = a_loaded.run_kernel_fn

    def _run_a():
        call_run_kernel(a_fn, q, k_cache, v_cache, make_output_tensor(cfg, device),
                        cache_seqlens, block_table, cfg, num_blocks)
    baseline_ms = _bench_median_ms(_run_a, cfg, warmup, repeats)
    output_ref = generate_reference(q, k_cache, v_cache, cache_seqlens, block_table, cfg)

    # ── Step 2: Coder 提出单点改动 ──
    proposal = _call_generate_proposal(generate_proposal_fn, cfg, bottleneck, memory, current_code)
    if proposal is None:
        _record("NOCHANGE", "nochange", None, None, True, "Coder 未提出提案")
        return ABResult(iteration=iteration, proposal=None, baseline_version=current_version_id,
                        baseline_time_ms=baseline_ms, candidate_time_ms=None, speedup=None,
                        correctness_passed=True, verdict="NOCHANGE",
                        reject_reason="Coder 未提出提案", noise_margin=noise_margin)

    proposal_name = _safe_artifact_name(proposal.proposal_id)
    (workdir / "proposal.json").write_text(
        json.dumps(proposal.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 校验单点改动约束
    ok, reason = validate_single_change(proposal)
    if not ok:
        _record("NOCHANGE", "nochange", None, None, True, reason, proposal=proposal)
        return ABResult(iteration=iteration, proposal=proposal, baseline_version=current_version_id,
                        baseline_time_ms=baseline_ms, candidate_time_ms=None, speedup=None,
                        correctness_passed=True, verdict="NOCHANGE",
                        reject_reason=reason, noise_margin=noise_margin)

    try:
        candidate_code = apply_change_proposal(current_code, proposal)
    except PatchApplyError as e:
        reason = str(e)
        memory.record_failure(category="patch_error", symptom=reason,
                              root_cause=f"{proposal.proposal_id} patch apply failed",
                              fix="make before snippet exact and unique",
                              config=f"b={cfg.batch_size},seq={cfg.seqlen_kv}")
        _record("ERROR", "error_patch", None, None, False, reason, proposal=proposal)
        (workdir / "decision.json").write_text(json.dumps({
            "iteration": iteration,
            "verdict": "ERROR",
            "category": "error_patch",
            "reason": reason,
            "proposal_id": proposal.proposal_id,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return ABResult(iteration=iteration, proposal=proposal,
                        baseline_version=current_version_id,
                        baseline_time_ms=baseline_ms, candidate_time_ms=None,
                        speedup=None, correctness_passed=False, verdict="ERROR",
                        reject_reason=reason, noise_margin=noise_margin,
                        errors=[reason])

    (workdir / f"cand_{proposal_name}.cu").write_text(candidate_code, encoding="utf-8")

    # ── Step 3: 编译 B ──
    b_so = str(workdir / f"cand_{proposal_name}.so")
    b_cres = compile_source(candidate_code, b_so, timeout=120)
    if not b_cres.success:
        cat = classify_error(b_cres.stderr)
        err = b_cres.error_msg[:120]
        memory.record_failure(category=cat, symptom=err,
                              root_cause=f"{proposal.proposal_id} 编译失败",
                              fix="待分析", config=f"b={cfg.batch_size},seq={cfg.seqlen_kv}")
        _record("ERROR", "error_compile", None, None, False, f"编译失败({cat}): {err}",
                proposal=proposal)
        return ABResult(iteration=iteration, proposal=proposal, baseline_version=current_version_id,
                        baseline_time_ms=baseline_ms, candidate_time_ms=None, speedup=None,
                        correctness_passed=False, verdict="ERROR",
                        reject_reason=f"编译失败({cat})", noise_margin=noise_margin,
                        errors=[f"compile: {err}"])

    # ── Step 4: 运行 B + 正确性校验 ──
    b_loaded = load_kernel(b_so)
    if not b_loaded.success:
        _record("ERROR", "error_runtime", None, None, False,
                f"加载失败: {b_loaded.error_msg[:80]}", proposal=proposal)
        return ABResult(iteration=iteration, proposal=proposal, baseline_version=current_version_id,
                        baseline_time_ms=baseline_ms, candidate_time_ms=None, speedup=None,
                        correctness_passed=False, verdict="ERROR",
                        reject_reason=f"加载失败", noise_margin=noise_margin,
                        errors=[f"load: {b_loaded.error_msg[:80]}"])
    b_fn = b_loaded.run_kernel_fn
    try:
        output_b = make_output_tensor(cfg, device)
        call_run_kernel(b_fn, q, k_cache, v_cache, output_b,
                        cache_seqlens, block_table, cfg, num_blocks)
    except Exception as e:
        msg = str(e)[:120]
        memory.record_failure(category="runtime_error", symptom=msg,
                              root_cause=f"{proposal.proposal_id} 运行崩溃",
                              fix="检查寻址/越界", config=f"b={cfg.batch_size},seq={cfg.seqlen_kv}")
        _record("ERROR", "error_runtime", None, None, False, f"运行崩溃: {msg}", proposal=proposal)
        return ABResult(iteration=iteration, proposal=proposal, baseline_version=current_version_id,
                        baseline_time_ms=baseline_ms, candidate_time_ms=None, speedup=None,
                        correctness_passed=False, verdict="ERROR",
                        reject_reason=f"运行崩溃", noise_margin=noise_margin, errors=[msg])

    corr = correctness_check(output_b, output_ref)
    if not corr.passed:
        detail = f"max_abs={corr.max_abs_diff:.4f}"
        memory.record_failure(category="correctness_error", symptom=detail,
                              root_cause=f"{proposal.proposal_id} 数值错误",
                              fix="检查寻址/归约/bf16转换", config=f"b={cfg.batch_size},seq={cfg.seqlen_kv}")
        _record("REJECT", "reject", None, None, False, f"正确性失败: {detail}", proposal=proposal)
        return ABResult(iteration=iteration, proposal=proposal, baseline_version=current_version_id,
                        baseline_time_ms=baseline_ms, candidate_time_ms=None, speedup=None,
                        correctness_passed=False, verdict="REJECT",
                        reject_reason=f"正确性失败 {detail}", noise_margin=noise_margin)

    # ── Step 5: benchmark B，取中位数 ──
    def _run_b():
        call_run_kernel(b_fn, q, k_cache, v_cache, make_output_tensor(cfg, device),
                        cache_seqlens, block_table, cfg, num_blocks)
    candidate_ms = _bench_median_ms(_run_b, cfg, warmup, repeats)
    speedup = baseline_ms / candidate_ms if candidate_ms > 0 else 0.0

    # ── Step 6: KEEP 判定（带噪声容限）──
    # B 必须比 A 快 *超过* noise_margin（非噪声），才视为真实优化
    keep = candidate_ms < baseline_ms * (1.0 - noise_margin)

    if keep:
        verdict = "KEEP"
        # 版本链推进：以当前最优为 parent，持久化新版本
        version = version_store.add_source(
            operator_id="flashattention_kvcache_decode",
            source_code=candidate_code,
            description=proposal.one_line_summary or proposal.target,
            verdict="KEEP",
            metrics={
                "iteration": iteration, "speedup": speedup,
                "baseline_ms": baseline_ms, "candidate_ms": candidate_ms,
                "noise_margin": noise_margin, "target": proposal.target,
                "config": f"b={cfg.batch_size},seq={cfg.seqlen_kv},d={cfg.headdim}",
            },
            parent=current_version_id,
        )
        version_store.promote(version.version_id)
        promoted = version.version_id
        memory.record_belief(
            observation=f"第{iteration}轮 {proposal.target} 加速 {speedup:.3f}x "
                        f"({baseline_ms:.4f}→{candidate_ms:.4f}ms, 超 {noise_margin:.0%} 容限)",
            rule=f"{proposal.target}: {proposal.one_line_summary} 有效",
            confidence=0.75,
        )
        _record("KEEP", "keep", candidate_ms, speedup, True, "", proposal=proposal, promoted=promoted)
        return ABResult(iteration=iteration, proposal=proposal, baseline_version=current_version_id,
                        baseline_time_ms=baseline_ms, candidate_time_ms=candidate_ms,
                        speedup=speedup, correctness_passed=True, verdict="KEEP",
                        noise_margin=noise_margin, promoted_version=promoted)

    # 未超过噪声容限 → NOCHANGE（不推进版本）
    reason = (f"提速不足: {candidate_ms:.4f}ms vs {baseline_ms:.4f}ms "
              f"(speedup {speedup:.3f}x < 1+{noise_margin:.0%} 容限)")
    memory.record_belief(
        observation=f"第{iteration}轮 {proposal.target} 提速不显著 ({speedup:.3f}x)",
        rule=f"{proposal.target}: 改动在噪声内，需换方向或增大改动",
        confidence=0.4,
    )
    _record("NOCHANGE", "nochange", candidate_ms, speedup, True, reason, proposal=proposal)
    return ABResult(iteration=iteration, proposal=proposal, baseline_version=current_version_id,
                    baseline_time_ms=baseline_ms, candidate_time_ms=candidate_ms,
                    speedup=speedup, correctness_passed=True, verdict="NOCHANGE",
                    reject_reason=reason, noise_margin=noise_margin)


def run_ab_loop(
    cfg: KernelConfig,
    initial_code: str,
    initial_version_id: str,
    generate_proposal_fn: Callable,
    memory: DomainMemory,
    log: OptimizationLog,
    version_store: KernelVersionStore,
    workdir: Path,
    max_iterations: int = 5,
    noise_margin: float = 0.03,
    warmup: int = 5,
    repeats: int = 30,
) -> List[ABResult]:
    """运行多轮滚动 A/B 闭环。

    每轮的 A 自动滚到上一轮 KEEP 的版本（版本链 monotonically 递进）。
    连续多轮无 KEEP 时提前停止。
    """
    results = []
    current_code = initial_code
    current_version_id = initial_version_id
    for i in range(max_iterations):
        print(f"\n{'='*50}\n  A/B 迭代 {i+1}/{max_iterations}  (A={current_version_id})\n{'='*50}")
        round_dir = Path(workdir) / f"round{i+1}"
        result = run_ab_iteration(
            iteration=i + 1, cfg=cfg,
            current_code=current_code, current_version_id=current_version_id,
            generate_proposal_fn=generate_proposal_fn,
            memory=memory, log=log, version_store=version_store,
            workdir=round_dir,
            noise_margin=noise_margin, warmup=warmup, repeats=repeats,
        )
        results.append(result)
        (round_dir / "decision.json").write_text(json.dumps({
            "iteration": result.iteration,
            "verdict": result.verdict,
            "baseline_version": result.baseline_version,
            "promoted_version": result.promoted_version,
            "baseline_time_ms": result.baseline_time_ms,
            "candidate_time_ms": result.candidate_time_ms,
            "speedup": result.speedup,
            "correctness_passed": result.correctness_passed,
            "reject_reason": result.reject_reason,
            "noise_margin": result.noise_margin,
            "proposal_id": result.proposal.proposal_id if result.proposal else "",
            "proposal_target": result.proposal.target if result.proposal else "",
            "errors": result.errors,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  verdict={result.verdict}  "
              f"A={result.baseline_time_ms:.4f}ms  B={result.candidate_time_ms}  "
              f"speedup={result.speedup}")
        if result.reject_reason:
            print(f"  reason: {result.reject_reason[:70]}")
        if result.verdict == "KEEP" and result.promoted_version:
            # KEEP：版本链推进，下一轮 A = 新版本
            current_code = apply_change_proposal(current_code, result.proposal)
            current_version_id = result.promoted_version
            print(f"  ✓ 版本推进 → {current_version_id}")
        # 连续 2 轮无 KEEP，提前停止
        if i >= 1 and all(r.verdict != "KEEP" for r in results[-2:]):
            print("  连续 2 轮无 KEEP，提前停止")
            break
    return results
