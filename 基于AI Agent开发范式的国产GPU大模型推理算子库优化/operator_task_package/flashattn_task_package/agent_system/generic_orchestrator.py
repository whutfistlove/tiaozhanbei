"""
Generalized multi-agent optimization loop.

This module is intentionally operator-agnostic.  It works with OperatorSpec,
BackendAdapter, Evaluator and structured strategies, so FlashAttention is one
registered task rather than a hard-coded orchestrator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from agent_system.backend_adapter import CompileDiagnostic, make_backend_adapter
from agent_system.domain_memory import DomainMemory
from agent_system.evaluator import MatrixEvaluation, make_evaluator
from agent_system.kernel_version_store import KernelVersionStore
from agent_system.llm_cost_model import Candidate, two_stage_filter
from agent_system.operator_registry import get_operator
from agent_system.optimization_log import OptimizationEntry, OptimizationLog
from agent_system.roofline_engine import KernelConfig, analyze, suggest_split_k
from agent_system.specs import OperatorSpec
from agent_system.strategy_schema import OptimizationStrategy


StrategyGenerator = Callable[[OperatorSpec, str, DomainMemory], list[OptimizationStrategy]]


@dataclass
class AgentRoleReport:
    role: str
    summary: str
    artifacts: list[str] = field(default_factory=list)


@dataclass
class GenericIterationResult:
    iteration: int
    operator_id: str
    verdict: str
    analyst_report: AgentRoleReport
    generated: int
    survivors: int
    compile_ok: int
    best_strategy: Optional[OptimizationStrategy] = None
    best_evaluation: Optional[MatrixEvaluation] = None
    errors: list[str] = field(default_factory=list)


def analyze_operator(spec: OperatorSpec) -> AgentRoleReport:
    """Analyst role: generic task summary plus FlashAttention roofline hints."""
    lines = [spec.summary(), f"expected_bound={spec.expected_bound}"]
    if spec.operator_id == "flashattention_kvcache_decode":
        md = spec.metadata
        sample = KernelConfig(
            batch_size=1,
            seqlen_kv=4096,
            headdim=int(md.get("headdim", 128)),
            num_heads=int(md.get("num_heads", 8)),
            num_heads_k=int(md.get("num_heads_k", 8)),
        )
        r = analyze(sample)
        lines.append(
            f"sample roofline: {r.bound_type}, intensity={r.arithmetic_intensity:.3f}, "
            f"suggest_split_k={suggest_split_k(sample)}"
        )
    lines.append("skills=" + ", ".join(spec.skills))
    return AgentRoleReport(role="Analyst", summary="\n".join(lines))


def _strategy_to_candidate(strategy: OptimizationStrategy) -> Candidate:
    return strategy.to_candidate()


def run_generic_iteration(
    operator_id: str,
    iteration: int,
    generate_strategies_fn: StrategyGenerator,
    memory: DomainMemory,
    log: OptimizationLog,
    version_store: KernelVersionStore,
    workdir: Path,
    baseline_util: float = 0.45,
    max_survivors: int = 3,
    predict_fn=None,
) -> GenericIterationResult:
    spec = get_operator(operator_id)
    analyst = analyze_operator(spec)
    strategies = generate_strategies_fn(spec, analyst.summary, memory)
    candidates = [_strategy_to_candidate(s) for s in strategies]
    survivors, fstats = two_stage_filter(
        candidates,
        baseline_util=baseline_util,
        predict_fn=predict_fn,
        max_survivors=max_survivors,
    )
    survivor_ids = {c.candidate_id for c in survivors}
    survivor_strategies = [s for s in strategies if s.strategy_id in survivor_ids]

    backend = make_backend_adapter(spec.backend)
    evaluator = make_evaluator(spec)
    errors: list[str] = []
    compile_ok = 0
    best_strategy = None
    best_eval = None

    for strategy in survivor_strategies:
        cdiag = backend.compile_strategy(strategy, workdir)
        if not cdiag.success:
            memory.record_failure(
                category=cdiag.category or "compile_error",
                symptom=cdiag.error[:200],
                root_cause=f"{strategy.strategy_id} compile failed on {spec.backend.kind}",
                fix="use structured template or adjust backend-specific API",
                config=f"operator={operator_id}",
            )
            errors.append(f"{strategy.strategy_id}: compile failed ({cdiag.category})")
            continue
        compile_ok += 1
        loaded = backend.load(cdiag.artifact_path or "")
        if not loaded.success:
            errors.append(f"{strategy.strategy_id}: load failed {loaded.error}")
            continue
        evaluation = evaluator.evaluate_matrix(loaded.handle)
        if evaluation.all_correct:
            best_strategy = strategy
            best_eval = evaluation
            if strategy.source_code:
                version = version_store.add_source(
                    operator_id=operator_id,
                    source_code=strategy.source_code,
                    description=strategy.description or strategy.strategy_name,
                    verdict="KEEP",
                    metrics={"summary": evaluation.summary()},
                )
                version_store.promote(version.version_id)
            break
        errors.extend(evaluation.errors)

    verdict = "KEEP" if best_strategy and best_eval and best_eval.all_correct else "ROLLBACK"
    if verdict == "KEEP":
        memory.record_belief(
            observation=f"{operator_id} {best_strategy.strategy_name} passed matrix evaluation",
            rule=f"strategy params {best_strategy.params} are valid for {operator_id}",
            confidence=0.65,
        )
    else:
        memory.record_belief(
            observation=f"{operator_id} iteration {iteration} had no valid promoted strategy",
            rule="need broaden strategy or improve backend template",
            confidence=0.3,
        )

    log.record(OptimizationEntry(
        iteration=iteration,
        timestamp=__import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        change_description=best_strategy.description if best_strategy else "无有效泛化候选",
        target_config=f"operator={operator_id}",
        baseline_time_ms=0.0,
        candidate_time_ms=None,
        speedup=None,
        bandwidth_util_before=baseline_util,
        bandwidth_util_after=None,
        correctness_passed=(verdict == "KEEP"),
        verdict=verdict,
        notes=f"generated={len(strategies)} survivors={fstats.survivors} compile_ok={compile_ok}",
    ))

    return GenericIterationResult(
        iteration=iteration,
        operator_id=operator_id,
        verdict=verdict,
        analyst_report=analyst,
        generated=len(strategies),
        survivors=len(survivor_strategies),
        compile_ok=compile_ok,
        best_strategy=best_strategy,
        best_evaluation=best_eval,
        errors=errors,
    )
