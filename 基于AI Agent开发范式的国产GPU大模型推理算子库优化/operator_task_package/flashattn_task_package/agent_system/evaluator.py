"""
General evaluator layer for correctness, benchmark and score reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agent_system.specs import OperatorSpec, TestCaseSpec


@dataclass
class CaseEvaluation:
    case_id: str
    label: str
    correctness_passed: bool
    detail: str = ""
    bench: Optional[object] = None
    baseline_time_ms: Optional[float] = None
    speedup_vs_baseline: Optional[float] = None
    score_estimate: Optional[float] = None


@dataclass
class MatrixEvaluation:
    operator_id: str
    cases: list[CaseEvaluation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def all_correct(self) -> bool:
        return bool(self.cases) and all(c.correctness_passed for c in self.cases) and not self.errors

    @property
    def mean_speedup(self) -> Optional[float]:
        values = [c.speedup_vs_baseline for c in self.cases if c.speedup_vs_baseline]
        if not values:
            return None
        return sum(values) / len(values)

    def summary(self) -> str:
        passed = sum(1 for c in self.cases if c.correctness_passed)
        total = len(self.cases)
        speed = self.mean_speedup
        speed_s = f", mean_speedup={speed:.3f}x" if speed else ""
        return f"{self.operator_id}: correctness {passed}/{total}{speed_s}, errors={len(self.errors)}"


class Evaluator:
    """Base evaluator. Concrete operators can override evaluate_case."""

    def __init__(self, spec: OperatorSpec):
        self.spec = spec

    def evaluate_case(self, artifact_handle, case: TestCaseSpec) -> CaseEvaluation:
        raise NotImplementedError

    def evaluate_matrix(self, artifact_handle, case_ids: Optional[list[str]] = None) -> MatrixEvaluation:
        selected = self.spec.test_cases
        if case_ids:
            wanted = set(case_ids)
            selected = tuple(c for c in selected if c.case_id in wanted)
        result = MatrixEvaluation(operator_id=self.spec.operator_id)
        for case in selected:
            try:
                result.cases.append(self.evaluate_case(artifact_handle, case))
            except Exception as exc:
                result.errors.append(f"{case.case_id}: {exc}")
        return result


class FlashAttentionDecodeEvaluator(Evaluator):
    """Evaluator plugin for Track 2 FlashAttention paged KV decode."""

    def _cfg_from_case(self, case: TestCaseSpec) -> KernelConfig:
        from agent_system.roofline_engine import KernelConfig

        md = self.spec.metadata
        return KernelConfig(
            batch_size=int(case.params["batch_size"]),
            seqlen_kv=int(case.params["seqlen_kv"]),
            seqlen_q=int(md.get("seqlen_q", 1)),
            num_heads=int(md.get("num_heads", 8)),
            num_heads_k=int(md.get("num_heads_k", 8)),
            headdim=int(md.get("headdim", 128)),
            page_block_size=int(md.get("page_block_size", 16)),
        )

    def evaluate_case(self, artifact_handle, case: TestCaseSpec) -> CaseEvaluation:
        import torch
        from agent_system.benchmark_engine import benchmark_config
        from agent_system.correctness import check, generate_reference, make_test_inputs
        from agent_system.kernel_loader import call_run_kernel, make_output_tensor

        if artifact_handle is None:
            raise RuntimeError("FlashAttention evaluator requires a loaded run_kernel handle")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable")

        cfg = self._cfg_from_case(case)
        q, k_cache, v_cache, cache_seqlens, block_table = make_test_inputs(
            cfg, device="cuda", seed=42,
        )
        num_blocks = k_cache.shape[0]
        out = call_run_kernel(
            artifact_handle,
            q, k_cache, v_cache, make_output_tensor(cfg, "cuda"),
            cache_seqlens, block_table, cfg, num_blocks,
        )
        ref = generate_reference(q, k_cache, v_cache, cache_seqlens, block_table, cfg)
        corr = check(
            out, ref,
            rtol=self.spec.evaluation.accuracy.rtol,
            atol=self.spec.evaluation.accuracy.atol,
        )
        bench = None
        if corr.passed:
            def run():
                call_run_kernel(
                    artifact_handle,
                    q, k_cache, v_cache, make_output_tensor(cfg, "cuda"),
                    cache_seqlens, block_table, cfg, num_blocks,
                )
            bench = benchmark_config(
                run, cfg,
                warmup=self.spec.evaluation.warmup,
                repeats=self.spec.evaluation.repeats,
            )
        return CaseEvaluation(
            case_id=case.case_id,
            label=case.label(),
            correctness_passed=corr.passed,
            detail=corr.detail,
            bench=bench,
        )


class SpecOnlyEvaluator(Evaluator):
    """Evaluator for operators registered only to test framework generality."""

    def evaluate_case(self, artifact_handle, case: TestCaseSpec) -> CaseEvaluation:
        return CaseEvaluation(
            case_id=case.case_id,
            label=case.label(),
            correctness_passed=True,
            detail="spec-only evaluator: no runtime kernel executed",
        )


def make_evaluator(spec: OperatorSpec) -> Evaluator:
    if spec.operator_id == "flashattention_kvcache_decode":
        return FlashAttentionDecodeEvaluator(spec)
    return SpecOnlyEvaluator(spec)
