"""
Generic operator optimization specifications.

These dataclasses are the contract between the multi-agent loop and a concrete
operator task.  The orchestrator should depend on these abstractions instead of
hard-coding FlashAttention shapes or MACA C++ details.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional


BackendKind = Literal["maca_cpp", "triton", "tilelang", "mock"]
BoundKind = Literal["memory-bound", "compute-bound", "balanced", "unknown"]


@dataclass(frozen=True)
class TensorSpec:
    """A symbolic tensor contract for an operator input or output."""

    name: str
    shape: str
    dtype: str
    layout: str = "contiguous"
    role: str = "input"
    description: str = ""


@dataclass(frozen=True)
class TestCaseSpec:
    """One benchmark/correctness case in an operator test matrix."""

    case_id: str
    params: dict[str, Any]
    tags: tuple[str, ...] = ()
    weight: float = 1.0

    def label(self) -> str:
        parts = [f"{k}={v}" for k, v in self.params.items()]
        return ",".join(parts)


@dataclass(frozen=True)
class AccuracySpec:
    """Numerical correctness rule."""

    method: str = "torch.allclose"
    rtol: float = 1e-2
    atol: float = 1e-2
    reference: str = "pytorch"


@dataclass(frozen=True)
class BackendSpec:
    """How a candidate implementation is built and run."""

    kind: BackendKind
    language: str
    compile_command: str = ""
    runtime: str = ""
    include_paths: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class OptimizationParam:
    """A tunable parameter exposed to the Coder/Profiler agents."""

    name: str
    values: tuple[Any, ...]
    default: Any
    description: str
    risk: str = "medium"


@dataclass(frozen=True)
class OptimizationSpace:
    """Search space for strategy generation and autotuning."""

    params: tuple[OptimizationParam, ...] = ()
    strategy_names: tuple[str, ...] = ()
    mutually_exclusive: tuple[tuple[str, str], ...] = ()

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}


@dataclass(frozen=True)
class EvaluationSpec:
    """Evaluation policy shared by Profiler and Judge."""

    accuracy: AccuracySpec
    primary_metric: str = "time_ms"
    higher_is_better: bool = False
    warmup: int = 5
    repeats: int = 30
    require_all_correct: bool = True
    score_formula: str = "roofline_relative"


@dataclass(frozen=True)
class OperatorSpec:
    """Full task contract for an optimizable operator."""

    operator_id: str
    display_name: str
    category: str
    interface: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    test_cases: tuple[TestCaseSpec, ...]
    backend: BackendSpec
    evaluation: EvaluationSpec
    optimization_space: OptimizationSpace
    skills: tuple[str, ...] = ()
    expected_bound: BoundKind = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def case_by_id(self, case_id: str) -> TestCaseSpec:
        for case in self.test_cases:
            if case.case_id == case_id:
                return case
        raise KeyError(f"unknown case_id for {self.operator_id}: {case_id}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"{self.operator_id} ({self.category}) with {len(self.test_cases)} cases, "
            f"backend={self.backend.kind}, metric={self.evaluation.primary_metric}"
        )


def make_case_grid(prefix: str, grid: dict[str, tuple[Any, ...]],
                   tags: Optional[tuple[str, ...]] = None) -> tuple[TestCaseSpec, ...]:
    """Build a deterministic cartesian product of test cases."""
    import itertools

    keys = list(grid.keys())
    cases: list[TestCaseSpec] = []
    for idx, values in enumerate(itertools.product(*(grid[k] for k in keys)), 1):
        params = dict(zip(keys, values))
        cases.append(TestCaseSpec(
            case_id=f"{prefix}_{idx:03d}",
            params=params,
            tags=tags or (),
        ))
    return tuple(cases)
