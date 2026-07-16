"""
Structured optimization strategies and patch proposals produced by Coder agents.

There are two layers:

1. OptimizationStrategy: operator-agnostic strategy intent for generic agents.
2. ChangeProposal: concrete candidate source change for the A/B closed loop.

ChangeProposal is tiered so the system can explore large structural candidates
early, stabilize medium changes, and only then do tiny parameter tuning.
"""
from __future__ import annotations

import difflib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

from agent_system.llm_cost_model import Candidate


RiskLevel = Literal["low", "medium", "high"]
ChangeType = Literal[
    "param_tune",
    "loop_transform",
    "memory_layout",
    "api_swap",
    "template_swap",
    "structural_rewrite",
    "other",
]
ProposalScale = Literal["macro", "meso", "micro"]
OptimizationPhase = Literal["explore", "stabilize", "tune"]

_ALLOWED_CHANGE_TYPES = {
    "param_tune",
    "loop_transform",
    "memory_layout",
    "api_swap",
    "template_swap",
    "structural_rewrite",
    "other",
}
_ALLOWED_SCALES = {"macro", "meso", "micro"}
_ALLOWED_PHASES = {"explore", "stabilize", "tune"}
_MACRO_CHANGE_TYPES = {"template_swap", "structural_rewrite", "memory_layout", "loop_transform"}

DEFAULT_MAX_CHANGED_LINES = 6
SCALE_DEFAULT_MAX_CHANGED_LINES = {
    "micro": 6,
    "meso": 40,
    "macro": 400,
}


@dataclass
class OptimizationStrategy:
    strategy_id: str
    target_operator: str
    backend: str
    strategy_name: str
    params: dict[str, Any] = field(default_factory=dict)
    expected_effect: str = ""
    risk: RiskLevel = "medium"
    required_skills: list[str] = field(default_factory=list)
    description: str = ""
    code_path: Optional[str] = None
    source_code: Optional[str] = None

    def to_candidate(self) -> Candidate:
        cand = Candidate(
            candidate_id=self.strategy_id,
            description=self.description or self.strategy_name,
            params={"strategy_name": self.strategy_name, **self.params},
            confidence=0.6 if self.risk != "high" else 0.45,
        )
        if self.source_code:
            object.__setattr__(cand, "_code", self.source_code)
        if self.code_path:
            object.__setattr__(cand, "_code_path", self.code_path)
        return cand

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def strategy_from_candidate(
    candidate: Candidate,
    operator_id: str,
    backend: str = "maca_cpp",
) -> OptimizationStrategy:
    params = dict(candidate.params or {})
    strategy_name = str(params.pop("strategy_name", "candidate_patch"))
    return OptimizationStrategy(
        strategy_id=candidate.candidate_id,
        target_operator=operator_id,
        backend=backend,
        strategy_name=strategy_name,
        params=params,
        description=candidate.description,
        source_code=getattr(candidate, "_code", None),
        code_path=getattr(candidate, "_code_path", None),
    )


def parse_strategy_json(text: str) -> list[OptimizationStrategy]:
    data = json.loads(text)
    if isinstance(data, dict) and "strategies" in data:
        data = data["strategies"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("strategy JSON must be an object, list, or {'strategies': [...]}")
    return [OptimizationStrategy(**item) for item in data]


def validate_strategy(
    strategy: OptimizationStrategy,
    allowed_names: tuple[str, ...],
) -> tuple[bool, str]:
    if strategy.strategy_name not in allowed_names:
        return False, f"unknown strategy_name={strategy.strategy_name}"
    if not strategy.target_operator:
        return False, "missing target_operator"
    return True, "ok"


@dataclass
class ChangeProposal:
    """A single candidate change for the closed-loop A/B runner.

    scale controls how large the candidate may be:
    - macro: early exploration. Large isolated candidate, often patched_source.
    - meso: local rewrite or medium refactor.
    - micro: late tuning. Small before/after snippet, default <= 6 lines.
    """

    proposal_id: str
    target: str
    change_type: ChangeType = "param_tune"
    one_line_summary: str = ""
    before: str = ""
    after: str = ""
    hypothesis: str = ""
    risk: RiskLevel = "medium"
    max_changed_lines: int = DEFAULT_MAX_CHANGED_LINES
    patched_source: str = ""
    scale: ProposalScale = "micro"
    phase: OptimizationPhase = "tune"
    template_id: str = ""
    validation_scope: str = "single_case"
    rollback_plan: str = "discard candidate unless A/B KEEP"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def effective_max_changed_lines(self) -> int:
        if self.max_changed_lines != DEFAULT_MAX_CHANGED_LINES:
            return self.max_changed_lines
        return SCALE_DEFAULT_MAX_CHANGED_LINES.get(self.scale, DEFAULT_MAX_CHANGED_LINES)

    def changed_line_count(self) -> int:
        before_lines = [line for line in self.before.splitlines() if line.strip()]
        after_lines = [line for line in self.after.splitlines() if line.strip()]
        if not before_lines and not after_lines:
            return 0
        sm = difflib.SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
        changed = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            changed += (i2 - i1) + (j2 - j1)
        return changed

    def is_single_change(self) -> bool:
        return self.changed_line_count() <= self.effective_max_changed_lines()

    def to_diff_summary(self) -> str:
        diff = difflib.unified_diff(
            self.before.splitlines(),
            self.after.splitlines(),
            fromfile=f"{self.target} (before)",
            tofile=f"{self.target} (after)",
            lineterm="",
        )
        header = (
            f"[{self.proposal_id}] {self.one_line_summary} "
            f"({self.phase}/{self.scale}, {self.change_type}, risk={self.risk})"
        )
        parts = [header]
        if self.template_id:
            parts.append(f"template_id: {self.template_id}")
        if self.hypothesis:
            parts.append(f"hypothesis: {self.hypothesis}")
        body = "\n".join(diff)
        parts.append(body if body else "(no snippet diff; patched_source candidate)")
        return "\n".join(parts)


def validate_single_change(proposal: ChangeProposal) -> tuple[bool, str]:
    """Validate one candidate for the closed-loop A/B runner.

    The historical function name is preserved for compatibility.  It now checks
    tiered candidates, not only tiny micro patches.
    """
    if not proposal.proposal_id:
        return False, "missing proposal_id"
    if not proposal.target:
        return False, "missing target"
    if proposal.change_type not in _ALLOWED_CHANGE_TYPES:
        return False, f"unknown change_type={proposal.change_type}"
    if proposal.scale not in _ALLOWED_SCALES:
        return False, f"unknown scale={proposal.scale}"
    if proposal.phase not in _ALLOWED_PHASES:
        return False, f"unknown phase={proposal.phase}"
    if proposal.scale == "micro" and proposal.phase != "tune":
        return False, "micro proposal must use phase=tune"
    if proposal.scale == "macro" and proposal.phase == "tune":
        return False, "macro proposal must use phase=explore or stabilize"
    if proposal.scale == "macro" and proposal.change_type not in _MACRO_CHANGE_TYPES:
        return False, "macro proposal must use a structural change_type"
    if not proposal.before.strip() and not proposal.after.strip() and not proposal.patched_source.strip():
        return False, "before/after/patched_source are empty; no actual change"

    changed = proposal.changed_line_count()
    max_lines = proposal.effective_max_changed_lines()
    if changed > max_lines:
        return False, (
            f"改动 {changed} 行超过单轮上限 {max_lines} 行; "
            f"changed lines exceed {proposal.scale} limit; "
            "split into smaller rounds or provide a reviewed template candidate"
        )

    if proposal.scale in ("macro", "meso") and not proposal.rollback_plan.strip():
        return False, "macro/meso proposal must include rollback_plan"

    return True, "ok"


class PatchApplyError(ValueError):
    """Raised when a ChangeProposal cannot be applied to the current source."""


def apply_change_proposal(source: str, proposal: ChangeProposal) -> str:
    """Apply a ChangeProposal to full kernel source."""
    if proposal.patched_source:
        return proposal.patched_source

    if not proposal.before.strip():
        raise PatchApplyError(
            f"{proposal.proposal_id}: before is empty and patched_source is not provided"
        )

    occurrences = source.count(proposal.before)
    if occurrences == 0:
        raise PatchApplyError(
            f"{proposal.proposal_id}: before snippet was not found in current source"
        )
    if occurrences > 1:
        raise PatchApplyError(
            f"{proposal.proposal_id}: before snippet matched {occurrences} places; "
            "make the snippet more specific"
        )
    return source.replace(proposal.before, proposal.after, 1)
