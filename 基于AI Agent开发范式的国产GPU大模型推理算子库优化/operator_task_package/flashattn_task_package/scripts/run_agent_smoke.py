#!/usr/bin/env python
"""
Non-GPU smoke test for the generalized multi-agent framework.

It runs the generic orchestrator on the spec-only dummy operator, proving the
loop can operate on an operator other than FlashAttention without code changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_system.domain_memory import DomainMemory
from agent_system.generic_orchestrator import run_generic_iteration
from agent_system.kernel_version_store import KernelVersionStore
from agent_system.optimization_log import OptimizationLog
from agent_system.paths import new_run_dir, write_latest_run
from agent_system.specs import OperatorSpec
from agent_system.strategy_schema import OptimizationStrategy


def generate_dummy_strategy(spec: OperatorSpec, analysis: str, memory: DomainMemory):
    return [
        OptimizationStrategy(
            strategy_id="dummy_strategy_001",
            target_operator=spec.operator_id,
            backend=spec.backend.kind,
            strategy_name=spec.optimization_space.strategy_names[0],
            params=spec.optimization_space.defaults(),
            expected_effect="prove the generic loop works without a concrete GPU kernel",
            risk="low",
            required_skills=list(spec.skills),
            description="spec-only generalization smoke strategy",
        )
    ]


def main() -> int:
    workdir = new_run_dir("smoke")
    write_latest_run(workdir)
    memory = DomainMemory(base_dir=workdir / "memory")
    log = OptimizationLog(log_dir=workdir / "logs")
    versions = KernelVersionStore(workdir / "versions")

    result = run_generic_iteration(
        operator_id="dummy_bf16_gemm",
        iteration=1,
        generate_strategies_fn=generate_dummy_strategy,
        memory=memory,
        log=log,
        version_store=versions,
        workdir=workdir,
    )
    print(result.analyst_report.summary)
    print(f"verdict={result.verdict} generated={result.generated} survivors={result.survivors}")
    print(f"run_dir={workdir}")
    if result.errors:
        print("errors:")
        for err in result.errors:
            print(f"  - {err}")
    return 0 if result.verdict == "KEEP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
