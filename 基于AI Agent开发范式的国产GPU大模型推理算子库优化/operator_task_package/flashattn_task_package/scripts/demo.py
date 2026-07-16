#!/usr/bin/env python
"""Non-GPU demo for the closed-loop operator-agent system."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_system.closed_loop import run_closed_loop
from agent_system.roofline_engine import KernelConfig, analyze, suggest_split_k


def section(title: str) -> None:
    print(f"\n{'=' * 64}\n  {title}\n{'=' * 64}")


def main() -> int:
    section("Roofline quick view")
    for cfg in [
        KernelConfig(batch_size=1, seqlen_kv=1024),
        KernelConfig(batch_size=1, seqlen_kv=4096),
        KernelConfig(batch_size=16, seqlen_kv=4096),
    ]:
        r = analyze(cfg)
        print(
            f"b={cfg.batch_size:<2} seq={cfg.seqlen_kv:<5} "
            f"bound={r.bound_type:<12} lower={r.t_lower_bound_s * 1e3:.4f}ms "
            f"suggest_split_k={suggest_split_k(cfg)}"
        )

    section("Closed-loop dry-run")
    result = run_closed_loop(rounds=1, dry_run=True, tag="demo")
    print(f"status: {result.status}")
    print(f"run_dir: {result.run_dir}")
    print(f"logs:    {result.logs_dir}")
    print(f"rounds:  {result.rounds}")
    print("Dry-run validates proposal patching and artifact layout; use --real for A/B performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
