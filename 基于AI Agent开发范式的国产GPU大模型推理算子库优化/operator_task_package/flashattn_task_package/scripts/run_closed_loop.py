#!/usr/bin/env python
"""Run the deterministic operator-agent closed loop.

Default mode is dry-run, which verifies proposal patching, run directory layout,
logging and version manifests without requiring torch/MACA/GPU.  Use --real on
the target GPU machine to run compile -> correctness -> benchmark -> A/B judge.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_system.closed_loop import DEFAULT_KERNEL, result_to_json, run_closed_loop


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="closed-loop multi-agent A/B runner")
    p.add_argument("--operator", default="flashattention_kvcache_decode")
    p.add_argument("--kernel", default=str(DEFAULT_KERNEL.relative_to(ROOT)))
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--proposal", default="", help="JSON object/list of ChangeProposal entries")
    p.add_argument("--phase", choices=["explore", "stabilize", "tune"], default="tune")
    p.add_argument("--tag", default="closed_loop")
    p.add_argument("--real", action="store_true", help="run real compile/correctness/benchmark")
    p.add_argument("--baseline-source", choices=["auto", "best", "kernel"], default="auto")
    p.add_argument(
        "--proposal-required",
        action="store_true",
        help="fail if Coder did not provide enough proposal artifacts",
    )
    p.add_argument(
        "--no-auto-proposal",
        action="store_true",
        help="disable built-in smoke proposals; use with --proposal-required for real agent runs",
    )
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-kv", type=int, default=4096)
    p.add_argument("--headdim", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-heads-k", type=int, default=8)
    p.add_argument("--noise-margin", type=float, default=0.03)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=30)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    result = run_closed_loop(
        operator_id=args.operator,
        kernel_path=args.kernel,
        rounds=args.rounds,
        dry_run=not args.real,
        proposal_path=args.proposal or None,
        phase=args.phase,
        tag=args.tag,
        batch=args.batch,
        seq_kv=args.seq_kv,
        headdim=args.headdim,
        num_heads=args.num_heads,
        num_heads_k=args.num_heads_k,
        noise_margin=args.noise_margin,
        warmup=args.warmup,
        repeats=args.repeats,
        baseline_source=args.baseline_source,
        proposal_required=args.proposal_required,
        allow_auto_proposal=not args.no_auto_proposal,
    )
    print(result_to_json(result))
    print(f"\nrun_dir: {result.run_dir}")
    print(f"logs:    {result.logs_dir}")
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
