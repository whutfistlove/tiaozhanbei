"""Unified artifact paths for the operator-agent system.

New artifacts are bucketed by operator so FlashAttention, MoE, and future
tasks do not mix runs or agent outputs:

- runs/<operator_id>/<run_id>/       one closed-loop optimization run
- results/<operator_id>/             best source, proposal artifacts and logs
- results/current_best.json          cross-operator best index

Each run has the same shape:

  runs/<operator_id>/run_YYYYmmdd_HHMMSS_<tag>/
    run_manifest.json
    events.jsonl
    versions/       KEEP-promoted source snapshots + manifest_v2.json
    logs/           optimization_log.md/jsonl, kept/rejected/errors/summary
    memory/         failure cases and hardware beliefs for this run
    rounds/
      round_001/
        baseline_a.cu
        analysis.json
        proposal.json
        cand_<proposal>.cu
        decision.json

Keep this module as the single source of truth so agents and scripts do not
scatter logs into the repository root.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Backward-compatible alias for older scripts that imported EXPERIMENTS_DIR.
EXPERIMENTS_DIR = RUNS_DIR

RUN_SUBDIRS = ("versions", "logs", "memory", "rounds")


def safe_operator_id(operator_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in operator_id)
    return (safe or "unknown_operator")[:96]


def operator_runs_dir(operator_id: str = "") -> Path:
    return RUNS_DIR / safe_operator_id(operator_id) if operator_id else RUNS_DIR


def operator_results_dir(operator_id: str = "") -> Path:
    return RESULTS_DIR / safe_operator_id(operator_id) if operator_id else RESULTS_DIR


def ensure_output_roots(operator_id: str = "") -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if operator_id:
        operator_runs_dir(operator_id).mkdir(parents=True, exist_ok=True)
        operator_results_dir(operator_id).mkdir(parents=True, exist_ok=True)


def make_run_id(tag: str = "ab") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)[:32]
    return f"run_{ts}_{safe_tag or 'ab'}"


def prepare_run_dir(run_id: str, operator_id: str = "") -> Path:
    ensure_output_roots(operator_id)
    run_dir = operator_runs_dir(operator_id) / run_id
    for sub in RUN_SUBDIRS:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def new_run_dir(tag: str = "ab", operator_id: str = "") -> Path:
    """Create and return a new run directory.

    This preserves the old public function name while moving the location from
    experiments/ to runs/. Passing operator_id creates runs/<operator_id>/...
    """
    return prepare_run_dir(make_run_id(tag), operator_id=operator_id)


def _write_latest_file(path: Path, run_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(run_dir.resolve()), encoding="utf-8")


def write_latest_run(run_dir: Path, operator_id: str = "") -> None:
    ensure_output_roots(operator_id)
    _write_latest_file(RESULTS_DIR / "latest_run.txt", run_dir)
    if operator_id:
        _write_latest_file(operator_results_dir(operator_id) / "latest_run.txt", run_dir)


def _scan_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    direct = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    nested = [
        p
        for bucket in root.iterdir()
        if bucket.is_dir() and not bucket.name.startswith("run_")
        for p in bucket.iterdir()
        if p.is_dir() and p.name.startswith("run_")
    ]
    return sorted(direct + nested)


def latest_run_dir(operator_id: str = "") -> Path | None:
    latest_file = (
        operator_results_dir(operator_id) / "latest_run.txt"
        if operator_id
        else RESULTS_DIR / "latest_run.txt"
    )
    if latest_file.exists():
        p = Path(latest_file.read_text(encoding="utf-8").strip())
        if p.exists():
            return p
    runs = _scan_run_dirs(operator_runs_dir(operator_id))
    return runs[-1] if runs else None


def list_run_dirs(limit: int | None = None, operator_id: str = "") -> list[Path]:
    runs = sorted(_scan_run_dirs(operator_runs_dir(operator_id)), reverse=True)
    return runs if limit is None else runs[:limit]


def update_results_summary(lines: Iterable[str], operator_id: str = "") -> Path:
    ensure_output_roots(operator_id)
    text = "\n".join(lines).rstrip() + "\n"
    path = operator_results_dir(operator_id) / "summary.md"
    path.write_text(text, encoding="utf-8")
    if operator_id:
        (RESULTS_DIR / "summary.md").write_text(text, encoding="utf-8")
    return path
