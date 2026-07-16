"""
Unified artifact paths for the operator-agent system.

The project has two stable output roots:

- runs/<run_id>/     one closed-loop optimization run
- results/          global indexes and OpenCode event logs

Each run has the same shape:

  runs/run_YYYYmmdd_HHMMSS_<tag>/
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


def ensure_output_roots() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def make_run_id(tag: str = "ab") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)[:32]
    return f"run_{ts}_{safe_tag or 'ab'}"


def prepare_run_dir(run_id: str) -> Path:
    ensure_output_roots()
    run_dir = RUNS_DIR / run_id
    for sub in RUN_SUBDIRS:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def new_run_dir(tag: str = "ab") -> Path:
    """Create and return a new run directory.

    This preserves the old public function name while moving the location from
    experiments/ to runs/.
    """
    return prepare_run_dir(make_run_id(tag))


def write_latest_run(run_dir: Path) -> None:
    ensure_output_roots()
    (RESULTS_DIR / "latest_run.txt").write_text(str(run_dir.resolve()), encoding="utf-8")


def latest_run_dir() -> Path | None:
    latest_file = RESULTS_DIR / "latest_run.txt"
    if latest_file.exists():
        p = Path(latest_file.read_text(encoding="utf-8").strip())
        if p.exists():
            return p
    if not RUNS_DIR.exists():
        return None
    runs = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name.startswith("run_"))
    return runs[-1] if runs else None


def list_run_dirs(limit: int | None = None) -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    runs = sorted(
        (p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name.startswith("run_")),
        reverse=True,
    )
    return runs if limit is None else runs[:limit]


def update_results_summary(lines: Iterable[str]) -> Path:
    ensure_output_roots()
    path = RESULTS_DIR / "summary.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path
