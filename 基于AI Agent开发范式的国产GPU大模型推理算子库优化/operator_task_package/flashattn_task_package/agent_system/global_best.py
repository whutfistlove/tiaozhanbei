"""Global best kernel index shared across closed-loop runs.

Run-local version stores live under ``runs/<run_id>/versions``.  This module
adds the cross-run pointer the agent needs: every new run can start from the
best KEEP-promoted source seen so far instead of restarting from the seed
kernel file.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from agent_system.paths import PROJECT_ROOT, RESULTS_DIR, ensure_output_roots


BaselineSource = Literal["auto", "best", "kernel"]

BEST_DIR = RESULTS_DIR / "best"
CURRENT_BEST_INDEX = RESULTS_DIR / "current_best.json"


def _safe_operator_id(operator_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in operator_id)[:96]


def _load_index() -> dict[str, Any]:
    if not CURRENT_BEST_INDEX.exists():
        return {"operators": {}}
    try:
        data = json.loads(CURRENT_BEST_INDEX.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"operators": {}}
    if not isinstance(data, dict):
        return {"operators": {}}
    data.setdefault("operators", {})
    return data


def _save_index(data: dict[str, Any]) -> None:
    ensure_output_roots()
    CURRENT_BEST_INDEX.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def best_source_path(operator_id: str) -> Path:
    return BEST_DIR / f"{_safe_operator_id(operator_id)}_best.cu"


def get_global_best(operator_id: str) -> dict[str, Any] | None:
    """Return global-best metadata when both index and source file are valid."""
    entry = _load_index().get("operators", {}).get(operator_id)
    if not isinstance(entry, dict):
        return None
    source_path = Path(entry.get("source_path", ""))
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    if not source_path.exists():
        return None
    entry = dict(entry)
    entry["source_path"] = str(source_path)
    return entry


def publish_global_best(
    *,
    operator_id: str,
    source_code: str,
    run_id: str,
    run_dir: Path,
    version_id: str,
    metrics: dict[str, Any] | None = None,
    description: str = "",
) -> dict[str, Any]:
    """Persist a KEEP-promoted source as the cross-run current best."""
    ensure_output_roots()
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    dst = best_source_path(operator_id)
    dst.write_text(source_code, encoding="utf-8")

    entry = {
        "operator_id": operator_id,
        "source_path": str(dst.resolve()),
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "version_id": version_id,
        "description": description,
        "metrics": metrics or {},
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    data = _load_index()
    data.setdefault("operators", {})[operator_id] = entry
    _save_index(data)
    return entry


def resolve_baseline_source(
    *,
    operator_id: str,
    kernel_path: str | Path,
    baseline_source: BaselineSource = "auto",
) -> tuple[str, Path, dict[str, Any]]:
    """Resolve the source code used as the A-side baseline for a new run.

    ``auto`` means use the global best when available, otherwise fall back to
    ``kernel_path``.  ``best`` requires a valid global best.  ``kernel`` ignores
    global state and uses the supplied seed file.
    """
    if baseline_source not in {"auto", "best", "kernel"}:
        raise ValueError("baseline_source must be one of: auto, best, kernel")

    kernel = Path(kernel_path)
    if not kernel.is_absolute():
        kernel = PROJECT_ROOT / kernel
    if not kernel.exists():
        raise FileNotFoundError(f"kernel not found: {kernel}")

    best = get_global_best(operator_id)
    if baseline_source in {"auto", "best"} and best is not None:
        source_path = Path(best["source_path"])
        return (
            source_path.read_text(encoding="utf-8"),
            source_path,
            {
                "kind": "global_best",
                "requested": baseline_source,
                "fallback_kernel_path": str(kernel),
                "global_best": best,
            },
        )

    if baseline_source == "best":
        raise FileNotFoundError(
            f"global best not found for {operator_id}; run with baseline_source=kernel "
            "or produce a KEEP first"
        )

    return (
        kernel.read_text(encoding="utf-8"),
        kernel,
        {
            "kind": "kernel",
            "requested": baseline_source,
            "fallback_kernel_path": str(kernel),
            "global_best": None,
        },
    )
