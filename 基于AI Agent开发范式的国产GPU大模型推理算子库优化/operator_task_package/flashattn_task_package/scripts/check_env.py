#!/usr/bin/env python
"""
Environment checker for the generalized Agent optimization system.

This script is intentionally non-destructive and works on machines without a
MACA GPU.  It prints PASS/WARN/FAIL so reproduction issues are obvious before
running expensive Agent loops.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def status(ok: bool, label: str, detail: str = "", warn: bool = False) -> None:
    tag = "PASS" if ok else ("WARN" if warn else "FAIL")
    print(f"[{tag}] {label}" + (f": {detail}" if detail else ""))


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    print("FlashAttention Agent System Environment Check")
    print(f"Project: {ROOT}")
    print(f"Python: {sys.version.split()[0]}")

    required = ["torch"]
    optional = ["pytest", "flash_attn", "einops", "httpx"]
    failed = 0

    for name in required:
        ok = module_available(name)
        status(ok, f"python module {name}")
        failed += 0 if ok else 1

    for name in optional:
        ok = module_available(name)
        status(ok, f"python module {name}", warn=not ok)

    maca_path = os.environ.get("MACA_PATH", "/opt/maca")
    status(bool(maca_path), "MACA_PATH", maca_path, warn=not bool(maca_path))
    mxcc = shutil.which("mxcc") or str(Path(maca_path) / "mxgpu_llvm" / "bin" / "mxcc")
    status(Path(mxcc).exists(), "mxcc", mxcc, warn=not Path(mxcc).exists())
    mctlass_dir = Path(maca_path) / "include" / "mctlass"
    status(mctlass_dir.exists(), "mctlass headers", str(mctlass_dir), warn=not mctlass_dir.exists())

    if module_available("torch"):
        import torch
        status(torch.cuda.is_available(), "torch cuda available", str(torch.cuda.is_available()),
               warn=not torch.cuda.is_available())

    mx_smi = shutil.which("mx-smi")
    if mx_smi:
        try:
            out = subprocess.run([mx_smi], capture_output=True, text=True, timeout=5)
            first = (out.stdout or out.stderr).splitlines()[0] if (out.stdout or out.stderr) else ""
            status(out.returncode == 0, "mx-smi", first[:120], warn=out.returncode != 0)
        except Exception as exc:
            status(False, "mx-smi", str(exc), warn=True)
    else:
        status(False, "mx-smi", "not found in PATH", warn=True)

    from agent_system.operator_registry import list_operators
    specs = list_operators()
    status(bool(specs), "operator registry", f"{len(specs)} operators")
    for spec in specs:
        print(f"  - {spec.operator_id}: backend={spec.backend.kind}, cases={len(spec.test_cases)}")

    return failed


if __name__ == "__main__":
    raise SystemExit(main())
