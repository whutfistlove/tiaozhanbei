"""
Persistent kernel version store.

This closes an important loop: a KEEP verdict must promote code to a durable
version, not disappear after one orchestrator iteration.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class KernelVersion:
    version_id: str
    operator_id: str
    file: str
    description: str
    verdict: str
    created_at: str
    metrics: dict
    parent: Optional[str] = None


class KernelVersionStore:
    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.base_dir / "manifest_v2.json"
        self.versions: list[KernelVersion] = []
        self.current_best: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.manifest_path.exists():
            return
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.versions = [KernelVersion(**item) for item in data.get("versions", [])]
        self.current_best = dict(data.get("current_best", {}))

    def save(self) -> None:
        self.manifest_path.write_text(json.dumps({
            "versions": [asdict(v) for v in self.versions],
            "current_best": self.current_best,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_source(self, operator_id: str, source_code: str, description: str,
                   verdict: str = "PENDING", metrics: Optional[dict] = None,
                   parent: Optional[str] = None) -> KernelVersion:
        version_id = f"v{len(self.versions)+1:04d}_{int(time.time())}"
        rel_file = f"{version_id}.cu"
        (self.base_dir / rel_file).write_text(source_code, encoding="utf-8")
        version = KernelVersion(
            version_id=version_id,
            operator_id=operator_id,
            file=rel_file,
            description=description,
            verdict=verdict,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            metrics=metrics or {},
            parent=parent,
        )
        self.versions.append(version)
        self.save()
        return version

    def add_file(self, operator_id: str, source_path: Path | str, description: str,
                 verdict: str = "PENDING", metrics: Optional[dict] = None,
                 parent: Optional[str] = None) -> KernelVersion:
        src = Path(source_path)
        source_code = src.read_text(encoding="utf-8")
        return self.add_source(operator_id, source_code, description, verdict, metrics, parent)

    def promote(self, version_id: str) -> KernelVersion:
        version = self.get(version_id)
        version.verdict = "KEEP"
        self.current_best[version.operator_id] = version.version_id
        self.save()
        return version

    def get(self, version_id: str) -> KernelVersion:
        for version in self.versions:
            if version.version_id == version_id:
                return version
        raise KeyError(f"unknown kernel version: {version_id}")

    def current(self, operator_id: str) -> Optional[KernelVersion]:
        vid = self.current_best.get(operator_id)
        return self.get(vid) if vid else None

    def current_path(self, operator_id: str) -> Optional[Path]:
        current = self.current(operator_id)
        return self.base_dir / current.file if current else None
