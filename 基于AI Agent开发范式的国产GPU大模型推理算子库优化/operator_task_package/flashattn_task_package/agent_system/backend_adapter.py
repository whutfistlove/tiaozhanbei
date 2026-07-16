"""
Backend adapters for compiling and running candidate kernels.

The multi-agent framework calls this interface rather than mxcc/ctypes
directly.  MACA C++ is the first concrete backend; Triton/TileLang can be added
without changing the orchestrator.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from agent_system.kernel_compiler import compile_file, compile_source, classify_error
from agent_system.specs import BackendKind, BackendSpec
from agent_system.strategy_schema import OptimizationStrategy


@dataclass
class CompileDiagnostic:
    success: bool
    backend: BackendKind
    artifact_path: Optional[str] = None
    error: str = ""
    category: str = ""
    compile_time_s: float = 0.0


@dataclass
class LoadedArtifact:
    success: bool
    backend: BackendKind
    artifact_path: str
    handle: object | None = None
    error: str = ""


class BackendAdapter(Protocol):
    spec: BackendSpec

    def compile_strategy(self, strategy: OptimizationStrategy, workdir: Path) -> CompileDiagnostic:
        ...

    def load(self, artifact_path: str) -> LoadedArtifact:
        ...


class MacaCppBackend:
    def __init__(self, spec: BackendSpec):
        if spec.kind != "maca_cpp":
            raise ValueError(f"MacaCppBackend requires maca_cpp spec, got {spec.kind}")
        self.spec = spec

    def compile_strategy(self, strategy: OptimizationStrategy, workdir: Path) -> CompileDiagnostic:
        workdir.mkdir(parents=True, exist_ok=True)
        so_path = workdir / f"{strategy.strategy_id}.so"
        if strategy.source_code:
            result = compile_source(strategy.source_code, str(so_path))
        elif strategy.code_path:
            result = compile_file(strategy.code_path, str(so_path))
        else:
            return CompileDiagnostic(
                success=False,
                backend="maca_cpp",
                error="strategy has neither source_code nor code_path",
                category="missing_source",
            )
        if result.success:
            return CompileDiagnostic(
                success=True,
                backend="maca_cpp",
                artifact_path=result.so_path,
                compile_time_s=result.compile_time_s,
            )
        return CompileDiagnostic(
            success=False,
            backend="maca_cpp",
            error=result.error_msg,
            category=classify_error(result.stderr or result.error_msg),
            compile_time_s=result.compile_time_s,
        )

    def load(self, artifact_path: str) -> LoadedArtifact:
        # Lazy import: non-GPU/mock framework paths must not require torch.
        from agent_system.kernel_loader import load_kernel

        result = load_kernel(artifact_path)
        return LoadedArtifact(
            success=result.success,
            backend="maca_cpp",
            artifact_path=artifact_path,
            handle=result.run_kernel_fn,
            error=result.error_msg,
        )


class MockBackend:
    """Spec-only backend for generalization smoke tests."""

    def __init__(self, spec: BackendSpec):
        self.spec = spec

    def compile_strategy(self, strategy: OptimizationStrategy, workdir: Path) -> CompileDiagnostic:
        return CompileDiagnostic(success=True, backend=self.spec.kind, artifact_path=str(workdir / "mock"))

    def load(self, artifact_path: str) -> LoadedArtifact:
        return LoadedArtifact(success=True, backend=self.spec.kind, artifact_path=artifact_path, handle=None)


def make_backend_adapter(spec: BackendSpec) -> BackendAdapter:
    if spec.kind == "maca_cpp":
        return MacaCppBackend(spec)
    if spec.kind == "mock":
        return MockBackend(spec)
    raise NotImplementedError(f"backend adapter not implemented: {spec.kind}")
