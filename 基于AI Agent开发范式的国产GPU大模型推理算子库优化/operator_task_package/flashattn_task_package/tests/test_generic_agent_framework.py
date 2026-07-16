"""
Tests for the generalized multi-agent framework.

These tests do not require GPU.  They verify that the framework has moved past
a single FlashAttention-only prototype.
"""
from pathlib import Path

from agent_system.domain_memory import DomainMemory
from agent_system.generic_orchestrator import analyze_operator, run_generic_iteration
from agent_system.kernel_version_store import KernelVersionStore
from agent_system.operator_registry import get_operator, list_operators
from agent_system.optimization_log import OptimizationLog
from agent_system.strategy_schema import OptimizationStrategy, validate_strategy


def test_registry_has_flashattention_and_dummy():
    ids = {spec.operator_id for spec in list_operators()}
    assert "flashattention_kvcache_decode" in ids
    assert "dummy_bf16_gemm" in ids


def test_flashattention_spec_is_not_empty():
    spec = get_operator("flashattention_kvcache_decode")
    assert len(spec.test_cases) == 12
    assert spec.backend.kind == "maca_cpp"
    assert "split-k-pattern" in spec.skills


def test_analyze_operator_generic():
    spec = get_operator("dummy_bf16_gemm")
    report = analyze_operator(spec)
    assert "dummy_bf16_gemm" in report.summary
    assert report.role == "Analyst"


def test_strategy_validation():
    spec = get_operator("dummy_bf16_gemm")
    strategy = OptimizationStrategy(
        strategy_id="s1",
        target_operator=spec.operator_id,
        backend=spec.backend.kind,
        strategy_name=spec.optimization_space.strategy_names[0],
    )
    ok, reason = validate_strategy(strategy, spec.optimization_space.strategy_names)
    assert ok, reason


def test_generic_iteration_on_dummy_operator(tmp_path):
    spec = get_operator("dummy_bf16_gemm")
    memory = DomainMemory(base_dir=tmp_path / "memory")
    log = OptimizationLog(log_path=tmp_path / "opt.md")
    versions = KernelVersionStore(tmp_path / "versions")

    def generate(s, analysis, mem):
        return [OptimizationStrategy(
            strategy_id="s1",
            target_operator=s.operator_id,
            backend=s.backend.kind,
            strategy_name=s.optimization_space.strategy_names[0],
            params=s.optimization_space.defaults(),
            risk="low",
        )]

    result = run_generic_iteration(
        operator_id=spec.operator_id,
        iteration=1,
        generate_strategies_fn=generate,
        memory=memory,
        log=log,
        version_store=versions,
        workdir=tmp_path / "work",
    )
    assert result.verdict == "KEEP"
    assert result.generated == 1
    assert len(log.entries) == 1
