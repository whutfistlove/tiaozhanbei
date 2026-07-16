#!/usr/bin/env python
"""MCP tool server for the FlashAttention operator-agent system.

The server intentionally exposes deterministic tools.  OpenCode agents should
use these tools for operator specs, best-kernel resolution, proposal validation,
closed-loop A/B runs, and artifact inspection instead of inventing results in
chat.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MACA_PATH", "/opt/maca")


def tool_compile_kernel(source_path: str, output_so: str = "") -> dict:
    """Compile a .cu source file into a shared object."""
    from agent_system.kernel_compiler import compile_file

    src = Path(source_path)
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists():
        return {"success": False, "error": f"source file not found: {src}"}

    out = output_so or str(src.with_suffix(".so"))
    result = compile_file(str(src), out)
    return {
        "success": result.success,
        "so_path": result.so_path,
        "error": result.error_msg[:500] if not result.success else "",
        "compile_time_s": result.compile_time_s,
    }


def tool_run_correctness(
    so_path: str,
    batch: int = 1,
    seq_kv: int = 4096,
    headdim: int = 128,
    num_heads: int = 8,
    num_heads_k: int = 8,
) -> dict:
    """Run a compiled kernel and compare it with the PyTorch reference."""
    import torch

    if not torch.cuda.is_available():
        return {"success": False, "error": "CUDA/MACA GPU is not available"}

    from agent_system.correctness import check, generate_reference, make_test_inputs
    from agent_system.kernel_loader import call_run_kernel, load_kernel, make_output_tensor
    from agent_system.roofline_engine import KernelConfig

    lres = load_kernel(so_path)
    if not lres.success:
        return {"success": False, "error": f"load failed: {lres.error_msg}"}

    cfg = KernelConfig(
        batch_size=batch,
        seqlen_kv=seq_kv,
        headdim=headdim,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
    )
    q, k, v, lens, bt = make_test_inputs(cfg, device="cuda", seed=42)
    try:
        out = call_run_kernel(
            lres.run_kernel_fn,
            q,
            k,
            v,
            make_output_tensor(cfg, "cuda"),
            lens,
            bt,
            cfg,
            k.shape[0],
        )
        ref = generate_reference(q, k, v, lens, bt, cfg)
        result = check(out, ref)
        return {
            "success": True,
            "passed": result.passed,
            "detail": result.detail,
            "max_abs": result.max_abs_diff,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)[:300]}


def tool_run_benchmark(
    so_path: str,
    batch: int = 1,
    seq_kv: int = 4096,
    headdim: int = 128,
    num_heads: int = 8,
    num_heads_k: int = 8,
    warmup: int = 5,
    repeats: int = 30,
) -> dict:
    """Benchmark a compiled kernel on the active GPU."""
    import torch

    if not torch.cuda.is_available():
        return {"success": False, "error": "CUDA/MACA GPU is not available"}

    from agent_system.benchmark_engine import benchmark_config
    from agent_system.correctness import make_test_inputs
    from agent_system.kernel_loader import call_run_kernel, load_kernel, make_output_tensor
    from agent_system.roofline_engine import KernelConfig

    lres = load_kernel(so_path)
    if not lres.success:
        return {"success": False, "error": f"load failed: {lres.error_msg}"}

    cfg = KernelConfig(
        batch_size=batch,
        seqlen_kv=seq_kv,
        headdim=headdim,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
    )
    q, k, v, lens, bt = make_test_inputs(cfg, device="cuda", seed=42)

    def run() -> None:
        call_run_kernel(
            lres.run_kernel_fn,
            q,
            k,
            v,
            make_output_tensor(cfg, "cuda"),
            lens,
            bt,
            cfg,
            k.shape[0],
        )

    bench = benchmark_config(run, cfg, warmup=warmup, repeats=repeats)
    return {
        "success": True,
        "time_ms": bench.time_ms,
        "bandwidth_gb_s": bench.achievable_bw_gb_s,
        "utilization": bench.bandwidth_utilization,
        "gap_to_roofline": bench.gap_to_roofline,
        "bound_type": bench.bound_type,
        "summary": bench.summary(),
    }


def tool_roofline_analyze(
    batch: int = 1,
    seq_kv: int = 4096,
    headdim: int = 128,
    num_heads: int = 8,
    num_heads_k: int = 8,
) -> dict:
    """Return the roofline estimate for one decode case."""
    from agent_system.roofline_engine import KernelConfig, analyze, suggest_split_k

    cfg = KernelConfig(
        batch_size=batch,
        seqlen_kv=seq_kv,
        headdim=headdim,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
    )
    r = analyze(cfg)
    return {
        "success": True,
        "bound_type": r.bound_type,
        "arithmetic_intensity": r.arithmetic_intensity,
        "t_lower_bound_ms": r.t_lower_bound_s * 1e3,
        "flops": r.flops,
        "bytes": r.bytes,
        "suggested_split_k": suggest_split_k(cfg),
    }


def tool_record_iteration(
    iteration: int,
    description: str,
    verdict: str,
    baseline_ms: float,
    candidate_ms: float | None = None,
    speedup: float | None = None,
    correctness: bool | None = None,
    notes: str = "",
    run_dir: str = "",
    category: str = "",
    target_config: str = "via MCP",
) -> dict:
    """Append one structured optimization-log entry."""
    from agent_system.optimization_log import OptimizationEntry, OptimizationLog

    log_dir = Path(run_dir) / "logs" if run_dir else ROOT / "results" / "manual_log"
    log = OptimizationLog(log_dir=log_dir)
    entry = OptimizationEntry(
        iteration=iteration,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        change_description=description,
        target_config=target_config,
        baseline_time_ms=baseline_ms,
        candidate_time_ms=candidate_ms,
        speedup=speedup,
        bandwidth_util_before=0,
        bandwidth_util_after=None,
        correctness_passed=correctness,
        verdict=verdict,
        notes=notes,
        category=category,
    )
    log.record(entry)
    return {"success": True, "total_entries": len(log.entries), "log_dir": str(log.log_dir)}


def tool_query_memory(keyword: str = "") -> dict:
    """Return domain-memory context for agents."""
    from agent_system.domain_memory import DomainMemory

    mem = DomainMemory(base_dir=ROOT / "agent_system" / "domain_memory")
    return {
        "success": True,
        "failures_count": len(mem.failures),
        "beliefs_count": len(mem.belief.entries),
        "context": mem.build_context(keyword)[:1000] if keyword else "",
    }


def tool_list_operators() -> dict:
    """List registered operator specs."""
    from agent_system.operator_registry import list_operators

    specs = list_operators()
    return {
        "success": True,
        "operators": [
            {
                "operator_id": spec.operator_id,
                "display_name": spec.display_name,
                "category": spec.category,
                "backend": spec.backend.kind,
                "cases": len(spec.test_cases),
                "expected_bound": spec.expected_bound,
                "skills": list(spec.skills),
            }
            for spec in specs
        ],
    }


def tool_analyze_operator(operator_id: str = "flashattention_kvcache_decode") -> dict:
    """Run the generic Analyst pass for an operator."""
    from agent_system.generic_orchestrator import analyze_operator
    from agent_system.operator_registry import get_operator

    spec = get_operator(operator_id)
    report = analyze_operator(spec)
    return {
        "success": True,
        "operator_id": operator_id,
        "summary": report.summary,
        "artifacts": report.artifacts,
    }


def tool_get_operator_spec(
    operator_id: str = "flashattention_kvcache_decode",
    full: bool = False,
) -> dict:
    """Return an OperatorSpec.  By default this is the agent-sized summary."""
    from agent_system.operator_registry import get_operator

    spec = get_operator(operator_id)
    spec_dict = spec.to_dict()
    if full:
        return {"success": True, "operator": spec_dict}
    return {
        "success": True,
        "operator": {
            "operator_id": spec.operator_id,
            "display_name": spec.display_name,
            "category": spec.category,
            "interface": spec.interface,
            "backend": spec_dict["backend"],
            "evaluation": spec_dict["evaluation"],
            "test_cases": spec_dict["test_cases"],
            "optimization_space": {
                "strategy_names": list(spec.optimization_space.strategy_names),
                "params": [
                    {
                        "name": p.name,
                        "values": list(p.values),
                        "default": p.default,
                        "description": p.description,
                        "risk": p.risk,
                    }
                    for p in spec.optimization_space.params
                ],
            },
            "skills": list(spec.skills),
            "metadata": spec.metadata,
        },
    }


def tool_validate_strategy(strategy: dict, operator_id: str = "") -> dict:
    """Validate an OptimizationStrategy against the target OperatorSpec."""
    from agent_system.operator_registry import get_operator
    from agent_system.strategy_schema import OptimizationStrategy, validate_strategy

    try:
        parsed = OptimizationStrategy(**strategy)
    except Exception as exc:
        return {"success": False, "valid": False, "error": f"schema error: {exc}"}

    target = operator_id or parsed.target_operator
    try:
        spec = get_operator(target)
    except Exception as exc:
        return {"success": False, "valid": False, "error": str(exc)}

    ok, reason = validate_strategy(parsed, spec.optimization_space.strategy_names)
    allowed_params = {p.name for p in spec.optimization_space.params}
    unknown_params = sorted(set(parsed.params) - allowed_params - {"strategy_name"})
    if unknown_params:
        ok = False
        reason = f"unknown params for {target}: {unknown_params}"

    return {
        "success": True,
        "valid": ok,
        "reason": reason,
        "operator_id": target,
        "strategy_name": parsed.strategy_name,
        "unknown_params": unknown_params,
    }


def tool_run_agent_smoke() -> dict:
    """Run the non-GPU generic multi-agent smoke test."""
    import tempfile

    from agent_system.domain_memory import DomainMemory
    from agent_system.generic_orchestrator import run_generic_iteration
    from agent_system.kernel_version_store import KernelVersionStore
    from agent_system.optimization_log import OptimizationLog
    from agent_system.strategy_schema import OptimizationStrategy

    def generate(spec, analysis, memory):
        return [
            OptimizationStrategy(
                strategy_id="mcp_smoke_strategy",
                target_operator=spec.operator_id,
                backend=spec.backend.kind,
                strategy_name=spec.optimization_space.strategy_names[0],
                params=spec.optimization_space.defaults(),
                expected_effect="MCP smoke for generalized multi-agent loop",
                risk="low",
                required_skills=list(spec.skills),
                description="MCP spec-only smoke strategy",
            )
        ]

    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        result = run_generic_iteration(
            operator_id="dummy_bf16_gemm",
            iteration=1,
            generate_strategies_fn=generate,
            memory=DomainMemory(base_dir=workdir / "memory"),
            log=OptimizationLog(log_dir=workdir / "logs"),
            version_store=KernelVersionStore(workdir / "versions"),
            workdir=workdir / "work",
        )
    return {
        "success": result.verdict == "KEEP",
        "verdict": result.verdict,
        "operator_id": result.operator_id,
        "generated": result.generated,
        "survivors": result.survivors,
        "compile_ok": result.compile_ok,
        "analysis": result.analyst_report.summary,
        "errors": result.errors,
    }


def tool_list_kernel_versions(operator_id: str = "") -> dict:
    """List the legacy persistent kernel-version store."""
    from agent_system.kernel_version_store import KernelVersionStore

    store = KernelVersionStore(ROOT / "agent_system" / "kernel_versions")
    versions = store.versions
    if operator_id:
        versions = [v for v in versions if v.operator_id == operator_id]
    return {
        "success": True,
        "current_best": store.current_best,
        "versions": [
            {
                "version_id": v.version_id,
                "operator_id": v.operator_id,
                "file": v.file,
                "description": v.description,
                "verdict": v.verdict,
                "created_at": v.created_at,
                "metrics": v.metrics,
                "parent": v.parent,
            }
            for v in versions
        ],
    }


def tool_current_best_kernel(operator_id: str = "flashattention_kvcache_decode") -> dict:
    """Return the cross-run current best kernel pointer."""
    from agent_system.global_best import get_global_best

    best = get_global_best(operator_id)
    return {
        "success": True,
        "operator_id": operator_id,
        "has_global_best": best is not None,
        "global_best": best,
    }


def tool_prepare_proposal_artifact(tag: str = "coder") -> dict:
    """Create a unique proposal-artifact path for one Coder attempt."""
    from agent_system.paths import RESULTS_DIR, ensure_output_roots

    ensure_output_roots()
    safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)[:32] or "coder"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    millis = int((time.time() % 1) * 1000)
    artifact_dir = RESULTS_DIR / "agent_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{stamp}_{millis:03d}_{safe_tag}"
    return {
        "success": True,
        "artifact_dir": str(artifact_dir),
        "proposal_path": str(artifact_dir / f"{stem}_proposal.json"),
        "error_path": str(artifact_dir / f"{stem}_coder_error.json"),
    }


def tool_resolve_best_kernel(
    operator_id: str = "flashattention_kvcache_decode",
    kernel_path: str = "kernel/splitk_h128.cu",
    baseline_source: str = "auto",
) -> dict:
    """Resolve the baseline source path that the next run will analyze."""
    from agent_system.global_best import resolve_baseline_source

    source, path, meta = resolve_baseline_source(
        operator_id=operator_id,
        kernel_path=kernel_path,
        baseline_source=baseline_source,
    )
    return {
        "success": True,
        "operator_id": operator_id,
        "baseline_path": str(path),
        "baseline_kind": meta.get("kind"),
        "baseline_source": baseline_source,
        "global_best": meta.get("global_best"),
        "source_bytes": len(source.encode("utf-8")),
    }


def tool_validate_change_proposal(
    proposal_path: str,
    operator_id: str = "flashattention_kvcache_decode",
    kernel_path: str = "kernel/splitk_h128.cu",
    baseline_source: str = "auto",
) -> dict:
    """Validate that a Coder proposal artifact exists and applies to baseline."""
    from agent_system.closed_loop import load_proposals
    from agent_system.global_best import resolve_baseline_source
    from agent_system.strategy_schema import PatchApplyError, apply_change_proposal, validate_single_change

    path = Path(proposal_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return {"success": False, "valid": False, "reason": f"proposal artifact not found: {path}"}

    source, baseline_path, meta = resolve_baseline_source(
        operator_id=operator_id,
        kernel_path=kernel_path,
        baseline_source=baseline_source,
    )
    try:
        proposals = load_proposals(path)
    except Exception as exc:
        return {
            "success": False,
            "valid": False,
            "reason": f"proposal JSON parse/schema error: {exc}",
            "proposal_path": str(path),
        }
    if not proposals:
        return {
            "success": False,
            "valid": False,
            "reason": "proposal artifact contains no ChangeProposal",
            "proposal_path": str(path),
        }

    checked = []
    current = source
    for idx, proposal in enumerate(proposals, start=1):
        ok, reason = validate_single_change(proposal)
        if not ok:
            return {
                "success": False,
                "valid": False,
                "reason": f"proposal {idx} invalid: {reason}",
                "proposal_id": proposal.proposal_id,
                "proposal_path": str(path),
                "baseline_path": str(baseline_path),
            }
        try:
            current = apply_change_proposal(current, proposal)
        except PatchApplyError as exc:
            return {
                "success": False,
                "valid": False,
                "reason": str(exc),
                "proposal_id": proposal.proposal_id,
                "proposal_path": str(path),
                "baseline_path": str(baseline_path),
            }
        checked.append({
            "proposal_id": proposal.proposal_id,
            "phase": proposal.phase,
            "scale": proposal.scale,
            "target": proposal.target,
        })

    return {
        "success": True,
        "valid": True,
        "proposal_path": str(path),
        "baseline_path": str(baseline_path),
        "baseline_kind": meta.get("kind"),
        "count": len(proposals),
        "checked": checked,
    }


def tool_run_closed_loop(
    operator_id: str = "flashattention_kvcache_decode",
    kernel_path: str = "kernel/splitk_h128.cu",
    rounds: int = 1,
    dry_run: bool = True,
    proposal_path: str = "",
    phase: str = "tune",
    tag: str = "mcp_loop",
    batch: int = 1,
    seq_kv: int = 4096,
    headdim: int = 128,
    num_heads: int = 8,
    num_heads_k: int = 8,
    baseline_source: str = "auto",
    proposal_required: bool = True,
    allow_auto_proposal: bool = False,
) -> dict:
    """Run the deterministic closed-loop A/B pipeline."""
    from agent_system.closed_loop import run_closed_loop

    result = run_closed_loop(
        operator_id=operator_id,
        kernel_path=kernel_path,
        rounds=rounds,
        dry_run=dry_run,
        proposal_path=proposal_path or None,
        phase=phase,
        tag=tag,
        batch=batch,
        seq_kv=seq_kv,
        headdim=headdim,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        baseline_source=baseline_source,
        proposal_required=proposal_required,
        allow_auto_proposal=allow_auto_proposal,
    )
    data = result.to_dict()
    data["success"] = result.status == "completed"
    return data


def tool_list_runs(limit: int = 10) -> dict:
    """List recent closed-loop run directories."""
    from agent_system.paths import list_run_dirs

    runs = list_run_dirs(limit=limit)
    return {"success": True, "runs": [{"run_id": p.name, "run_dir": str(p)} for p in runs]}


def tool_latest_run() -> dict:
    """Return the latest run and key artifact locations."""
    from agent_system.paths import latest_run_dir

    run = latest_run_dir()
    if run is None:
        return {"success": True, "run": None}
    return {
        "success": True,
        "run": {
            "run_id": run.name,
            "run_dir": str(run),
            "manifest": str(run / "run_manifest.json"),
            "events": str(run / "events.jsonl"),
            "logs": str(run / "logs"),
            "rounds": str(run / "rounds"),
            "versions": str(run / "versions"),
        },
    }


TOOLS = {
    "compile_kernel": tool_compile_kernel,
    "run_correctness": tool_run_correctness,
    "run_benchmark": tool_run_benchmark,
    "roofline_analyze": tool_roofline_analyze,
    "record_iteration": tool_record_iteration,
    "query_memory": tool_query_memory,
    "list_operators": tool_list_operators,
    "analyze_operator": tool_analyze_operator,
    "get_operator_spec": tool_get_operator_spec,
    "validate_strategy": tool_validate_strategy,
    "run_agent_smoke": tool_run_agent_smoke,
    "list_kernel_versions": tool_list_kernel_versions,
    "current_best_kernel": tool_current_best_kernel,
    "prepare_proposal_artifact": tool_prepare_proposal_artifact,
    "resolve_best_kernel": tool_resolve_best_kernel,
    "validate_change_proposal": tool_validate_change_proposal,
    "run_closed_loop": tool_run_closed_loop,
    "list_runs": tool_list_runs,
    "latest_run": tool_latest_run,
}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "compile_kernel",
        "description": "Compile .cu to .so",
        "inputSchema": {
            "type": "object",
            "properties": {"source_path": {"type": "string"}, "output_so": {"type": "string"}},
            "required": ["source_path"],
        },
    },
    {
        "name": "run_correctness",
        "description": "Run allclose correctness check",
        "inputSchema": {
            "type": "object",
            "properties": {
                "so_path": {"type": "string"},
                "batch": {"type": "integer"},
                "seq_kv": {"type": "integer"},
                "headdim": {"type": "integer"},
                "num_heads": {"type": "integer"},
                "num_heads_k": {"type": "integer"},
            },
            "required": ["so_path"],
        },
    },
    {
        "name": "run_benchmark",
        "description": "Run GPU benchmark",
        "inputSchema": {
            "type": "object",
            "properties": {
                "so_path": {"type": "string"},
                "batch": {"type": "integer"},
                "seq_kv": {"type": "integer"},
                "headdim": {"type": "integer"},
                "num_heads": {"type": "integer"},
                "num_heads_k": {"type": "integer"},
                "warmup": {"type": "integer"},
                "repeats": {"type": "integer"},
            },
            "required": ["so_path"],
        },
    },
    {
        "name": "roofline_analyze",
        "description": "Roofline analysis without running GPU code",
        "inputSchema": {
            "type": "object",
            "properties": {
                "batch": {"type": "integer"},
                "seq_kv": {"type": "integer"},
                "headdim": {"type": "integer"},
                "num_heads": {"type": "integer"},
                "num_heads_k": {"type": "integer"},
            },
        },
    },
    {
        "name": "record_iteration",
        "description": "Append an optimization-log entry",
        "inputSchema": {
            "type": "object",
            "properties": {
                "iteration": {"type": "integer"},
                "description": {"type": "string"},
                "verdict": {"type": "string"},
                "baseline_ms": {"type": "number"},
            },
            "required": ["iteration", "description", "verdict", "baseline_ms"],
        },
    },
    {
        "name": "query_memory",
        "description": "Query domain memory",
        "inputSchema": {"type": "object", "properties": {"keyword": {"type": "string"}}},
    },
    {"name": "list_operators", "description": "List registered operators", "inputSchema": {"type": "object", "properties": {}}},
    {
        "name": "analyze_operator",
        "description": "Run generic Analyst for an operator",
        "inputSchema": {"type": "object", "properties": {"operator_id": {"type": "string"}}},
    },
    {
        "name": "get_operator_spec",
        "description": "Read OperatorSpec",
        "inputSchema": {
            "type": "object",
            "properties": {"operator_id": {"type": "string"}, "full": {"type": "boolean"}},
        },
    },
    {
        "name": "validate_strategy",
        "description": "Validate OptimizationStrategy",
        "inputSchema": {
            "type": "object",
            "properties": {"operator_id": {"type": "string"}, "strategy": {"type": "object"}},
            "required": ["strategy"],
        },
    },
    {"name": "run_agent_smoke", "description": "Run non-GPU generic agent smoke test", "inputSchema": {"type": "object", "properties": {}}},
    {
        "name": "list_kernel_versions",
        "description": "List legacy KEEP kernel versions",
        "inputSchema": {"type": "object", "properties": {"operator_id": {"type": "string"}}},
    },
    {
        "name": "current_best_kernel",
        "description": "Show cross-run current best kernel",
        "inputSchema": {"type": "object", "properties": {"operator_id": {"type": "string"}}},
    },
    {
        "name": "prepare_proposal_artifact",
        "description": "Allocate a unique file path for the next Coder proposal",
        "inputSchema": {"type": "object", "properties": {"tag": {"type": "string"}}},
    },
    {
        "name": "resolve_best_kernel",
        "description": "Resolve baseline kernel for the next closed-loop run",
        "inputSchema": {
            "type": "object",
            "properties": {
                "operator_id": {"type": "string"},
                "kernel_path": {"type": "string"},
                "baseline_source": {"type": "string", "enum": ["auto", "best", "kernel"]},
            },
        },
    },
    {
        "name": "validate_change_proposal",
        "description": "Validate a Coder ChangeProposal artifact",
        "inputSchema": {
            "type": "object",
            "properties": {
                "proposal_path": {"type": "string"},
                "operator_id": {"type": "string"},
                "kernel_path": {"type": "string"},
                "baseline_source": {"type": "string", "enum": ["auto", "best", "kernel"]},
            },
            "required": ["proposal_path"],
        },
    },
    {
        "name": "run_closed_loop",
        "description": "Run proposal -> patch -> compile/test/benchmark -> A/B -> logs",
        "inputSchema": {
            "type": "object",
            "properties": {
                "operator_id": {"type": "string"},
                "kernel_path": {"type": "string"},
                "rounds": {"type": "integer"},
                "dry_run": {"type": "boolean"},
                "proposal_path": {"type": "string"},
                "phase": {"type": "string", "enum": ["explore", "stabilize", "tune"]},
                "tag": {"type": "string"},
                "batch": {"type": "integer"},
                "seq_kv": {"type": "integer"},
                "headdim": {"type": "integer"},
                "num_heads": {"type": "integer"},
                "num_heads_k": {"type": "integer"},
                "baseline_source": {"type": "string", "enum": ["auto", "best", "kernel"]},
                "proposal_required": {"type": "boolean"},
                "allow_auto_proposal": {"type": "boolean"},
            },
        },
    },
    {
        "name": "list_runs",
        "description": "List recent closed-loop runs",
        "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer"}}},
    },
    {"name": "latest_run", "description": "Show latest closed-loop run", "inputSchema": {"type": "object", "properties": {}}},
]


def handle_request(req: dict) -> dict:
    method = req.get("method", "")
    req_id = req.get("id")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "flashattn-agent-tools", "version": "1.1"},
            },
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOL_SCHEMAS}}
    if method == "tools/call":
        params = req.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        if name not in TOOLS:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"unknown tool: {name}"},
            }
        try:
            result = TOOLS[name](**args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]},
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"error": str(exc)}, ensure_ascii=False)}],
                    "isError": True,
                },
            }
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"unknown method: {method}"},
    }


if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = handle_request(request)
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32000, "message": str(exc)},
            }
        print(json.dumps(response, ensure_ascii=False), flush=True)
