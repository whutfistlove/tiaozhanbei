#!/usr/bin/env python
"""
MCP Server —— Agent 工具层。

把 agent_system 的核心能力封装为 MCP（Model Context Protocol）工具，
让 OpenCode Agent 通过标准协议调用（而非手写 shell 命令）。

工具清单：
- compile_kernel: 编译 .cu 源码为 .so
- run_correctness: 运行 kernel + allclose 校验
- run_benchmark: GPU 计时 + 带宽分析
- roofline_analyze: 计算 roofline 理论下限
- record_iteration: 记录到优化日志
- query_memory: 查询领域记忆库（失败案例+信念）

启动：python mcp/mcp_server.py（被 opencode.json 的 mcp 配置拉起）
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

# 确保能 import agent_system
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MACA_PATH", "/opt/maca")


def tool_compile_kernel(source_path: str, output_so: str = "") -> dict:
    """编译 .cu 文件为 .so。"""
    from agent_system.kernel_compiler import compile_file
    src = Path(source_path)
    if not src.exists():
        return {"success": False, "error": f"文件不存在: {source_path}"}
    out = output_so or str(src.with_suffix(".so"))
    result = compile_file(source_path, out)
    return {
        "success": result.success,
        "so_path": result.so_path,
        "error": result.error_msg[:500] if not result.success else "",
        "compile_time_s": result.compile_time_s,
    }


def tool_run_correctness(so_path: str, batch: int = 1, seq_kv: int = 4096,
                          headdim: int = 128, num_heads: int = 8) -> dict:
    """运行 kernel + allclose 正确性校验。"""
    import torch
    if not torch.cuda.is_available():
        return {"success": False, "error": "GPU 不可用"}
    from agent_system.kernel_loader import load_kernel, call_run_kernel, make_output_tensor
    from agent_system.correctness import make_test_inputs, generate_reference, check
    from agent_system.roofline_engine import KernelConfig

    lres = load_kernel(so_path)
    if not lres.success:
        return {"success": False, "error": f"加载失败: {lres.error_msg}"}
    cfg = KernelConfig(batch_size=batch, seqlen_kv=seq_kv, headdim=headdim, num_heads=num_heads)
    q, k, v, lens, bt = make_test_inputs(cfg, device="cuda", seed=42)
    try:
        out = call_run_kernel(lres.run_kernel_fn, q, k, v, make_output_tensor(cfg, "cuda"),
                              lens, bt, cfg, k.shape[0])
        ref = generate_reference(q, k, v, lens, bt, cfg)
        result = check(out, ref)
        return {"success": True, "passed": result.passed, "detail": result.detail,
                "max_abs": result.max_abs_diff}
    except Exception as e:
        return {"success": False, "error": str(e)[:200]}


def tool_run_benchmark(so_path: str, batch: int = 1, seq_kv: int = 4096,
                       headdim: int = 128, num_heads: int = 8,
                       warmup: int = 5, repeats: int = 30) -> dict:
    """GPU 计时 + 带宽分析。"""
    import torch
    if not torch.cuda.is_available():
        return {"success": False, "error": "GPU 不可用"}
    from agent_system.kernel_loader import load_kernel, call_run_kernel, make_output_tensor
    from agent_system.benchmark_engine import benchmark_config
    from agent_system.correctness import make_test_inputs
    from agent_system.roofline_engine import KernelConfig

    lres = load_kernel(so_path)
    if not lres.success:
        return {"success": False, "error": f"加载失败: {lres.error_msg}"}
    cfg = KernelConfig(batch_size=batch, seqlen_kv=seq_kv, headdim=headdim, num_heads=num_heads)
    q, k, v, lens, bt = make_test_inputs(cfg, device="cuda", seed=42)

    def run():
        call_run_kernel(lres.run_kernel_fn, q, k, v, make_output_tensor(cfg, "cuda"),
                        lens, bt, cfg, k.shape[0])
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


def tool_roofline_analyze(batch: int = 1, seq_kv: int = 4096,
                          headdim: int = 128, num_heads: int = 8) -> dict:
    """计算 roofline 理论下限（不依赖 GPU）。"""
    from agent_system.roofline_engine import KernelConfig, analyze, suggest_split_k
    cfg = KernelConfig(batch_size=batch, seqlen_kv=seq_kv, headdim=headdim, num_heads=num_heads)
    r = analyze(cfg)
    return {
        "bound_type": r.bound_type,
        "arithmetic_intensity": r.arithmetic_intensity,
        "t_lower_bound_ms": r.t_lower_bound_s * 1e3,
        "flops": r.flops,
        "bytes": r.bytes,
        "suggested_split_k": suggest_split_k(cfg),
    }


def tool_record_iteration(iteration: int, description: str, verdict: str,
                          baseline_ms: float, candidate_ms: float = None,
                          speedup: float = None, correctness: bool = None,
                          notes: str = "", run_dir: str = "",
                          category: str = "", target_config: str = "via MCP") -> dict:
    """记录到优化日志。"""
    import time
    from agent_system.optimization_log import OptimizationLog, OptimizationEntry
    if run_dir:
        log_dir = Path(run_dir) / "logs"
    else:
        log_dir = ROOT / "results" / "manual_log"
    log = OptimizationLog(log_dir=log_dir)
    entry = OptimizationEntry(
        iteration=iteration,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        change_description=description,
        target_config=target_config,
        baseline_time_ms=baseline_ms,
        candidate_time_ms=candidate_ms,
        speedup=speedup,
        bandwidth_util_before=0, bandwidth_util_after=None,
        correctness_passed=correctness,
        verdict=verdict, notes=notes,
        category=category,
    )
    log.record(entry)
    return {"success": True, "total_entries": len(log.entries), "log_dir": str(log.log_dir)}


def tool_query_memory(keyword: str = "") -> dict:
    """查询领域记忆库（失败案例+硬件信念）。"""
    from agent_system.domain_memory import DomainMemory
    mem = DomainMemory(base_dir=ROOT / "agent_system" / "domain_memory")
    return {
        "failures_count": len(mem.failures),
        "beliefs_count": len(mem.belief.entries),
        "context": mem.build_context(keyword)[:1000] if keyword else "",
    }


def tool_list_operators() -> dict:
    """列出泛化 Agent 框架已注册的算子规格。"""
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
    """用泛化 Analyst 分析指定算子。"""
    from agent_system.operator_registry import get_operator
    from agent_system.generic_orchestrator import analyze_operator
    spec = get_operator(operator_id)
    report = analyze_operator(spec)
    return {
        "success": True,
        "operator_id": operator_id,
        "summary": report.summary,
        "artifacts": report.artifacts,
    }


def tool_get_operator_spec(operator_id: str = "flashattention_kvcache_decode",
                           full: bool = False) -> dict:
    """读取算子规格。full=false 时返回 Agent 常用摘要，避免上下文过大。"""
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
    """校验 Coder 输出的 OptimizationStrategy 是否符合 OperatorSpec。"""
    from agent_system.operator_registry import get_operator
    from agent_system.strategy_schema import OptimizationStrategy, validate_strategy

    try:
        parsed = OptimizationStrategy(**strategy)
    except Exception as e:
        return {"success": False, "valid": False, "error": f"schema error: {e}"}

    target = operator_id or parsed.target_operator
    try:
        spec = get_operator(target)
    except Exception as e:
        return {"success": False, "valid": False, "error": str(e)}

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
    """运行非 GPU 泛化多 Agent smoke，证明主循环可切换 operator。"""
    import tempfile
    from pathlib import Path
    from agent_system.domain_memory import DomainMemory
    from agent_system.generic_orchestrator import run_generic_iteration
    from agent_system.kernel_version_store import KernelVersionStore
    from agent_system.optimization_log import OptimizationLog
    from agent_system.strategy_schema import OptimizationStrategy

    def generate(spec, analysis, memory):
        return [OptimizationStrategy(
            strategy_id="mcp_smoke_strategy",
            target_operator=spec.operator_id,
            backend=spec.backend.kind,
            strategy_name=spec.optimization_space.strategy_names[0],
            params=spec.optimization_space.defaults(),
            expected_effect="MCP smoke for generalized multi-agent loop",
            risk="low",
            required_skills=list(spec.skills),
            description="MCP spec-only smoke strategy",
        )]

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
    """查看持久化 kernel 版本库。"""
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


def tool_run_closed_loop(operator_id: str = "flashattention_kvcache_decode",
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
                         num_heads_k: int = 8) -> dict:
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
    )
    data = result.to_dict()
    data["success"] = result.status == "completed"
    return data


def tool_list_runs(limit: int = 10) -> dict:
    """List recent closed-loop run directories."""
    from agent_system.paths import list_run_dirs
    runs = list_run_dirs(limit=limit)
    return {
        "success": True,
        "runs": [{"run_id": p.name, "run_dir": str(p)} for p in runs],
    }


def tool_latest_run() -> dict:
    """Return latest run and key artifact locations."""
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


# ── MCP 协议处理（stdin/stdout JSON-RPC）──
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
    "run_closed_loop": tool_run_closed_loop,
    "list_runs": tool_list_runs,
    "latest_run": tool_latest_run,
}

TOOL_SCHEMAS = [
    {"name": "compile_kernel", "description": "编译 .cu 为 .so",
     "inputSchema": {"type": "object", "properties": {
         "source_path": {"type": "string"}, "output_so": {"type": "string"}},
         "required": ["source_path"]}},
    {"name": "run_correctness", "description": "运行+allclose校验",
     "inputSchema": {"type": "object", "properties": {
         "so_path": {"type": "string"}, "batch": {"type": "integer"},
         "seq_kv": {"type": "integer"}, "headdim": {"type": "integer"}},
         "required": ["so_path"]}},
    {"name": "run_benchmark", "description": "GPU计时+带宽分析",
     "inputSchema": {"type": "object", "properties": {
         "so_path": {"type": "string"}, "batch": {"type": "integer"},
         "seq_kv": {"type": "integer"}}, "required": ["so_path"]}},
    {"name": "roofline_analyze", "description": "Roofline理论分析(免GPU)",
     "inputSchema": {"type": "object", "properties": {
         "batch": {"type": "integer"}, "seq_kv": {"type": "integer"}}}},
    {"name": "record_iteration", "description": "记录优化日志",
     "inputSchema": {"type": "object", "properties": {
         "iteration": {"type": "integer"}, "description": {"type": "string"},
         "verdict": {"type": "string"}, "baseline_ms": {"type": "number"}}, "required": ["iteration", "description", "verdict", "baseline_ms"]}},
    {"name": "query_memory", "description": "查询领域记忆库",
     "inputSchema": {"type": "object", "properties": {"keyword": {"type": "string"}}}},
    {"name": "list_operators", "description": "列出已注册的泛化算子规格",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "analyze_operator", "description": "用泛化 Analyst 分析算子",
     "inputSchema": {"type": "object", "properties": {
         "operator_id": {"type": "string"}}}},
    {"name": "get_operator_spec", "description": "读取 OperatorSpec 算子规格",
     "inputSchema": {"type": "object", "properties": {
         "operator_id": {"type": "string"}, "full": {"type": "boolean"}}}},
    {"name": "validate_strategy", "description": "校验 Coder 输出的 OptimizationStrategy",
     "inputSchema": {"type": "object", "properties": {
         "operator_id": {"type": "string"},
         "strategy": {"type": "object"}}, "required": ["strategy"]}},
    {"name": "run_agent_smoke", "description": "运行非GPU泛化多Agent smoke",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "list_kernel_versions", "description": "列出 KEEP 后持久化的 kernel 版本",
     "inputSchema": {"type": "object", "properties": {
         "operator_id": {"type": "string"}}}},
    {"name": "run_closed_loop", "description": "运行确定性闭环：提案->补丁->编译/测试/benchmark->A/B裁决->日志",
     "inputSchema": {"type": "object", "properties": {
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
         "num_heads_k": {"type": "integer"}}}},
    {"name": "list_runs", "description": "列出最近的闭环运行目录",
     "inputSchema": {"type": "object", "properties": {
         "limit": {"type": "integer"}}}},
    {"name": "latest_run", "description": "查看最新闭环运行和日志位置",
     "inputSchema": {"type": "object", "properties": {}}},
]


def handle_request(req: dict) -> dict:
    method = req.get("method", "")
    req_id = req.get("id")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": req_id, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "flashattn-agent-tools", "version": "1.0"}}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOL_SCHEMAS}}
    if method == "tools/call":
        params = req.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        if name in TOOLS:
            try:
                result = TOOLS[name](**args)
                return {"jsonrpc": "2.0", "id": req_id, "result": {
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}}
            except Exception as e:
                return {"jsonrpc": "2.0", "id": req_id, "result": {
                    "content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
                    "isError": True}}
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"未知方法: {method}"}}


if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except json.JSONDecodeError:
            continue
