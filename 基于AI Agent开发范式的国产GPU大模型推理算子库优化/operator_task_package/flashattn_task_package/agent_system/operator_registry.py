"""
Operator registry for the generalized Agent optimization framework.

The registry is intentionally small and explicit: adding a new operator should
mean adding a new OperatorSpec and evaluator plugin, not rewriting the
orchestrator.
"""
from __future__ import annotations

from typing import Iterable

from agent_system.specs import (
    AccuracySpec,
    BackendSpec,
    EvaluationSpec,
    OperatorSpec,
    OptimizationParam,
    OptimizationSpace,
    TestCaseSpec,
    TensorSpec,
    make_case_grid,
)


_REGISTRY: dict[str, OperatorSpec] = {}


def register_operator(spec: OperatorSpec, *, replace: bool = False) -> OperatorSpec:
    if spec.operator_id in _REGISTRY and not replace:
        raise ValueError(f"operator already registered: {spec.operator_id}")
    _REGISTRY[spec.operator_id] = spec
    return spec


def get_operator(operator_id: str) -> OperatorSpec:
    ensure_default_operators()
    try:
        return _REGISTRY[operator_id]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"unknown operator {operator_id!r}; known: {known}") from exc


def list_operators() -> list[OperatorSpec]:
    ensure_default_operators()
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def operator_ids() -> list[str]:
    return [spec.operator_id for spec in list_operators()]


def ensure_default_operators() -> None:
    if _REGISTRY:
        return
    register_operator(_flashattention_kvcache_decode_spec())
    register_operator(_fused_moe_i8_tn_spec())
    register_operator(_dummy_gemm_spec())


def _flashattention_kvcache_decode_spec() -> OperatorSpec:
    cases = make_case_grid(
        "fa_decode",
        {
            "batch_size": (1, 4, 16),
            "seqlen_kv": (1024, 4096, 8192, 16384),
        },
        tags=("oj_matrix", "decode", "paged_kv"),
    )
    backend = BackendSpec(
        kind="maca_cpp",
        language="CUDA Maca C++",
        compile_command="mxcc -std=c++17 -fPIC -shared -DMACA_ARCH=1000 -I$MACA_PATH/include",
        runtime="ctypes .so + PyTorch CUDA tensors",
        include_paths=("$MACA_PATH/include",),
        constraints=(
            "extern C run_kernel signature must match XPU-OJ",
            "do not synchronize inside run_kernel",
            "core QK/PV computation should use mctlass or mctlass primitives",
        ),
    )
    eval_spec = EvaluationSpec(
        accuracy=AccuracySpec(method="torch.allclose", rtol=1e-2, atol=1e-2),
        primary_metric="time_ms",
        higher_is_better=False,
        warmup=5,
        repeats=30,
    )
    opt_space = OptimizationSpace(
        params=(
            OptimizationParam("split_k", (1, 2, 4, 8, 12, 16), 1,
                              "Split KV sequence to increase decode parallelism", "medium"),
            OptimizationParam("tile_n", (16, 32, 64, 128), 64,
                              "KV tokens processed per tile", "medium"),
            OptimizationParam("num_warps", (1, 2, 4, 8), 4,
                              "Warps per thread block", "medium"),
            OptimizationParam("use_mctlass", (True, False), True,
                              "Use mctlass/Tensor Core path for QK/PV", "high"),
        ),
        strategy_names=(
            "mctlass_qk_tile",
            "online_softmax_fusion",
            "split_kv_decode",
            "paged_block_coalescing",
            "workspace_reuse",
        ),
    )
    return OperatorSpec(
        operator_id="flashattention_kvcache_decode",
        display_name="FlashAttention KV Cache Decode",
        category="attention.decode.paged_kv",
        interface=(
            "extern \"C\" void run_kernel(q, k_cache_paged, v_cache_paged, output, "
            "cache_seqlens, block_table, batch_size, seqlen_k, seqlen_q, "
            "num_heads, num_heads_k, headdim, page_block_size, num_blocks, causal)"
        ),
        inputs=(
            TensorSpec("q", "(B, 1, H, D)", "bf16", description="decode query"),
            TensorSpec("k_cache_paged", "(num_blocks, 16, HK, D)", "bf16"),
            TensorSpec("v_cache_paged", "(num_blocks, 16, HK, D)", "bf16"),
            TensorSpec("cache_seqlens", "(B,)", "int32"),
            TensorSpec("block_table", "(B, num_blocks/B)", "int32"),
        ),
        outputs=(TensorSpec("output", "(B, 1, H, D)", "bf16", role="output"),),
        test_cases=cases,
        backend=backend,
        evaluation=eval_spec,
        optimization_space=opt_space,
        skills=(
            "roofline-spec",
            "paged-addressing",
            "online-softmax",
            "split-k-pattern",
            "mctlass-usage",
            "run-benchmark",
        ),
        expected_bound="memory-bound",
        metadata={
            "headdim": 128,
            "num_heads": 8,
            "num_heads_k": 8,
            "seqlen_q": 1,
            "page_block_size": 16,
            "causal": 0,
            "dtype_bytes": 2,
        },
    )


def _fused_moe_i8_tn_spec() -> OperatorSpec:
    cases = (
        TestCaseSpec(
            "moe_i8_tn_public_001",
            {"em": 4096, "n": 4096, "k": 7168, "topk": "dynamic"},
            tags=("oj_matrix", "w8a8", "small_em"),
        ),
        TestCaseSpec(
            "moe_i8_tn_public_002",
            {"em": 32768, "n": 4096, "k": 7168, "topk": "dynamic"},
            tags=("oj_matrix", "w8a8", "large_em"),
        ),
        TestCaseSpec(
            "moe_i8_tn_public_003",
            {"em": 4096, "n": 7168, "k": 2048, "topk": "dynamic"},
            tags=("oj_matrix", "w8a8", "wide_n"),
        ),
        TestCaseSpec(
            "moe_i8_tn_public_004",
            {"em": 32768, "n": 7168, "k": 2048, "topk": "dynamic"},
            tags=("oj_matrix", "w8a8", "large_em", "wide_n"),
        ),
        TestCaseSpec(
            "moe_i8_tn_benchmark_topk1",
            {"rows": 256, "n": 128, "k": 128, "topk": 1},
            tags=("local_benchmark", "smoke"),
        ),
        TestCaseSpec(
            "moe_i8_tn_benchmark_topk2",
            {"rows": 256, "n": 128, "k": 128, "topk": 2},
            tags=("local_benchmark", "smoke"),
        ),
        TestCaseSpec(
            "moe_i8_tn_benchmark_topk3",
            {"rows": 128, "n": 128, "k": 128, "topk": 3},
            tags=("local_benchmark", "smoke"),
        ),
    )
    backend = BackendSpec(
        kind="maca_cpp",
        language="CUDA Maca C++",
        compile_command=(
            "bash operator_task_package/fused_moe_task_package/benchmark/scripts/"
            "build_fused_moe_i8_tn_pybind.sh"
        ),
        runtime="pybind benchmark + XPU-OJ run_kernel submission",
        include_paths=(
            "$MACA_PATH/include",
            "../fused_moe_task_package/benchmark/standalone/fused_moe_i8_tn/src",
        ),
        constraints=(
            "extern C run_kernel signature must match XPU-OJ Fused MoE",
            "W8A8 int8 inputs with fp32 scale_a/scale_b and bf16 output",
            "do not rely on benchmark-only pybind symbols in OJ submission",
        ),
    )
    eval_spec = EvaluationSpec(
        accuracy=AccuracySpec(method="torch/reference allclose", rtol=1e-2, atol=1e-2),
        primary_metric="time_ms",
        higher_is_better=False,
        warmup=5,
        repeats=20,
    )
    opt_space = OptimizationSpace(
        params=(
            OptimizationParam("block_m", (16, 32, 64, 128), 32,
                              "Rows of routed tokens per CTA", "medium"),
            OptimizationParam("block_n", (32, 64, 128), 64,
                              "Output columns per CTA", "medium"),
            OptimizationParam("block_k", (64, 128, 256), 128,
                              "K tile for int8 GEMM mainloop", "medium"),
            OptimizationParam("threads", (128, 256), 256,
                              "Threads per CTA", "medium"),
            OptimizationParam("use_mctlass", (True, False), True,
                              "Use mctlass/int8 GEMM primitives when available", "high"),
            OptimizationParam("fuse_epilogue_scale", (True, False), True,
                              "Fuse scale_a/scale_b/moe_weights into epilogue", "low"),
        ),
        strategy_names=(
            "grouped_gemm_by_expert",
            "int8_mma_tile",
            "expert_routing_reorder",
            "scale_epilogue_fusion",
            "persistent_expert_blocks",
            "pybind_benchmark_adapter",
        ),
    )
    return OperatorSpec(
        operator_id="fused_moe_i8_tn",
        display_name="Fused MoE W8A8 TN",
        category="moe.grouped_gemm.w8a8",
        interface=(
            "extern \"C\" void run_kernel(a, b_col_major, scale_a, scale_b, "
            "moe_weights, token_ids, expert_ids, topk, out)"
        ),
        inputs=(
            TensorSpec("a", "(EM, K)", "int8", description="routed token activations"),
            TensorSpec("b_col_major", "(E, N, K)", "int8", layout="expert-major TN"),
            TensorSpec("scale_a", "(EM,)", "float32"),
            TensorSpec("scale_b", "(E, N)", "float32"),
            TensorSpec("moe_weights", "(EM,)", "float32", description="routing weights per routed row"),
            TensorSpec("token_ids", "(EM,)", "int32"),
            TensorSpec("expert_ids", "(ceil(EM/tile),)", "int32"),
            TensorSpec("topk", "scalar", "int64"),
        ),
        outputs=(TensorSpec("out", "(EM, N)", "bf16", role="output"),),
        test_cases=cases,
        backend=backend,
        evaluation=eval_spec,
        optimization_space=opt_space,
        skills=(
            "mctlass-usage",
            "roofline-spec",
            "run-benchmark",
        ),
        expected_bound="compute-bound",
        metadata={
            "task_package": "../fused_moe_task_package",
            "benchmark_dir": "../fused_moe_task_package/benchmark",
            "baseline_kernel": "../fused_moe_task_package/kernel/baseline_kernel.cu",
            "starter_cuda_maca": "../fused_moe_task_package/kernel/baseline_kernel.cu",
            "starter_cuda_maca_original": "../fused_moe_task_package/starter/示例冒烟代码-CUDA Maca.txt",
            "benchmark_script": "scripts/run_fused_moe_i8_tn_benchmark.sh",
            "correctness_script": "scripts/run_fused_moe_i8_tn_pybind_test.sh",
            "dtype": "W8A8",
            "requires_operator_specific_evaluator": True,
        },
    )


def _dummy_gemm_spec() -> OperatorSpec:
    """A tiny second operator used to prove the framework is not single-task."""
    return OperatorSpec(
        operator_id="dummy_bf16_gemm",
        display_name="Dummy BF16 GEMM",
        category="gemm",
        interface="C = A @ B",
        inputs=(
            TensorSpec("A", "(M, K)", "bf16"),
            TensorSpec("B", "(K, N)", "bf16"),
        ),
        outputs=(TensorSpec("C", "(M, N)", "bf16", role="output"),),
        test_cases=make_case_grid("gemm", {"M": (128,), "N": (128,), "K": (128,)},
                                  tags=("generalization_smoke",)),
        backend=BackendSpec(kind="mock", language="spec-only"),
        evaluation=EvaluationSpec(accuracy=AccuracySpec()),
        optimization_space=OptimizationSpace(
            params=(
                OptimizationParam("tile_m", (64, 128), 128, "M tile"),
                OptimizationParam("tile_n", (64, 128), 128, "N tile"),
                OptimizationParam("tile_k", (32, 64), 64, "K tile"),
            ),
            strategy_names=("mctlass_gemm_tile", "pipeline_stages"),
        ),
        skills=("mctlass-usage", "roofline-spec"),
        expected_bound="compute-bound",
        metadata={"purpose": "framework generalization smoke test"},
    )
