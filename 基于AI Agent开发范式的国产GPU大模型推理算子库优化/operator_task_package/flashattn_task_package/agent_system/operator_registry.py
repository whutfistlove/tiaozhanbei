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
