"""
Kernel Loader —— 用 ctypes 加载编译好的 .so，调用 run_kernel。

把 PyTorch GPU 张量的指针传给 C 函数，运行后取回输出。
这是 Profiler/Judge 调用真实 kernel 的桥梁。

关键：PyTorch 的 .data_ptr() 给出 GPU 显存指针，ctypes 传给 .so。
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from agent_system.roofline_engine import KernelConfig


# run_kernel 的 C 签名参数类型（全部是指针 + int64）
RUN_KERNEL_ARGTYPES = [
    ctypes.c_void_p,  # q
    ctypes.c_void_p,  # k_cache_paged
    ctypes.c_void_p,  # v_cache_paged
    ctypes.c_void_p,  # output
    ctypes.c_void_p,  # cache_seqlens
    ctypes.c_void_p,  # block_table
    ctypes.c_int64,   # batch_size
    ctypes.c_int64,   # seqlen_k
    ctypes.c_int64,   # seqlen_q
    ctypes.c_int64,   # num_heads
    ctypes.c_int64,   # num_heads_k
    ctypes.c_int64,   # headdim
    ctypes.c_int64,   # page_block_size
    ctypes.c_int64,   # num_blocks
    ctypes.c_int64,   # causal
]


@dataclass
class LoadResult:
    """kernel 加载结果。"""
    success: bool
    run_kernel_fn: Optional[object] = None  # ctypes 函数对象
    error_msg: str = ""


def load_kernel(so_path: str) -> LoadResult:
    """加载 .so 并取出 run_kernel 函数。"""
    if not os.path.exists(so_path):
        return LoadResult(success=False, error_msg=f".so 不存在: {so_path}")
    try:
        lib = ctypes.CDLL(so_path)
        fn = lib.run_kernel
        fn.argtypes = RUN_KERNEL_ARGTYPES
        fn.restype = None
        return LoadResult(success=True, run_kernel_fn=fn)
    except Exception as e:
        return LoadResult(success=False, error_msg=f"加载失败: {e}")


def _ptr(t: torch.Tensor) -> int:
    """取张量的 GPU 数据指针（兼容 cuda/metax）。"""
    if not t.is_contiguous():
        t = t.contiguous()
    return t.data_ptr()


def call_run_kernel(
    run_kernel_fn,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    output: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    cfg: KernelConfig,
    num_blocks: int,
) -> torch.Tensor:
    """
    调用 run_kernel，返回 output 张量。

    参数布局对齐 OJ 接口。
    output 需预先分配（bf16），函数内填充。
    """
    if q.device.type != "cuda":
        raise RuntimeError(f"kernel 需 GPU 张量，得到 {q.device}")
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA/MACA GPU")

    # 确保类型正确
    assert q.dtype == torch.bfloat16, f"q 需 bf16，得到 {q.dtype}"
    assert cache_seqlens.dtype == torch.int32
    assert block_table.dtype == torch.int32

    # 确保 output 连续且已分配
    out = output if output.is_contiguous() else output.contiguous()

    run_kernel_fn(
        _ptr(q), _ptr(k_cache), _ptr(v_cache), _ptr(out),
        _ptr(cache_seqlens), _ptr(block_table),
        ctypes.c_int64(cfg.batch_size),
        ctypes.c_int64(cfg.seqlen_kv),    # seqlen_k
        ctypes.c_int64(cfg.seqlen_q),
        ctypes.c_int64(cfg.num_heads),
        ctypes.c_int64(cfg.num_heads_k),
        ctypes.c_int64(cfg.headdim),
        ctypes.c_int64(cfg.page_block_size),
        ctypes.c_int64(num_blocks),
        ctypes.c_int64(0),                # causal=0
    )
    torch.cuda.synchronize()
    return out


def make_output_tensor(cfg: KernelConfig, device: str = "cuda") -> torch.Tensor:
    """预分配输出张量。"""
    return torch.zeros(
        cfg.batch_size, cfg.seqlen_q, cfg.num_heads, cfg.headdim,
        device=device, dtype=torch.bfloat16,
    )
