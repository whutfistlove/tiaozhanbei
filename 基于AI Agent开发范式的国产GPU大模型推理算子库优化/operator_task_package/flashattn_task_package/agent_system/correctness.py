"""
正确性校验器（Correctness Checker）。

生成参考输出（用 PyTorch 原生 attention），与待测 kernel 输出做 allclose 校验。
精度要求：torch.allclose(rtol=1e-2, atol=1e-2)（OJ 题包固定）。

这是 Judge 角色的核心工具——独立验证（防 Coder 自评盲区 + 防 reward hacking）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from agent_system.roofline_engine import KernelConfig


def generate_reference(
    q: torch.Tensor,
    k_cache_paged: torch.Tensor,
    v_cache_paged: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    cfg: KernelConfig,
) -> torch.Tensor:
    """
    用 PyTorch 原生 attention 生成参考输出。

    参数布局（对齐 OJ 接口）：
      q: (B, seqlen_q, num_heads, headdim)
      k_cache_paged: (num_blocks, page_block_size, num_heads_k, headdim)
      v_cache_paged: 同上
      cache_seqlens: (B,) int32
      block_table: (B, num_blocks//B) int32

    返回：output (B, seqlen_q, num_heads, headdim) float32（参考真值）
    """
    B = cfg.batch_size
    H = cfg.num_heads
    HK = cfg.num_heads_k
    D = cfg.headdim
    PBS = cfg.page_block_size
    blocks_per_batch = block_table.shape[1]
    device = q.device

    output = torch.zeros(B, cfg.seqlen_q, H, D, device=device, dtype=torch.float32)

    for b in range(B):
        seq_len = int(cache_seqlens[b].item())
        # 把 paged cache 还原成连续的 K/V: (seq_len, HK, D)
        k_contig = torch.zeros(seq_len, HK, D, device=device, dtype=k_cache_paged.dtype)
        v_contig = torch.zeros(seq_len, HK, D, device=device, dtype=v_cache_paged.dtype)
        for t in range(seq_len):
            page_idx = t // PBS
            page_off = t % PBS
            phys = int(block_table[b, page_idx].item())
            k_contig[t] = k_cache_paged[phys, page_off]
            v_contig[t] = v_cache_paged[phys, page_off]

        # 对每个 query head 做 attention
        groups = max(1, H // HK)  # GQA: 每 group 个 q head 共享 1 个 kv head
        for h in range(H):
            kv_h = h // groups  # GQA head 映射
            qv = q[b, 0, h].float()                    # (D,)
            k_sel = k_contig[:, kv_h].float()          # (seq_len, D)
            v_sel = v_contig[:, kv_h].float()          # (seq_len, D)
            scores = qv @ k_sel.T / math.sqrt(D)       # (seq_len,)
            # causal=0，不 mask；但只取前 seq_len 个
            attn = torch.softmax(scores, dim=0)         # (seq_len,)
            out = attn @ v_sel                          # (D,)
            output[b, 0, h] = out

    return output


@dataclass
class CorrectnessResult:
    """正确性校验结果。"""
    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    mean_abs_diff: float
    num_elements: int
    rtol: float
    atol: float
    detail: str = ""


def check(
    output_test: torch.Tensor,
    output_ref: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> CorrectnessResult:
    """
    校验待测输出与参考输出的数值一致性（OJ 标准：allclose rtol=atol=1e-2）。

    全部转 float32 比较（对齐 OJ：output_t.float() vs output_ref.float()）。
    """
    t = output_test.float().reshape(-1)
    r = output_ref.float().reshape(-1)
    assert t.shape == r.shape, f"shape 不匹配: {t.shape} vs {r.shape}"

    abs_diff = (t - r).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    # 相对误差（避免除零）
    denom = r.abs().clamp(min=1e-6)
    rel_diff = abs_diff / denom
    max_rel = rel_diff.max().item()

    passed = torch.allclose(t, r, rtol=rtol, atol=atol)

    detail = (
        f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} "
        f"max_rel={max_rel:.4f} elements={t.numel()}"
    )

    return CorrectnessResult(
        passed=passed,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        mean_abs_diff=mean_abs,
        num_elements=t.numel(),
        rtol=rtol,
        atol=atol,
        detail=detail,
    )


def make_test_inputs(
    cfg: KernelConfig,
    num_blocks: Optional[int] = None,
    device: str = "cuda",
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成一组随机测试输入（对齐 OJ benchmark 的输入构造方式）。

    返回 (q, k_cache_paged, v_cache_paged, cache_seqlens, block_table)。
    block_table 用 randperm 模拟 paged 随机分布（与官方 benchmark 一致）。
    """
    torch.manual_seed(seed)
    if num_blocks is None:
        import math as _m
        num_blocks = max(1024, _m.ceil(cfg.seqlen_kv / cfg.page_block_size) * cfg.batch_size * 3)

    q = torch.randn(cfg.batch_size, cfg.seqlen_q, cfg.num_heads, cfg.headdim,
                    device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(num_blocks, cfg.page_block_size, cfg.num_heads_k, cfg.headdim,
                          device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(num_blocks, cfg.page_block_size, cfg.num_heads_k, cfg.headdim,
                          device=device, dtype=torch.bfloat16)

    blocks_per_batch = num_blocks // cfg.batch_size
    # 与官方 benchmark 一致：randperm 后 reshape
    perm = torch.randperm(num_blocks, dtype=torch.int32, device=device)
    block_table = perm[: cfg.batch_size * blocks_per_batch].reshape(
        cfg.batch_size, blocks_per_batch
    )

    cache_seqlens = torch.full((cfg.batch_size,), cfg.seqlen_kv,
                               dtype=torch.int32, device=device)

    return q, k_cache, v_cache, cache_seqlens, block_table
