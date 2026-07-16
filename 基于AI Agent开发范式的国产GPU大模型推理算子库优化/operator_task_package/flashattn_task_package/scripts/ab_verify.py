#!/usr/bin/env python
"""
A/B 验证 helper —— 给 OpenCode 主 agent 用 bash 调用。

用法：
  python scripts/ab_verify.py <baseline.cu> <candidate.cu> <output_dir>

对一个 baseline(A) 和 candidate(B) kernel 做严格 A/B 实验：
编译 → 正确性校验 → benchmark(中位数) → 噪声容限判定 → 分类归档

输出 <output_dir> 下完整的 logs/ + versions/，返回 verdict。
主 agent 只需调这一个脚本就能完成 Profiler+Judge+Logger 的工作。
"""
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from agent_system.roofline_engine import KernelConfig
from agent_system.domain_memory import DomainMemory
from agent_system.optimization_log import OptimizationLog
from agent_system.kernel_version_store import KernelVersionStore
from agent_system.real_orchestrator import run_ab_iteration
from agent_system.strategy_schema import ChangeProposal


def main():
    if len(sys.argv) < 4:
        print("用法: python scripts/ab_verify.py <baseline.cu> <candidate.cu> <output_dir> [target] [summary]")
        return 2
    baseline_cu, candidate_cu, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    target = sys.argv[4] if len(sys.argv) > 4 else "auto_optimization"
    summary = sys.argv[5] if len(sys.argv) > 5 else "OpenCode 主 agent 自动优化"

    if not torch.cuda.is_available():
        print("FAIL: 需要 GPU")
        return 2

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("versions", "logs", "memory", "rounds"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    a_code = Path(baseline_cu).read_text()
    b_code = Path(candidate_cu).read_text()
    cfg = KernelConfig(batch_size=1, seqlen_kv=4096, headdim=128, num_heads=8, num_heads_k=8)

    memory = DomainMemory(base_dir=out / "memory")
    log = OptimizationLog(log_dir=out / "logs")
    store = KernelVersionStore(out / "versions")
    v0 = store.add_source("flashattention_kvcache_decode", a_code, "baseline", "KEEP", {})
    store.promote(v0.version_id)

    proposal = ChangeProposal(
        proposal_id="oc_auto", target=target, change_type="loop_transform",
        one_line_summary=summary,
        before="(见 baseline.cu)", after="(见 candidate.cu)",
        hypothesis="主 agent 自动优化", risk="medium", patched_source=b_code,
    )

    result = run_ab_iteration(
        1, cfg, a_code, v0.version_id, lambda c, b, m: proposal,
        memory, log, store, out / "rounds", noise_margin=0.03, warmup=5, repeats=30,
    )

    print("=" * 56)
    print(f"  A/B 裁决: {result.verdict}")
    print(f"  A (baseline):  {result.baseline_time_ms:.4f} ms")
    if result.candidate_time_ms:
        print(f"  B (candidate): {result.candidate_time_ms:.4f} ms")
    if result.speedup:
        print(f"  speedup: {result.speedup:.3f}x")
    if result.reject_reason:
        print(f"  原因: {result.reject_reason}")
    if result.promoted_version:
        print(f"  ✓ 版本已保留: {result.promoted_version}")
    print(f"  日志: {out}/logs/")
    print("=" * 56)
    # 退出码：KEEP=0, 其他=1（方便主 agent 用 shell 判断）
    return 0 if result.verdict == "KEEP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
