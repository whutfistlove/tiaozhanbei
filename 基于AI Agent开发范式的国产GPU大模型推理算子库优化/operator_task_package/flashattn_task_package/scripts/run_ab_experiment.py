#!/usr/bin/env python
"""
严格的滚动 A/B 优化实验 demo。

展示改造后的 agent 流程：
1. 每轮只做一处单点改动（ChangeProposal，由 validate_single_change 硬约束）
2. A = 当前已 KEEP 的最优版本（滚动基线），B = 本轮改动后的代码
3. KEEP 判定带噪声容限（默认 3%）：B 中位数必须 < A 中位数 × (1 - margin)
4. 只有真实提速才保留代码（版本链推进），否则归档为 NOCHANGE/REJECT/ERROR
5. 日志分类归档：kept_changes.md / rejected_changes.md / errors.md / summary.md

每轮的 proposal 由一个本地生成器轮换产生（模拟 Coder 角色），涵盖：
  - NUM_SPLITS 调参（真实提速，应 KEEP）
  - 无效改动（在噪声内，应 NOCHANGE）
  - 编译失败（应 ERROR）

运行：python scripts/run_ab_experiment.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from agent_system.roofline_engine import KernelConfig
from agent_system.domain_memory import DomainMemory
from agent_system.optimization_log import OptimizationLog
from agent_system.kernel_version_store import KernelVersionStore
from agent_system.real_orchestrator import run_ab_loop
from agent_system.strategy_schema import ChangeProposal
from agent_system.paths import new_run_dir

# 统一输出到 runs/run_<timestamp>_ab/，不再散落到 .agent_work 或项目根目录
WORK = new_run_dir("ab")

# A：splitk_h128 强制 NUM_SPLITS=1（关闭 split，慢且正确）—— 作为初始版本
SPLITK_SRC = (ROOT / "kernel" / "splitk_h128.cu").read_text()
INITIAL_CODE = SPLITK_SRC.replace("static constexpr int NUM_SPLITS = 12;",
                                  "static constexpr int NUM_SPLITS = 1;")
# KEEP 候选：恢复自适应 splits
KEEP_CODE = SPLITK_SRC
# NOCHANGE 候选：与 A 相同（注释微调，无性能影响）
NOCHANGE_CODE = SPLITK_SRC.replace("static constexpr int NUM_SPLITS = 12;",
                                   "static constexpr int NUM_SPLITS = 12; // tuned")
# ERROR 候选：引入未知类型，编译失败
ERROR_CODE = KEEP_CODE.replace("typedef mctlass::bfloat16_t __nv_bfloat16;",
                               "typedef BROKEN_TYPE __nv_bfloat16;")


def make_proposal(round_idx: int) -> ChangeProposal:
    """轮换产生不同性质的 proposal，演示四种结局。"""
    proposals = [
        # 第1轮：真实提速（NUM_SPLITS 1→12 自适应）→ KEEP
        ChangeProposal(
            proposal_id="r1_splitk", target="NUM_SPLITS", change_type="param_tune",
            one_line_summary="NUM_SPLITS 1→12 自适应填满 SM",
            before="static constexpr int NUM_SPLITS = 1;",
            after="static constexpr int NUM_SPLITS = 12;",
            hypothesis="b=1 时 split 并行 decode 填满 96 个 SM",
            risk="low", patched_source=KEEP_CODE,
        ),
        # 第2轮：无效改动（仅注释）→ NOCHANGE
        ChangeProposal(
            proposal_id="r2_noop", target="comment", change_type="other",
            one_line_summary="添加注释（无性能影响）",
            before="// tuned", after="// tuned v2",
            hypothesis="测试噪声容限：不应 KEEP",
            risk="low", patched_source=NOCHANGE_CODE,
        ),
        # 第3轮：编译失败 → ERROR
        ChangeProposal(
            proposal_id="r3_broken", target="type_swap", change_type="api_swap",
            one_line_summary="误用未知类型",
            before="typedef mctlass::bfloat16_t __nv_bfloat16;",
            after="typedef BROKEN_TYPE __nv_bfloat16;",
            hypothesis="演示编译失败归档",
            risk="high", patched_source=ERROR_CODE,
        ),
    ]
    return proposals[round_idx % len(proposals)]


def main() -> int:
    if not torch.cuda.is_available():
        print("此 demo 需 GPU（编译运行真实 kernel）")
        return 1

    cfg = KernelConfig(batch_size=1, seqlen_kv=4096, headdim=128,
                       num_heads=8, num_heads_k=8)
    memory = DomainMemory(base_dir=WORK / "memory")
    log = OptimizationLog(log_dir=WORK / "logs")
    store = KernelVersionStore(WORK / "versions")
    # 初始版本 v0 = 退化 baseline（NUM_SPLITS=1）
    v0 = store.add_source("flashattention_kvcache_decode", INITIAL_CODE,
                          "initial: splitk with NUM_SPLITS forced to 1",
                          verdict="KEEP", metrics={})
    store.promote(v0.version_id)

    print("=" * 64)
    print("  严格的滚动 A/B 优化实验")
    print("  规则: 每轮单点改动 | 滚动当前最优为基线 | 噪声容限 3%")
    print("  日志: 分类归档 kept/rejected/errors + summary")
    print("=" * 64)
    print(f"输出目录: {WORK}")
    print(f"  ├── versions/   保留下来的算子源码 (.cu)")
    print(f"  ├── logs/       可审视日志 (summary/kept/rejected/errors)")
    print(f"  ├── memory/     领域记忆库")
    print(f"  └── rounds/     每轮 A/B 可复现物料")
    print(f"初始版本 A = {v0.version_id} (NUM_SPLITS=1, 退化)\n")

    # proposal 生成器（按轮次轮换）
    proposal_state = {"i": 0}

    def gen(c, bottleneck, mem):
        p = make_proposal(proposal_state["i"])
        proposal_state["i"] += 1
        return p

    results = run_ab_loop(
        cfg=cfg, initial_code=INITIAL_CODE, initial_version_id=v0.version_id,
        generate_proposal_fn=gen, memory=memory, log=log, version_store=store,
        workdir=WORK / "rounds", max_iterations=3,
        noise_margin=0.03, warmup=3, repeats=20,
    )

    # 打印最终汇总
    print("\n" + "=" * 64)
    print("  实验结束 — 分类日志归档")
    print("=" * 64)
    log_dir = WORK / "logs"
    for f in ["optimization_log.md", "kept_changes.md", "rejected_changes.md",
              "errors.md", "summary.md"]:
        p = log_dir / f
        if p.exists():
            n = sum(1 for line in p.read_text().splitlines() if line.startswith("### 轮次"))
            print(f"  {f:<26} ({n} 条详细记录)")

    print("\n--- summary.md ---")
    print((log_dir / "summary.md").read_text())

    print("--- 版本链 ---")
    for v in store.versions:
        print(f"  {v.version_id} [{v.verdict}] {v.description[:50]}")
        if v.parent:
            print(f"    parent: {v.parent}")
    best = store.current("flashattention_kvcache_decode")
    if best:
        print(f"\n当前最优版本: {best.version_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
