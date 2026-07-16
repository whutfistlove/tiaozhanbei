# AGENTS.md - Operator Optimization Agent Contract

本项目的目标是构建可泛化的算子优化 Agent 系统。Agent 不是只做微调，也不是无约束生成整份 kernel，而是按阶段执行：

```text
Explore 大结构候选 -> Stabilize 局部重构 -> Tune 小参数微调
```

每个阶段都必须进入确定性闭环：

```text
proposal -> patch/candidate -> compile -> correctness -> benchmark -> A/B decision -> KEEP/rollback -> logs/memory
```

## Coder 分级职责

### 1. Macro Coder / Architect

用途：前期探索，允许较大结构性变化。

适合：
- 新增 split-k/reduce 结构
- 切换 kernel 组织方式
- 引入模板化 mctlass 路径
- 改变 paged addressing 或 online-softmax 主流程

输出要求：
- `scale="macro"`
- `phase="explore"`
- `change_type` 必须是 `template_swap`、`structural_rewrite`、`memory_layout` 或 `loop_transform`
- 可以提供完整 `patched_source`，但必须作为候选版本隔离验证
- 必须写 `rollback_plan`
- 不允许直接覆盖 current best

### 2. Meso Coder / Stabilizer

用途：中期稳定结构候选。

适合：
- 修复 macro candidate 的边界、同步、workspace、launch 参数
- 调整一段循环或 reduce 逻辑
- 局部访存布局修正

输出要求：
- `scale="meso"`
- `phase="stabilize"`
- 默认净改动上限 40 行
- 必须有明确 target 和 rollback plan

### 3. Micro Coder / Tuner

用途：后期精调。

适合：
- `NUM_SPLITS`
- block size
- unroll 因子
- launch/grid 参数
- 小范围条件判断

输出要求：
- `scale="micro"`
- `phase="tune"`
- 默认净改动上限 6 行
- `before` 必须在当前源码中唯一匹配

## 核心原则

1. 每轮只验证一个候选，保证因果归因清晰。
2. 大改可以做，但只能在 Explore/Macro 阶段，以候选版本进入 A/B，不直接污染 best。
3. 后期才使用 Micro `ChangeProposal` 做小步微调。
4. Profiler/Judge 必须调用 `scripts/run_closed_loop.py` 或 MCP `run_closed_loop`，不靠聊天结论 KEEP。
5. KEEP 必须来自真实 A/B：正确性通过，candidate 中位数耗时小于 baseline * `(1 - noise_margin)`。
6. 所有产物进入 `runs/<run_id>/`，全局索引进入 `results/`。
7. Logger 不是聊天角色：工具事件写入 `results/opencode_events.jsonl`，优化结论写入 `runs/<run_id>/logs/`。

## 固定目录

```text
runs/<run_id>/
  run_manifest.json
  events.jsonl
  versions/
  logs/
    optimization_log.md
    optimization_log.jsonl
    kept_changes.md
    rejected_changes.md
    errors.md
    summary.md
  memory/
  rounds/
    round_001/
      baseline_a.cu
      analysis.json
      proposal.json
      cand_<proposal>.cu
      decision.json

results/
  latest_run.txt
  summary.md
  opencode_events.jsonl
```

## Agent 分工

Analyst:
- 读取 OperatorSpec、roofline、latest_run、memory。
- 判断当前阶段应该 explore、stabilize 还是 tune。
- 给出 1 到 3 个候选方向，但不写代码。

Coder:
- 按阶段输出一个 `ChangeProposal`。
- 前期可以输出 macro candidate，后期输出 micro patch。
- 不跑 benchmark，不自评 KEEP。

Profiler:
- 执行 `run_closed_loop`。
- dry-run 验证 patch/候选落盘。
- real-run 在目标 GPU 上执行 compile/correctness/benchmark。

Judge:
- 只读 `decision.json`、`optimization_log.jsonl`、`manifest_v2.json`。
- 裁决 KEEP/REJECT/ERROR/NOCHANGE/SKIP。

Reflector:
- ERROR/REJECT 写失败模式。
- KEEP 写可迁移硬件/算子规律。

## 当前算子重点

- operator_id: `flashattention_kvcache_decode`
- 默认 kernel: `kernel/splitk_h128.cu`
- dtype: BF16
- 当前本地默认矩阵: batch `[1,4,16]`, seq_kv `[1024,4096,8192,16384]`, page size 16
- 关键方向: split-k, paged addressing, online softmax, memory coalescing, mctlass template path

## 常用命令

```bash
python scripts/run_closed_loop.py --phase explore --rounds 1
python scripts/run_closed_loop.py --phase stabilize --rounds 1
python scripts/run_closed_loop.py --phase tune --rounds 1
python scripts/run_closed_loop.py --real --phase tune --rounds 1 --batch 1 --seq-kv 4096 --headdim 128
```
