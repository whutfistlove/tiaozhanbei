---
name: strategy-generation
description: Use when the Coder agent proposes optimization candidates. It enforces structured OptimizationStrategy JSON instead of unconstrained full-code generation. Trigger keywords: strategy, OptimizationStrategy, candidate, structured output, 策略生成, 候选生成.
---

# Strategy Generation

普通闭环优化优先使用 `ChangeProposal`，但它不是只能做小补丁。当前分三层：

- `scale=macro, phase=explore`: 前期结构性候选，可用 `patched_source` 或模板输出。
- `scale=meso, phase=stabilize`: 中期局部重构/修复，默认 40 行以内。
- `scale=micro, phase=tune`: 后期参数微调，默认 6 行以内。

只有在新增算子、跨后端泛化或模板渲染阶段，才使用 `OptimizationStrategy`。

## Closed-loop ChangeProposal

```json
{
  "proposal_id": "r1_num_splits_12_to_8",
  "target": "NUM_SPLITS",
  "change_type": "param_tune",
  "one_line_summary": "Tune Split-K fanout",
  "before": "static constexpr int NUM_SPLITS = 12;",
  "after": "static constexpr int NUM_SPLITS = 8;",
  "hypothesis": "Reduce reduction overhead for this shape",
  "risk": "low"
}
```

约束：
- `before` 必须唯一匹配当前源码。
- `after` 只放替换片段，不放整份 `.cu`。
- 每轮只输出一个 proposal。
- 交给 `run_closed_loop` 做 patch/compile/correctness/benchmark/A-B。

Coder 角色优先生成 `OptimizationStrategy`，而不是直接生成整份 `.cu`。这样 Profiler 可以统一过滤、渲染、编译和评测。

## JSON Schema

```json
{
  "strategies": [
    {
      "strategy_id": "unique_id",
      "target_operator": "flashattention_kvcache_decode",
      "backend": "maca_cpp",
      "strategy_name": "split_kv_decode",
      "params": {"split_k": 8, "tile_n": 64},
      "expected_effect": "increase parallel blocks for small-batch decode",
      "risk": "medium",
      "required_skills": ["split-k-pattern", "online-softmax"],
      "description": "One concise sentence"
    }
  ]
}
```

字段必须匹配 `agent_system.strategy_schema.OptimizationStrategy`。

## 生成流程

1. 调用 `get_operator_spec`。
2. 从 `optimization_space.strategy_names` 中选择 strategy。
3. 参数只能来自 `optimization_space.params`，除非明确标注为 template-only。
4. 每个 strategy 只包含一个主优化点。
5. 调用 `validate_strategy`。
6. 需要代码时，附带最小 source_code 或 code_path。

## 风险分级

- `low`：只调整模板参数或 launch 配置。
- `medium`：改变并行策略、tile、split、workspace。
- `high`：更换核心计算路径、重写 softmax/reduce、改变数据布局。

## 好策略示例

```json
{
  "strategy_id": "fa_splitk_b1_seq4096_001",
  "target_operator": "flashattention_kvcache_decode",
  "backend": "maca_cpp",
  "strategy_name": "split_kv_decode",
  "params": {"split_k": 8, "tile_n": 64, "num_warps": 4},
  "expected_effect": "fill more C500 SMs for batch=1 decode",
  "risk": "medium",
  "required_skills": ["split-k-pattern", "online-softmax"],
  "description": "Split KV into 8 segments and merge partial online-softmax states"
}
```

## 坏策略示例

```json
{"strategy_name": "make_it_faster", "params": {"magic": 100}}
```

问题：

- strategy_name 未注册。
- 参数不在优化空间。
- 没有 target_operator/backend。
- 没有可验证预期效果。
