---
name: memory-reflection
description: Use when updating failure cases, hardware beliefs, or skill suggestions after an optimization iteration. Trigger keywords: Reflector, failure_cases, hardware_belief, memory, reflection, 复盘, 记忆.
---

# Memory Reflection

Reflector 负责把一次迭代转化为可复用知识。记忆必须有证据、适用边界和置信度。

## FailureCase 记录模板

```json
{
  "category": "api_misuse|type_error|runtime_error|correctness_error|performance_regression",
  "symptom": "short observable symptom",
  "root_cause": "why it happened",
  "fix": "actionable fix",
  "code_snippet": "optional",
  "config": "operator_id=..., strategy=..., params=..."
}
```

## HardwareBelief 记录模板

```json
{
  "observation": "measured fact with config",
  "rule": "generalized rule with boundary",
  "confidence": 0.3
}
```

## 置信度规则

- 0.3：单次失败或弱观察。
- 0.5：单配置正确且性能趋势合理。
- 0.7：多配置支持。
- 0.85+：跨多轮/多算子支持。

## 好的 belief

```text
观察: flashattention_kvcache_decode 中 B=1, seq>=4096 时 split_k=8 比 split_k=1 快 1.4x。
规律: decode attention 小 batch 长序列优先尝试 split_kv_decode，但 split 数过大会增加 reduce HBM 流量。
置信度: 0.7
```

## 坏的 belief

```text
split_k=8 总是最快。
```

问题：没有 operator、配置边界、证据或置信度。
