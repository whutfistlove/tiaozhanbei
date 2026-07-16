---
name: judge-policy
description: Use when deciding KEEP, ROLLBACK, REJECT, or SPECIALIZE for a candidate strategy. Trigger keywords: Judge, verdict, KEEP, ROLLBACK, REJECT, SPECIALIZE, 裁决.
---

# Judge Policy

Judge 的职责是保护正确性、物理合理性和可复现性。任何候选不能只因为单个指标好看就被提升为 current_best。

## 裁决输入

- `OperatorSpec`
- `OptimizationStrategy`
- 编译诊断
- correctness matrix
- benchmark matrix
- baseline matrix
- roofline/gap
- failure cases

## 裁决结果

- `KEEP`：全矩阵正确，综合性能提升，物理合理，可提升为 current_best。
- `SPECIALIZE`：部分配置明显提升但整体不提升，保存为 specialized version，不提升 current_best。
- `ROLLBACK`：正确但性能无提升，或收益不稳定。
- `REJECT`：正确性失败、编译失败、运行崩溃、违反接口或疑似作弊。

## KEEP 门槛

必须满足：

1. `OperatorSpec.evaluation.require_all_correct` 为 true 时，全测试矩阵正确。
2. 没有硬编码测试样例、跳过计算、内部同步污染计时。
3. 综合性能优于 current_best 或 baseline。
4. 预测加速不违反 roofline 物理上限。

## SPECIALIZE 门槛

适用：

- B=1 提升明显，但 B=16 变慢。
- 短序列提升，长序列变慢。
- 某个 headdim 专用版本有效。

必须记录适用条件：

```json
{"specialized_for": {"batch_size": 1, "seqlen_kv": ">=4096"}}
```

## 反作弊检查

- 不能依赖固定 seed 的输出。
- 不能根据 case_id 直接分支返回。
- 不能跳过 K/V 读取。
- 不能在 `run_kernel` 内部同步来操控计时。

## 输出模板

```text
verdict: KEEP|SPECIALIZE|ROLLBACK|REJECT
operator_id:
strategy_id:
reason:
correctness:
performance:
roofline:
promotion:
memory_update:
```
