---
name: operator-spec-contract
description: Use when adding a new operator task, reading OperatorSpec, or adapting the multi-agent loop to a non-FlashAttention operator. Trigger keywords: OperatorSpec, BackendSpec, EvaluationSpec, operator registry, 泛化, 跨算子, multi-agent framework.
---

# OperatorSpec Contract

本 skill 定义多 Agent 系统的泛化契约。任何算子优化任务都必须先被描述成 `OperatorSpec`，再交给 Analyst/Coder/Profiler/Judge。

## 核心原则

- Orchestrator 不直接认识某个具体算子。
- 算子接口、shape、测试矩阵、精度、后端、优化空间全部从 `OperatorSpec` 读取。
- 新增算子时，优先新增 spec/evaluator/backend adapter，不改主循环。

## 规格结构

`agent_system/specs.py` 中的关键 dataclass：

- `OperatorSpec`
- `TensorSpec`
- `TestCaseSpec`
- `AccuracySpec`
- `BackendSpec`
- `OptimizationSpace`
- `OptimizationParam`
- `EvaluationSpec`

## 新增算子的最小步骤

1. 在 `agent_system/operator_registry.py` 注册一个 `OperatorSpec`。
2. 描述输入输出：
   - name
   - shape
   - dtype
   - layout
   - role
3. 描述测试矩阵：
   - case_id
   - params
   - tags
   - weight
4. 描述 backend：
   - `maca_cpp`
   - `triton`
   - `tilelang`
   - `mock`
5. 描述 evaluation：
   - correctness 方法
   - rtol/atol
   - benchmark warmup/repeats
6. 描述 optimization space：
   - 参数名
   - 候选值
   - 默认值
   - 风险
   - strategy_names

## 反模式

- 在 `real_orchestrator.py` 或 `generic_orchestrator.py` 写死 batch/seq/head_dim。
- Coder 直接猜接口，而不读取 `get_operator_spec`。
- 把 FlashAttention 的 memory-bound 结论套到 GEMM/MoE。
- 在 MCP 工具中为每个算子复制一套主循环。

## 推荐检查

使用 MCP：

```text
list_operators
get_operator_spec(operator_id="flashattention_kvcache_decode")
analyze_operator(operator_id="flashattention_kvcache_decode")
```

非 GPU smoke：

```bash
python scripts/run_agent_smoke.py
```
