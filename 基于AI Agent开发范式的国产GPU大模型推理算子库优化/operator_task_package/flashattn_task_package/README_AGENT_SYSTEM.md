# FlashAttention 算子优化 Agent 系统（真实实现）

## 当前推荐闭环入口

本项目当前的主线已经切换为确定性闭环多 Agent 架构，并采用分级 Coder：

```text
Analyst -> Coder(macro/meso/micro ChangeProposal) -> run_closed_loop -> Judge -> Reflector/Logger
```

核心入口：

```bash
python scripts/run_closed_loop.py --phase explore --rounds 1
python scripts/run_closed_loop.py --phase stabilize --rounds 1
python scripts/run_closed_loop.py --phase tune --rounds 1
python scripts/run_closed_loop.py --real --phase tune --rounds 1 --batch 1 --seq-kv 4096 --headdim 128
```

产物统一写入：

- `runs/<run_id>/rounds/round_###/`
- `runs/<run_id>/logs/`
- `runs/<run_id>/versions/`
- `results/latest_run.txt`
- `results/opencode_events.jsonl`

前期允许 `scale=macro` 的结构性候选，后期再切换到 `scale=micro` 的小补丁微调；所有候选都必须通过闭环验证，不能直接覆盖 current best。详细说明见 `CLOSED_LOOP_RUNBOOK.md`。

> 沐曦 C500 / MXC500 国产 GPU 上的 FlashAttention Decode 算子优化，由真实 AI Agent 驱动。  
> 赛题：基于 AI Agent 开发范式的国产 GPU 大模型推理算子库优化 — Track 2 FlashAttention
> **状态：完整实现（非原型）—— 真实 LLM + 真实编译 + 真实 GPU 运行**

## 已验证的真实能力（非 mock）

| 能力 | 实现 | 验证状态 |
|------|------|---------|
| mxcc 编译 .cu → .so | `kernel_compiler.py` | ✅ 真实编译通过 |
| ctypes 加载 .so 调用 run_kernel | `kernel_loader.py` | ✅ 真实 GPU 运行 |
| 正确的 attention decode kernel | `kernel/baseline_kernel.cu` | ✅ allclose 通过 |
| 模力方舟 API 真实调用 | `llm_client.py` | ✅ MiniMax-M2.7 调通 |
| LLM 生成 kernel 代码 | `real_coder.py` | ✅ 真实生成+解析 |
| LLM 性能预测 + roofline clip | `real_cost_model.py` | ✅ 真实预测 |
| 完整闭环（生成→编译→运行→校验→记录）| `real_orchestrator.py` | ✅ 真实跑通 |
| 失败案例自动记录 | `domain_memory.py` | ✅ 真实记录 2 条 |

## 快速开始

```bash
cd flashattn_task_package/

# 1. 全部测试（132 通过，9 slow 跳过）
python -m pytest tests/ -v

# 2. 演示（三大创新点）
python scripts/demo.py

# 3. 真实 GPU baseline（需 GPU）
python -m pytest tests/test_e2e.py::TestRealBaseline --runslow -s

# 4. 真实 LLM 闭环（消耗 token + GPU）
python -m pytest tests/test_real_e2e.py --runslow -v -s
```

## 泛化多 Agent 框架（新增）

当前系统已从单一 FlashAttention 原型升级为可注册多算子的 Agent 优化框架：

| 层 | 模块 | 作用 |
|----|------|------|
| 算子规格 | `agent_system/specs.py` | 用 `OperatorSpec` 描述接口、shape、测试矩阵、精度和优化空间 |
| 算子注册 | `agent_system/operator_registry.py` | 注册 `flashattention_kvcache_decode` 与泛化 smoke 算子 |
| 后端适配 | `agent_system/backend_adapter.py` | 封装 MACA C++ / mock 后端，后续可扩展 Triton、TileLang |
| 评测器 | `agent_system/evaluator.py` | 把 correctness、benchmark、score 从 orchestrator 中解耦 |
| 策略 schema | `agent_system/strategy_schema.py` | Coder 输出结构化优化策略，而不是只吐整份代码 |
| 泛化编排 | `agent_system/generic_orchestrator.py` | Operator-agnostic 多 Agent 优化循环 |
| 版本库 | `agent_system/kernel_version_store.py` | KEEP 后持久化最优代码，支持下一轮继续演化 |

非 GPU smoke：

```bash
python scripts/run_agent_smoke.py
```

环境检查：

```bash
python scripts/check_env.py
```

MCP 新增工具：

- `list_operators`
- `analyze_operator`
- `get_operator_spec`
- `validate_strategy`
- `run_agent_smoke`
- `list_kernel_versions`

新增泛化 skills：

- `operator-spec-contract`：新增/读取算子规格时使用
- `strategy-generation`：Coder 生成结构化 OptimizationStrategy 时使用
- `judge-policy`：Judge 裁决 KEEP/ROLLBACK/REJECT/SPECIALIZE 时使用
- `memory-reflection`：Reflector 更新 failure_cases/hardware_belief 时使用

推荐 Agent 工作流：

```text
list_operators
→ get_operator_spec
→ analyze_operator
→ Coder 输出 OptimizationStrategy JSON
→ validate_strategy
→ Profiler 编译/正确性/benchmark
→ Judge 裁决
→ Reflector 更新记忆
→ KernelVersionStore 持久化 KEEP 版本
```

## 三大创新点（全部真实实现）

| 创新点 | 模块 | 状态 |
|--------|------|------|
| **A 主** Roofline-Anchored LLM | `roofline_engine.py` | ✅ 实测验证（47/47配置memory-bound）|
| **B 辅** Co-Evolving Hardware Belief | `domain_memory.py` | ✅ 真实记录失败案例 |
| **C 增强** 双层候选过滤 | `llm_cost_model.py` + `real_cost_model.py` | ✅ 真实 LLM 预测 |

## 真实闭环架构

```
LLM(MiniMax-M2.7)生成候选
  → 双层过滤(roofline物理+LLM预测)
  → mxcc真实编译.cu→.so
  → ctypes加载→GPU真实运行
  → torch.allclose正确性校验
  → GPU benchmark精确计时
  → Judge裁决(KEEP/ROLLBACK)
  → Reflector记录失败/信念到记忆库
  → Logger生成优化日志
```

## 项目结构

```
flashattn_task_package/
├── agent_system/                  # ★ 核心代码（真实实现）
│   ├── roofline_engine.py         # 创新点A：Roofline物理模型
│   ├── correctness.py             # PyTorch参考实现+allclose
│   ├── benchmark_engine.py        # GPU计时+带宽分析
│   ├── llm_cost_model.py          # 创新点C：双层过滤框架
│   ├── domain_memory.py           # 创新点B：领域记忆+硬件信念
│   ├── optimization_log.py        # 优化日志（可复现物料）
│   ├── orchestrator_loop.py       # 闭环调度（mock版，已测）
│   ├── kernel_compiler.py         # ★ 真实mxcc编译
│   ├── kernel_loader.py           # ★ 真实ctypes加载+GPU运行
│   ├── llm_client.py              # ★ 真实模力方舟API
│   ├── real_coder.py              # ★ 真实LLM生成kernel
│   ├── real_cost_model.py         # ★ 真实LLM性能预测
│   ├── real_orchestrator.py       # ★ 真实完整闭环
│   └── domain_memory/failure_cases/seed_cases.md  # 错误衍生监督种子
├── kernel/
│   └── baseline_kernel.cu         # ★ 正确的baseline（allclose通过）
├── tests/                         # 141个测试
│   ├── test_roofline_engine.py (26)
│   ├── test_correctness.py (17)
│   ├── test_kernel_compiler_loader.py (9)  # ★ 真实编译+运行
│   ├── test_llm_client.py (12)             # ★ 真实API
│   ├── test_real_e2e.py (5)                # ★ 真实LLM闭环
│   └── ...
├── .opencode/                     # OpenCode Agent配置
│   ├── agent/ (5个角色)
│   └── skills/ (3个核心skill)
├── AGENTS.md                      # Agent行为规范
└── scripts/demo.py                # 一键演示
```

## 真实实验数据

| 指标 | 数值 | 来源 |
|------|------|------|
| C500 带宽 | 1.8 TB/s | 沐曦官网 |
| 官方baseline峰值 | 821 GB/s (45.6%) | benchmark CSV |
| MXC500 profiler | 不存在 | 本地实测 |
| 真实LLM生成候选 | 3个/轮 | MiniMax-M2.7 |
| 真实编译失败率 | ~67%（API误用）| 实测（印证AscendKernelGen）|
| 真实失败自动记录 | 2条/轮 | domain_memory |

## 关键发现（答辩素材）

1. **MXC500 无 profiler** → 创新点A（roofline替代）有实证
2. **LLM 在国产硬件 API 误用率高** → 创新点B（领域记忆）印证 AscendKernelGen
3. **mctlass::bfloat16_t 是正确路径**（非 __maca_bfloat16 直接用）→ 真实编译验证
4. **完整闭环零崩溃** → 工程可靠性验证
