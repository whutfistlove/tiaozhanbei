# Track 2 FlashAttention Agent 优化系统重规划

> 日期：2026-07-16  
> 范围：`flashattn_task_package`，面向 XPU-OJ FlashAttention KV Cache Decode / `flash_attn_with_kvcache`。  
> 目标：把当前“可演示原型”推进为“可复现、可提交、可持续迭代”的 Agent 驱动算子优化系统。

---

## 1. 当前项目判断

当前项目已经不是空白工程，已有较完整的 Agent 系统雏形：

- `agent_system/roofline_engine.py`：可做 roofline 下限、带宽利用率、split-k 建议。
- `agent_system/correctness.py`：已有 PyTorch 参考实现与 allclose 校验。
- `agent_system/benchmark_engine.py`：已有计时、带宽、gap_to_roofline 计算。
- `agent_system/llm_cost_model.py`、`real_cost_model.py`：已有双层过滤框架与 LLM 预测入口。
- `agent_system/domain_memory.py`：已有失败案例与 hardware belief 持久化。
- `mcp/mcp_server.py`、`.opencode/opencode.json`：已有 OpenCode/MCP 工具化入口。
- `.opencode/agent/*` 与 `.opencode/skills/*`：已有多角色和技能库雏形。

但它仍然是一个不完整原型，主要缺口如下：

1. **算子实现未满足性能主线**  
   当前 `kernel/baseline_kernel.cu` 与 `kernel/splitk_h128.cu` 主要是手写 CUDA/MACA 循环，只用了 `mctlass::bfloat16_t` 类型，没有真正使用 mctlass Tensor Core / GEMM / softmax 原语链。`hardware_belief.json` 也已经记录：手写 Split-K 正确但很慢，必须转向 mctlass。

2. **正确性覆盖不足**  
   当前真实 GPU 集成测试多使用 `headdim=4/32` 的小配置，不能证明赛题固定配置 `B={1,4,16}, seq={1024,4096,8192,16384}, D=128` 全部通过。`baseline_kernel.cu` 注释还保留 `headdim<=32` 的假设，与 D=128 不一致。

3. **Agent 闭环未真正闭合**  
   `real_orchestrator.py` 找到 KEEP 后没有把 `best_code` 写入版本库或作为下一轮 `current_code`，多轮优化状态无法自然进化。

4. **Benchmark 与 OJ 反馈未打通**  
   项目有 benchmark CSV 和 roofline 分析，但缺少统一的“全矩阵跑分 -> 与官方 baseline 对齐 -> 生成提交报告 -> 解析 OJ SPJ report -> 更新记忆”的流程。

5. **工程可复现性不足**  
   `scripts/reproduce.sh` 是好的入口，但当前本地 Python 环境缺 `pytest`，且脚本没有环境自检与依赖修复建议。文档声称“132 通过”需要可验证。

6. **文档与实际代码有偏差**  
   `AGENT_DESIGN_PLAN.md` 是愿景型“最终版”，`README_AGENT_SYSTEM.md` 声称“完整实现”，但实际核心 mctlass 算子、多轮状态更新、全矩阵验证仍未完成。

---

## 2. 重规划总目标

### 2.1 近期目标：可提交

在当前 OJ 题包范围内，先稳定覆盖：

- `HEAD_DIM=128`
- `BATCH={1,4,16}`
- `SEQ_KV={1024,4096,8192,16384}`
- `NUM_HEADS=NUM_HEADS_K=8`
- `PAGE_BLOCK_SIZE=16`
- `BF16`
- `rtol=atol=1e-2`

验收标准：

- 全 12 个配置正确性通过。
- `run_kernel` 不做内部同步。
- 至少一个配置快于官方 baseline 或当前手写版本。
- 每次提交能生成完整复现日志。

### 2.2 中期目标：能打榜

- 用 mctlass 或 mctlass 基础原语重写 QK/PV 核心计算。
- 小 batch 长序列场景通过 Split-K 提升并行度。
- 对每个测试点形成可解释的性能报告：`T_k / T_b / T_h / gap / bandwidth / score estimate`。

### 2.3 长期目标：Agent 作品完整

- Agent 能自动完成“分析 -> 候选生成 -> 编译 -> 正确性 -> benchmark -> 裁决 -> 版本固化 -> 记忆更新”的闭环。
- Skill 和 failure case 随迭代增长，能证明 Agent 真实参与优化。
- 文档、PPT、演示脚本、日志与代码一致。

### 2.4 核心补充：Agent 必须具备泛化性

比赛核心不是“人工把某个 FlashAttention kernel 调快”，而是证明 Agent 形成了可迁移的算子优化范式。因此本项目后续设计必须遵守：

- **任务规格抽象化**：Agent 不应把 `flash_attn_with_kvcache` 的 shape、接口、score 公式硬编码在主循环里，而应读取 `OperatorSpec`。
- **后端抽象化**：编译、加载、运行、benchmark 通过 `BackendAdapter` 完成，当前是 MACA C++，后续可扩展 Triton / TileLang / 其他国产 GPU 后端。
- **评测抽象化**：correctness、benchmark、score estimate 通过 `Evaluator` 完成，FlashAttention 只是一个 evaluator 实例。
- **优化策略可迁移**：Split-K、online softmax、paged addressing、roofline analysis、failure-case learning 都应沉淀为可检索 skill，而不是散落在某个 kernel 文件中。
- **记忆结构通用化**：failure case 和 hardware belief 需要记录“适用条件、触发症状、修复动作、验证配置”，以便迁移到 FlashInfer/MoE 等任务。
- **Coder 输出结构化**：Agent 优先输出优化意图和参数空间，再由模板/后端渲染代码；减少“LLM 写整份代码”的不可控性。

也就是说，FlashAttention Track 2 应被当作 **泛化 Agent 优化框架的第一个场景**。最终答辩叙事应从“我优化了一个算子”升级为“我构建了一个能理解算子规格、形成候选、自动验证并沉淀经验的国产 GPU 算子优化 Agent”。

---

## 3. 阶段性实施计划

## 阶段 -1：建立泛化 Agent 内核

目标：先把 Agent 主循环从 FlashAttention 细节中解耦，形成可复用优化框架。

步骤：

1. 新增 `agent_system/specs.py`：
   - `OperatorSpec`：算子名称、接口签名、输入输出 shape、dtype、精度要求、测试矩阵。
   - `BackendSpec`：后端语言、编译命令、include 路径、运行时约束。
   - `OptimizationSpace`：可调参数、合法范围、互斥关系、风险标签。
   - `EvaluationSpec`：correctness、benchmark、score、稳定性规则。

2. 新增 `agent_system/operator_registry.py`：
   - 注册 `flashattention_kvcache_decode`。
   - 后续可注册 `flashinfer_paged_decode`、`flashinfer_paged_prefill`、`fused_moe_i8_tn`。
   - Orchestrator 只通过 registry 获取任务，不直接 import 某个具体题目的常量。

3. 新增 `agent_system/backend_adapter.py`：
   - `MacaCppBackend`：封装当前 `mxcc -> .so -> ctypes`。
   - 预留 `TritonBackend`、`TileLangBackend` 接口。
   - 编译错误统一输出 `CompileDiagnostic`，供 Reflector 学习。

4. 新增 `agent_system/evaluator.py`：
   - `CorrectnessEvaluator`
   - `BenchmarkEvaluator`
   - `ScoreEvaluator`
   - `RegressionEvaluator`
   - FlashAttention 的 PyTorch reference 作为 evaluator plugin，而不是写死在 orchestrator 中。

5. 新增 `agent_system/strategy_schema.py`：
   - 定义 Agent 候选输出 JSON schema：
     - `strategy_name`
     - `target_operator`
     - `backend`
     - `params`
     - `expected_effect`
     - `risk`
     - `required_skills`
   - Coder 优先输出 strategy，不优先输出整份 `.cu`。

验收：

- `real_orchestrator.py` 的输入从 `KernelConfig` 升级为 `OperatorSpec + BackendSpec + EvaluationSpec`。
- FlashAttention 当前任务可通过 registry 跑通原有流程。
- 新增一个 dummy operator spec，可不写 kernel，但能通过 Analyst/Judge 框架生成分析报告，证明框架不是单题硬编码。

## 阶段 0：基线审计与环境封口

目标：把当前项目真实状态固定下来，避免继续在不确定基础上优化。

步骤：

1. 增加环境检查脚本 `scripts/check_env.py`：
   - 检查 Python、pytest、torch、flash_attn、einops。
   - 检查 `MACA_PATH`、`mxcc`、`mctlass` 头文件。
   - 检查 GPU 是否可用，输出 `mx-smi` 摘要。

2. 整理当前能力矩阵：
   - 哪些测试是纯 CPU/mock。
   - 哪些测试需要 GPU。
   - 哪些测试需要 `MOARK_API_KEY`。
   - 哪些测试是真实 D=128。

3. 修正 README 表述：
   - 把“完整实现”改为“可运行原型 + 待完成闭环”。
   - 明确当前性能瓶颈：手写 Split-K 慢于官方 baseline。

4. 固化当前 kernel 版本：
   - 将 `kernel/baseline_kernel.cu`、`kernel/splitk_h128.cu` 的正确性/性能记录写入 `agent_system/kernel_versions/manifest.json`。
   - 给每个版本记录：正确性覆盖、性能覆盖、是否符合 mctlass 约束。

验收：

- `python scripts/check_env.py` 能给出明确 PASS/WARN/FAIL。
- `README_AGENT_SYSTEM.md` 与实际代码状态一致。
- 当前版本可复现信息完整。

---

## 阶段 1：正确性优先，建立 D=128 全矩阵回归

目标：先保证赛题固定矩阵的正确性，不再依赖 `headdim=4/32` 的演示配置。

步骤：

1. 新增 `tests/test_oj_matrix_correctness.py`：
   - 覆盖 12 个核心配置。
   - 每个配置固定 seed。
   - 先跑 candidate 输出，再跑 PyTorch reference。
   - 输出 `max_abs / max_rel / mean_abs`。

2. 修正当前 baseline kernel 的 D=128 问题：
   - 当前 `baseline_kernel.cu` 单 warp 只覆盖 32 个维度，不适合 D=128。
   - 先实现一个“慢但正确”的 D=128 参考 kernel，作为 OJ 冒烟版本。
   - 允许用手写循环作为 correctness baseline，但标注“不作为最终性能方案”。

3. 统一 head 映射和 paged addressing：
   - 把公式写成单独文档与单元测试。
   - 覆盖 `num_heads == num_heads_k` 和 GQA 的泛化逻辑。

4. 增加边界测试：
   - `seqlen_kv` 不整除 split。
   - `cache_seqlens[b] < seqlen_k`。
   - `num_blocks / batch_size` 与 `block_table` 边界。

验收：

- 慢速参考 kernel 在 12 个核心配置全部 allclose 通过。
- 错误时能定位到配置、最大误差、可能的寻址点。
- `run_correctness` MCP 支持一次跑全矩阵。

---

## 阶段 2：Benchmark 基线与评分模型对齐

目标：把性能评价从“单次 time_ms”升级为“可对照 OJ 的全矩阵报告”。

步骤：

1. 新增 `scripts/run_oj_matrix.py`：
   - 输入 `.cu` 或 `.so`。
   - 编译、加载、全矩阵 correctness、全矩阵 benchmark。
   - 输出 CSV + Markdown。

2. 扩展 `benchmark_engine.py`：
   - 支持官方 baseline 数据导入。
   - 支持 `T_b/T_k/T_h` 与估算分数。
   - 对每个配置输出 `speedup_vs_baseline` 和 `gap_to_roofline`。

3. 对齐现有 CSV：
   - 当前 `benchmark_kvcache_*.csv` 是 headdim=256，不能直接证明 D=128 赛题表现。
   - 需要重新生成 D=128 的官方 baseline CSV。

4. 增加性能异常检测：
   - `bandwidth_utilization > 1.1` 标记 bytes 估算异常。
   - `std/mean > 10%` 标记计时不稳定。
   - candidate 慢于 correctness reference 时标记为无效。

验收：

- 每个 kernel 版本都有 `reports/bench_<version>.csv` 与 `reports/bench_<version>.md`。
- 能看到全 12 配置的 `T_k/T_b/T_h/score_estimate`。
- Agent 的 Judge 可以基于报告做 KEEP/ROLLBACK。

---

## 阶段 3：闭合真实 Agent 优化循环

目标：让 `real_orchestrator.py` 不再只是“一轮演示”，而是能持续迭代并保留最优版本。

步骤：

1. 修复 `run_real_iteration` 的状态传递：
   - 返回 `best_code`、`best_so_path`、`best_report_path`。
   - KEEP 时写入 `agent_system/kernel_versions/vN_<tag>.cu`。
   - 更新 `manifest.json` 的 Pareto 信息。

2. 增加版本管理器 `kernel_version_store.py`：
   - `save_candidate`
   - `promote_best`
   - `rollback`
   - `load_current_best`
   - `list_pareto`

3. 调整 Judge 规则：
   - 正确性失败：REJECT。
   - 全矩阵正确但几何平均慢于 baseline：ROLLBACK。
   - 局部配置更快但整体变慢：保存为 specialized，不提升为 current_best。
   - 全矩阵分数提升：KEEP。

4. 增加预测校准：
   - 将 LLM 预测和真实结果写入 `prediction_history.jsonl`。
   - 若方向准确率低于 50%，自动降低 LLM 过滤权重，退化为全量评测。

验收：

- 连续 3 轮真实迭代后，`current_best` 自动更新或明确回滚。
- `optimization_log.md` 能追溯每轮候选、裁决、性能变化。
- `hardware_belief.json` 只记录经过 benchmark 支撑的规则。

---

## 阶段 4：mctlass 主路径落地

目标：突破手写 CUDA 循环的性能天花板，满足“核心矩阵计算必须使用 mctlass”的约束。

步骤：

1. 建立最小 mctlass GEMM 冒烟样例：
   - 从 `mctlass-usage` skill 中抽取最小 `MacaGemmUniversal` 示例。
   - 先做普通 BF16 GEMM 正确性测试，不接 attention。

2. 做 QK tile 原型：
   - 固定 `D=128`。
   - 每个 tile 处理 `Q(1,D) x K(tile,D)^T`。
   - 输出 score tile 或 partial score。
   - softmax 暂时可手写，先验证 mctlass 调用正确。

3. 做 PV tile 原型：
   - 输入 softmax 权重 tile 与 V tile。
   - 使用 mctlass 或基础 MMA 原语做累加。

4. 融合 online softmax：
   - 引入 `(m, l, o)` 三元组。
   - 每个 tile 更新在线 softmax，避免完整 score 中间落 HBM。

5. 替换手写标量点积：
   - 先替换 QK。
   - 再替换 PV。
   - 最后融合 QK/softmax/PV。

验收：

- 代码中核心 QK/PV 不再是纯标量 for-loop。
- D=128 全矩阵正确。
- 至少在 B=1 或长序列配置达到官方 baseline 的 60% 以上，作为进入下一阶段门槛。

---

## 阶段 5：Split-K / Split-KV 性能路线

目标：解决 decode 阶段小 batch 并行度不足。

步骤：

1. 重新设计 split 策略：
   - `B=1`：split 8~16。
   - `B=4`：split 2~8。
   - `B=16`：split 1~2。
   - 依据真实 benchmark 自动调参，而不是固定常数。

2. 改造 partial 存储：
   - `partial_o` 优先 fp32，避免 BF16 中间截断。
   - `partial_m`、`partial_l` fp32。
   - 避免 `run_kernel` 内反复 `mcMalloc/mcFree` 造成计时污染，改为静态 workspace 或外部缓存策略。

3. 优化 reduce kernel：
   - 对 split 数小的场景，用一个 block 合并所有 split。
   - 对 `D=128` 做向量化写回。
   - 评估 FlashDecoding++ 的统一 max 策略，减少二次同步和 HBM 往返。

4. 建立 autotune 搜索空间：
   - `num_splits`
   - `tile_n`
   - `num_warps`
   - `stages`
   - `vector_width`

验收：

- B=1 长序列不再严重低带宽。
- split 多时 reduce 开销可解释、可被日志捕获。
- 每个配置的最佳 split 被记录到 hardware belief。

---

## 阶段 6：候选生成从“LLM 写整份代码”转为“结构化补丁”

目标：降低 LLM 生成不可编译代码的概率，让 Agent 更像工程自动调优系统。

步骤：

1. 约束 Coder 输出：
   - 不直接生成完整 `.cu`。
   - 输出结构化 patch plan：修改参数、模板常量、tile 配置、split 策略。
   - 只有在新增 kernel 变体时才生成代码块。

2. 建立模板化 kernel：
   - `templates/decode_mctlass.cu.j2`
   - `templates/splitk_reduce.cu.j2`
   - `templates/config_space.json`

3. Coder 只选择策略：
   - `variant=mctlass_qk_tile`
   - `split_k=8`
   - `tile_n=64`
   - `num_warps=4`

4. Profiler 生成候选源码：
   - 用模板渲染。
   - 编译失败时把模板参数和错误分类写入 failure cases。

验收：

- 编译失败率显著下降。
- 每轮候选可结构化比较。
- Agent 日志能说明“为什么选这个候选”。
- 同一套候选 schema 能描述 FlashAttention 的 Split-K、FlashInfer 的 page scheduling、MoE 的 tile/block 配置。

---

## 阶段 7：多 Agent 工作流产品化

目标：让多角色不只是配置文件，而是真正有输入输出契约。

步骤：

1. Analyst：
   - 输入：全矩阵 report + OJ SPJ report。
   - 输出：`reports/bottleneck_<run>.md`。

2. Coder：
   - 输入：bottleneck report + memory context + current_best。
   - 输出：候选配置 JSON + patch/code。

3. Profiler：
   - 输入：候选列表。
   - 输出：compile/correctness/benchmark report。

4. Judge：
   - 输入：Profiler report。
   - 输出：KEEP/ROLLBACK/REJECT + 理由。

5. Reflector：
   - 输入：失败日志和成功报告。
   - 输出：failure_cases、hardware_belief、skill 更新建议。

6. Logger：
   - 输入：每轮结构化事件。
   - 输出：`optimization_log.md/jsonl`、`docs/agent_logs/*`。

验收：

- 每个角色产物都落盘。
- `scripts/run_agent_loop.py --rounds 3` 可复现一段完整优化历史。
- 没有用户手工补写日志的必要。
- 更换 `--operator` 参数时，Agent 主循环不需要改代码，只需要切换 OperatorSpec 和 evaluator plugin。

---

## 阶段 7.5：跨算子泛化验证

目标：用最小成本证明 Agent 范式可迁移，不只是 FlashAttention 单点工程。

步骤：

1. 选择一个轻量泛化对象：
   - 优先：`flashinfer_task_package` 中的 paged decode 或 paged prefill benchmark。
   - 备选：`fused_moe_task_package` 的 i8 tn pybind benchmark。

2. 为该对象编写最小 `OperatorSpec`：
   - 输入输出 shape。
   - correctness 规则。
   - benchmark 命令。
   - 允许的优化空间。

3. 只跑 Agent 分析与候选生成，不强求完成高性能 kernel：
   - Analyst 能识别瓶颈。
   - Coder 能给出结构化策略。
   - Profiler 能调用对应 benchmark 或给出缺失工具诊断。
   - Reflector 能把失败归档到通用 memory。

4. 形成 `docs/generalization_report.md`：
   - FlashAttention 完整闭环。
   - 第二算子最小迁移验证。
   - 哪些组件复用，哪些组件需新增。

验收：

- 至少两个 `OperatorSpec` 存在。
- Agent 能在不修改主循环的情况下切换 operator。
- 答辩中可以展示“泛化接口 + FlashAttention 深度落地 + 第二算子浅层迁移验证”。

---

## 阶段 8：提交物与答辩材料收口

目标：把工程、性能和 Agent 创新叙事统一起来。

步骤：

1. 更新文档：
   - `README_AGENT_SYSTEM.md`：运行方式、环境、当前最好成绩。
   - `AGENT_DESIGN_PLAN.md`：保留创新叙事，但删掉未实现的“已完成”口吻。
   - 新增 `docs/final_report.md`：技术路线、实验结果、失败案例、Agent 参与证据。

2. 增加一键复现：
   - `scripts/reproduce.sh` 先跑环境检查。
   - 再跑全矩阵正确性。
   - 再跑全矩阵 benchmark。
   - 最后输出日志摘要。

3. 准备答辩证据：
   - Agent prompt / tool call / 修改记录。
   - before/after 性能表。
   - mctlass 使用证据。
   - failure case 如何反向约束后续生成。

验收：

- 新机器上按 README 能复现功能结果。
- 至少能复现标称性能的 80%。
- 文档与代码、日志、benchmark 数据互相一致。

---

## 4. 优先级排序

P0：必须立即做

- 建立 `OperatorSpec / BackendAdapter / Evaluator` 三个泛化接口，避免主循环继续硬编码 FlashAttention。
- D=128 全矩阵 correctness。
- 修复真实 orchestrator 的 KEEP 后版本持久化。
- 建立全矩阵 benchmark report。
- 改正 README/计划与实际不一致的表述。

P1：性能主线

- 最小 mctlass GEMM 冒烟。
- mctlass QK/PV tile 原型。
- Split-K workspace 和 reduce 优化。

P2：Agent 作品质量

- 结构化候选模板。
- prediction_history 校准。
- OJ report parser。
- Skill 自动沉淀。
- 第二算子最小泛化验证。

P3：扩展能力

- 支持 `headdim=[64,128,256]`。
- 支持更长 `seq_kv`。
- 支持 causal/window 等更完整 FlashAttention 参数。

---

## 5. 建议的两周执行节奏

### 第 1-2 天：封口与真基线

- 定义 `OperatorSpec / BackendSpec / EvaluationSpec`，把 FlashAttention 当前常量迁入 spec。
- 完成环境检查脚本。
- 修正文档状态。
- 跑通或补齐 D=128 正确性参考 kernel。
- 生成全矩阵 baseline report。

### 第 3-4 天：闭环工程

- 修复 `real_orchestrator.py` 状态更新。
- 实现 `kernel_version_store.py`。
- MCP 增加全矩阵 correctness/benchmark。
- 日志包含候选源码路径和报告路径。

### 第 5-7 天：mctlass 冒烟与替换

- 跑通最小 mctlass BF16 GEMM。
- 接入 QK tile。
- 接入 PV tile。
- 证明一个 D=128 小配置正确。

### 第 8-10 天：Split-K + Autotune

- 固化 split 搜索空间。
- 优化 partial workspace。
- 跑 B=1/4/16 三类配置。
- 更新 hardware belief。

### 第 11-12 天：Agent 候选模板化

- Coder 输出结构化候选。
- Profiler 自动渲染模板。
- Cost model 记录预测误差。

### 第 13-14 天：收口复现与答辩

- 一键复现脚本。
- final report。
- PPT 素材。
- OJ 提交记录与性能表。

---

## 6. 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| mctlass API 复杂，LLM 容易误用 | 编译失败率高 | 先人工/Agent 共同固化最小可编译模板，再让 LLM 调参 |
| 手写 Split-K 正确但慢 | 无法打榜 | 明确停止手写标量主线，转 mctlass/Tensor Core |
| D=128 全矩阵慢或不稳定 | 无法提交 | 先保证慢速正确版本可提交，再逐项替换性能内核 |
| benchmark 与 OJ 不一致 | 本地成绩不可用 | 增加 OJ report parser，以 OJ 结果更新判断 |
| Agent 只是“文档型” | 可复现性失分 | 所有角色产物落盘，日志记录 prompt、候选、裁决、结果 |
| LLM 预测误导过滤 | 错过好候选 | 记录 prediction_history，低准确率时退化全量评测 |

---

## 7. 下一步建议

下一步不要继续扩写概念文档，建议直接从四个工程动作开始：

1. 先抽象 `OperatorSpec / BackendAdapter / Evaluator`，把 FlashAttention 从主循环硬编码中拆出来。
2. 实现 `scripts/check_env.py` 和全矩阵 correctness runner。
3. 修复 `real_orchestrator.py` 的 best code 持久化。
4. 写一个最小 mctlass BF16 GEMM 冒烟测试，作为替换手写 QK/PV 的起点。

这四件事完成后，项目会从“看起来完整的单题原型”进入“可以真实迭代、且具备跨算子迁移能力的 Agent 优化系统”。
