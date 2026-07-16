# 论文精读笔记：MARCO + AscendKernelGen

> 本文档是对两篇与本题最相关论文的深度精读，提取可直接落地到"用 OpenCode Agent 优化沐曦 C500 FlashAttention 算子"的架构细节。
> 精读人：ZCode Agent | 日期：2026-07-15

---

## 论文一：MARCO（最同构的系统先例）

**标题**：MARCO: Multi-Agent Code Optimization with Real-Time Knowledge Integration for High-Performance Computing
**arXiv**：2505.03906 | **作者**：Asif Rahman 等 (Virginia Tech) | **年份**：2025
**链接**：https://arxiv.org/abs/2505.03906

### 1.1 核心思想（一句话）
用**分离的多个 agent**（代码生成 agent + 性能评估 agent）+ **反馈闭环** + **实时 web 搜索**来增强 LLM 生成 HPC 代码，比单模型 Claude 3.5 Sonnet 平均减少 14.6% 运行时间，加 web 搜索后再提升 30.9%。

### 1.2 架构（可直接借鉴的三层设计）

```
┌─────────────────────────────────────────────┐
│  MARCO 三 agent 架构                          │
├─────────────────────────────────────────────┤
│  ① Code Generation Agent（代码生成）          │
│     - 接收问题描述 + 优化反馈                  │
│     - 生成/优化代码                           │
│                                              │
│  ② Performance Evaluation Agent（性能评估）   │
│     - 独立运行代码、采集运行时间              │
│     - 生成性能反馈（不写代码，只评判）         │
│                                              │
│  ③ Web Search Agent（知识检索）               │
│     - 检索最新会议论文/优化技术               │
│     - 弥补 LLM 预训练知识滞后                 │
└─────────────────────────────────────────────┘
        ↕ 三者通过反馈闭环连接，迭代优化 ↕
```

**关键设计决策**：代码生成与性能评估**严格分离到不同 agent**。理由：同一个 agent 既写代码又评价自己，会产生"自评盲区"——倾向于认为自己写的代码没问题。

### 1.3 反馈闭环机制（MARCO 的核心创新）

```
迭代循环：
  Code Agent 生成代码 v_n
       ↓
  Perf Agent 评估 v_n → 产出"性能反馈报告"（哪里慢、为什么慢）
       ↓
  Web Agent 检索针对该瓶颈的最新优化技术
       ↓
  把 [性能反馈 + 检索到的优化知识] 一起喂给 Code Agent
       ↓
  Code Agent 生成改进版 v_{n+1}
       ↓ (重复直到收敛或达迭代上限)
```

**这个闭环直接对应到 OpenCode 实现**：
- Code Agent = `coder` subagent
- Perf Agent = `profiler` + `judge` subagent（跑 benchmark + 判定是否提升）
- Web Agent = 可用 OpenCode 的 `websearch` / `webfetch` 工具，或把 FlashAttention 论文/mctlass 文档做成 skill

### 1.4 可落地到本题的具体做法

| MARCO 组件 | 本题 OpenCode 落地 | 优先级 |
|-----------|-------------------|--------|
| Code Generation Agent | `.opencode/agent/coder.md`，每次只改一处 | ⭐⭐⭐⭐⭐ |
| Performance Evaluation Agent | `.opencode/agent/judge.md` + `.opencode/agent/profiler.md`，独立跑 benchmark | ⭐⭐⭐⭐⭐ |
| Web Search Agent | 把 FlashAttention-2/3 论文要点、mctlass 用法预先做成 skill（减少实时搜索依赖） | ⭐⭐⭐ |
| 反馈闭环 | Coder→Profiler→Judge→（失败则）反思喂回 Coder | ⭐⭐⭐⭐⭐ |
| 生成与评估分离 | **绝不让 coder 自己跑 benchmark 自评**，必须由 judge 独立验证 | ⭐⭐⭐⭐⭐ |

### 1.5 MARCO 的数据（对你的预期管理）
- 平均 14.6% 运行时间减少（vs Claude 3.5 单模型）
- 加 web 搜索后额外 30.9% 提升 → **说明"注入领域知识"价值巨大**
- **启示**：本题中把 mctlass 用法、Paged KV 寻址、online softmax 做成 skill 注入，可能比调 prompt 更有效

---

## 论文二：AscendKernelGen（国产硬件先例，答辩差异化核心）

**标题**：A Systematic Study of LLM-Based Kernel Generation for Neural Processing Units
**arXiv**：2601.07160 | **作者**：Xinzi Cao, Yonghong Tian 等（鹏城实验室+华为+北大）| **年份**：2026
**链接**：https://arxiv.org/html/2601.07160v2

### 2.1 核心发现（对你最重要的警示）

> **通用 LLM 在国产硬件上几乎完全失败。**

Table 1 的零样本测试结果（生成华为昇腾 AscendC kernel）：

| 模型 | 任务难度 | 编译成功率 | 功能正确性 |
|------|---------|-----------|-----------|
| Qwen3-8B | L1(简单) | 8.22% | 1.08% |
| Qwen3-8B | L2/L3(复杂) | 1.39% | **0.00%** |
| Qwen2.5-Coder-7B | L1 | 9.19% | 0.47% |
| Llama3.1-8B | L1 | 23.97% | 0.69% |
| Mistral-7B | 全部 | 0.00% | 0.00% |

**根因（论文分析的四大挑战，直接适用于沐曦 MACA）**：
1. **API 幻觉**：LLM 编造不存在的 API（如 `AscendC::Softmax`）→ 对应：mctlass API 也可能被编造
2. **长程语义依赖**：block index、tiling factor 跨阶段引用 → 对应：paged KV 寻址的 block_table 映射
3. **显式同步推理**：异步流水线的手动同步 → 对应：split-k 的 reduce 同步
4. **边界敏感算术**：offset/mask 计算 → 对应：尾部 page 有效 token 判断

### 2.2 错误分布（对你调试的指导）

论文统计了 ~4000 次失败生成：

| 错误类型 | 占比 | 本题对应 |
|---------|------|---------|
| **API 签名/重载错误** | **51.9%** | mctlass 函数参数类型/顺序错 |
| 数据类型/转换错误 | 19.8% | bf16↔float32 转换错 |
| 变量作用域/生命周期 | 16.4% | kernel 内变量未定义 |
| 内存/对象使用错误 | 8.1% | shared memory 越界 |
| 语法/结构错误 | 3.7% | 少见，说明 LLM 语法基本过关 |

**启示**：超过一半的失败是 **API 用错**。所以你的 skill 库里**最该优先沉淀的是 mctlass 的正确 API 用法**（参数类型、顺序、约束）。

### 2.3 AscendKernelGen 的解决方案（三件套）

```
① Ascend-CoT 数据集（83916 样本）
   - 文档型 CoT：从官方手册提炼 API 推理链
   - 代码型 CoT：从真实 kernel 提炼 tiling/pipeline 推理
   - 通用 CoT：保持泛化能力

② KernelGen-LM 模型（Qwen3-32B 微调）
   - 阶段1：SFT（含错误衍生监督）
   - 阶段2：RL（DPO，执行反馈偏好）

③ NPUKernelBench（158 个 kernel，三级难度）
```

**效果**：L2 复杂 kernel 编译成功率从 0% → 95.5% (Pass@10)，功能正确性 64.3%。

### 2.4 论文里最值得借鉴的两个具体技术

#### 技术 A：错误衍生监督（Error-Derived Supervision）—— 6.1 节

不只是用正确代码训练，还把**失败案例**变成训练数据：

```
编译失败日志 + 错误代码 + API 文档
       ↓ (LLM 分析根因)
"为什么这个 API 调用错了"的推理链 + 修正版
       ↓ (作为 SFT 数据)
```

两层错误修正：
- **API 级**：编译失败的 → 分析 API 误用 → 修正
- **Kernel 级**：编译过但数值错的 → 对比 ground truth → 重建

**本题落地**：你的 `optimization_log.md` 里记录的每次失败（错误日志+原因+修正），本身就是"错误衍生监督"的素材。这正是赛题"可复现性"要的——**失败的迭代也要完整记录**。

#### 技术 B：双路径评测（Dual-Path Evaluation）—— 7.3 节

```
路径1：Device-Only（只评 kernel 代码，host 固定）
       → 隔离 kernel 本身的能力，适合简单算子

路径2：Host+Device（评完整算子，含 host 调度）
       → 测试真实部署能力，含 shape 推断/tiling/launch
```

**本题落地**：你的 benchmark 应支持两种模式——只测 kernel 时间（对齐 OJ），和测端到端（含 launch 开销）。

### 2.5 对本题（沐曦 C500）的直接启示

| 启示 | 具体行动 |
|------|---------|
| 通用 LLM 在国产硬件会失败 | **必须**把 mctlass/MACA 的领域知识注入 agent（AGENTS.md + skill） |
| 51.9% 错误是 API 误用 | **优先**建 `mctlass-usage` skill，沉淀正确 API 签名 |
| 需要专用 benchmark | 你的 benchmark_kvcache.py + OJ SPJ 就是现成的"国产 benchmark" |
| 错误衍生监督有价值 | 失败迭代要完整记录到 optimization_log |
| SFT+RL 成本高 | 本题不允许改模型权重，**用 in-context skill 注入替代 SFT**，用 Reflexion 替代 RL |

### 2.6 答辩话术（差异化定位）

> "现有 LLM-kernel 工作（CudaForge、KernelBench 等）几乎都针对 NVIDIA CUDA。AscendKernelGen (2601.07160) 是唯一针对国产硬件(NPU)的工作，它证明通用 LLM 在国产硬件上功能正确率接近 0%。我们的工作针对**沐曦 C500/MACA** 这一另一种国产硬件，通过 [AGENTS.md 领域知识注入 + mctlass skill 库 + 错误衍生记录] 的 in-context 方案（无需微调），验证了 Agent 驱动的算子优化在国产 GPU 上的可行性。"

---

## 三、两篇论文综合 → 本题架构的最终修正

基于精读，对 AGENT_DESIGN_PLAN.md 的架构做三处强化：

### 强化1：Coder 与 Judge 必须严格分离（来自 MARCO）
- **绝不让** coder subagent 自己跑 benchmark 自评
- judge subagent 独立编译、运行、用 `torch.allclose(rtol=1e-2,atol=1e-2)` 硬校验
- 这是防 reward hacking（Sakana AI 教训）和自评盲区的关键

### 强化2：mctlass skill 优先级最高（来自 AscendKernelGen 错误分布）
- 51.9% 失败是 API 误用 → 第一个要建的 skill 是 `mctlass-usage`
- 内容：mctlass 的 NumericConverter、bfloat16_t、GEMM 原语的正确签名和用法
- 来源：`$MACA_PATH/include/mctlass/` 头文件 + 教程附录的转换代码

### 强化3：失败即资产（来自 AscendKernelGen 错误衍生监督）
- 每次 OJ Wrong Answer / 编译失败，不只是"回滚重来"
- 要记录：错误日志 + 根因分析 + 修正方案 → 沉淀到 `failure_cases.md`
- 这些失败案例会成为后续迭代的"负样本记忆"（Reflexion 思想）

---

## 四、精读结论

两篇论文从不同角度印证了 AGENT_DESIGN_PLAN 的架构方向，并给出三个**必须执行的强化**：

1. ✅ **生成/评估分离**（MARCO）→ coder 和 judge 独立 subagent
2. ✅ **领域知识注入优先**（AscendKernelGen）→ mctlass skill 第一优先
3. ✅ **失败案例沉淀**（AscendKernelGen 错误衍生监督）→ failure_cases.md

下一步：可以开始落地 `.opencode/` 骨架了。建议第一个 skill 就建 `mctlass-usage`。
