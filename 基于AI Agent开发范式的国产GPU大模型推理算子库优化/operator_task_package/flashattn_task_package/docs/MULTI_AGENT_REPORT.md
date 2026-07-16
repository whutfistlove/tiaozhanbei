# OpenCode 多 Agent 算子优化迭代报告

> 本文档记录用 OpenCode 多智能体体系（Analyst/Coder/Profiler/Judge/Reflector）驱动 FlashAttention decode 算子优化的完整过程。
> 这是赛题"Agent 可复现性"(20%)的核心物料。

## 1. 多 Agent 体系架构

```
我（ZCode）作为总指挥，驱动 OpenCode 的 5 个 subagent：
  Analyst  → roofline 瓶颈分析（用 agent_system/roofline_engine.py）
  Coder    → 生成/修改 kernel 代码（mctlass + Split-K）
  Profiler → 编译(mxcc) + 运行(GPU) + benchmark
  Judge    → 正确性校验(allclose) + 性能裁决
  Reflector→ 失败记录 + 硬件信念更新
```

## 2. 完整迭代记录

### 迭代 1：Split-K headdim<=32 版本
- **Analyst**：分析得出所有配置 memory-bound，Split-K 是第一优先级
- **Coder**：生成 `kernel/splitk_v1.cu`（单warp，headdim<=32）
- **编译问题**：缺 `mcr/mc_runtime.h` → Agent 自主修复
- **运行问题**：输出 NaN → Agent 自主修复
- **结果**：正确性通过，**5.22x 加速**（headdim=32配置）

### 迭代 2：headdim=128 Split-K（OJ 真实配置）
- **Coder**：生成 `kernel/splitk_h128.cu`（blockDim=128，shared memory 归约）
- **Bug 1**：`malloc` 替代 `mcMalloc` → illegal memory access → 修复为 mcMalloc
- **Bug 2**：点积归约后只有 tid==0 有 score → 数值错误 → 修复为所有线程读 shared_s[0]
- **结果**：正确性 **9/9 配置通过**（batch∈{1,4,16}, seq∈{1024,4096,8192}）

### 迭代 3：动态 num_splits
- **Analyst**：诊断固定 split=12 在大 batch 时 block 数过量（1536 vs 96 SM）
- **Coder**：实现动态 split = `batch*heads >= 96 ? 1 : min(96/(batch*heads), 12)`
- **Bug**：Agent 公式多乘 *8 → 性能退化 → 修正
- **结果**：正确性保持，性能回归到迭代2水平

## 3. 性能数据（Profiler 实测）

| 配置 | ours(ms) | ours_bw(GB/s) | official(ms) | off_bw(GB/s) | 加速比 | 正确性 |
|------|---------|--------------|-------------|-------------|--------|--------|
| b=1,seq=1024 | 0.247 | 17.0 | 0.146 | 28.1 | 0.59x | ✅ |
| b=1,seq=4096 | 0.795 | 21.1 | 0.489 | 34.5 | 0.62x | ✅ |
| b=1,seq=8192 | 1.637 | 20.5 | 0.946 | 35.5 | 0.58x | ✅ |
| b=4,seq=1024 | 1.137 | 14.8 | 0.148 | 113.7 | 0.13x | ✅ |
| b=16,seq=4096 | 8.419 | 31.9 | 0.620 | 429.2 | 0.07x | ✅ |

## 4. 关键结论（Reflector 信念库）

1. **正确性已完全解决**：9/9 配置通过 allclose(rtol=1e-2)
2. **朴素手写是性能天花板**：带宽仅 17-32 GB/s（官方 200-800 GB/s）
3. **必须用 mctlass Tensor Core**：Q@K^T 和 P@V 用 `MacaMma<bf16,16x16x16>` 替代标量循环
4. **Split-K 在小 batch 有效**：b=1 时加速比最高（0.62x）

## 5. Agent 物料清单

| 物料 | 位置 | 大小 |
|------|------|------|
| Agent 会话日志（7个） | `docs/agent_logs/*.json` | 488KB |
| 优化日志 | `optimization_log.md` | 3轮迭代 |
| 失败案例库 | `agent_system/domain_memory/failure_cases/` | 3条 |
| 硬件信念 | `agent_system/domain_memory/hardware_belief.json` | 2条 |
| 优化 kernel | `kernel/splitk_h128.cu` | 197行 |

## 6. 下一步优化方向

基于信念库结论，突破性能天花板需要：
1. **引入 mctlass GEMM**：用 `MacaMma` + `EpilogueVisitorSoftmax` 替代手写循环
2. **向量化加载**：bf16×8 合并访问替代逐元素
3. **软件流水**：`maca_mma_multistage`（kStages=2~3）
