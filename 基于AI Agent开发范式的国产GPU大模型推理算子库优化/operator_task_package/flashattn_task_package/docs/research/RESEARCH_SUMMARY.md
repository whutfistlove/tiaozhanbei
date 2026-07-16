# 深度研究报告汇总（三份）

> 本文件汇总三个并行深度研究 Agent 的核心结论，作为方案设计的实证基础。
> 完整原始报告保存在同目录下三个子文件。

---

## 报告一：mctlass 能力深度调研

**结论**：mctlass 是 CUTLASS **2.x**（非 3.x）移植，版本 2.10.0，650 个头文件。

**FlashAttention decode 可用的完整原语链**（已逐字确认）：
```
arch::MacaMma<bf16, 16x16x16>           # 最底层 MMA 指令（4096 MAC/指令）
  ↑
warp::MacaMmaTensorOp + MacaConvertAndPack  # bf16 fragment 必须重排！
  ↑
threadblock::DefaultMmaSoftmaxMainloopFusion  # mainloop 融合 scale
  ↑
epilogue::EpilogueVisitorSoftmax<UseMasking>  # epilogue 融合 online softmax
  ↑
device::MacaGemmUniversal                      # 设备级启动
```

**关键陷阱**（必须写入 mctlass-usage skill）：
1. bf16 fragment 灌回需 `MacaConvertAndPack`（带 index 重排），否则数值错位
2. softmax 必须在 f32（ElementSoftmaxCompute=float）
3. 128-bit 对齐：bf16 至少 8 元素
4. C500 = `__MACA_ARCH__==1000`，f32 MMA `16x16x4f32` 仅 C500 有
5. 运行时用 `mc_runtime_api`（mcMalloc/mcStreamSynchronize），非 cuda_*
6. GitHub 无 example，参考 `frontend_op/` 封装

**对算子优化的意义**：Q@K^T（scale+softmax 融合）+ P@V 两次 GEMM 全部有现成 mctlass 原语，**满足"核心矩阵计算必须用 mctlass"的提交约束**。

---

## 报告二：FlashAttention decode 内存原理

**实测算术强度分析**：
- decode 每读一对 KV（2×d×2 bytes）算 ~4d FLOPs → 算术强度 ≈ 1 FLOP/byte
- C500 平衡点 = 280 TFLOPS / 1.8 TB/s ≈ 155 FLOP/byte（注：研究 Agent 用 19.5 TFLOPS 得 10.8，以沐曦官方 280 TFLOPS 为准）
- **1 << 155 → decode 严格 memory-bound**，性能 = 带宽

**官方 baseline 实测瓶颈**（来自 CSV）：
| 现象 | 数值 | 根因 |
|------|------|------|
| batch=1→8 耗时几乎不变 | 8192下 1.34→1.38ms | 仅 8 个 block，SM 严重欠载 |
| batch=1 带宽 | 50 GB/s | 串行扫描+非合并访存 |
| 峰值带宽 | 821 GB/s（45.6%） | paged 随机 block_table 破坏合并 |

**优化技术优先级**（按收益）：
1. ⭐⭐⭐⭐⭐ **Split-K（FlashDecoding）**：grid 从 (B,H)→(B,H,N_splits)，batch=1 可 50→300+ GB/s
2. ⭐⭐⭐⭐ **block 粒度合并加载 + 软件流水**（cp.async / num_stages>1）
3. ⭐⭐⭐ **Q 常驻 SRAM**（避免循环内重载）
4. ⭐⭐⭐ **统一 max value（FlashDecoding++）**：消除 reduce 同步（split≥8 时）

**目标**：带宽 45% → 70-80%（峰值 ~1300 GB/s），整体 1.7-2× 加速。

---

## 报告三：Agent 硬件反馈闭环与创新点（决定性）

### 决定性实测发现
⭐ **MXC500 没有 per-kernel profiler**：
- mx-smi 仅是功耗/温度/利用率监视器（类似 nvidia-smi），无 NCU 那样的 SOL/occupancy/stall 指标
- 唯一 profiling 手段 = torch.profiler（仅 kernel device_time 毫秒计时）
- **结论**：CudaForge/KernelAgent 的"NCU 硬件反馈"范式在国产 GPU 上直接失效

### 明确的学术空白（已检索确认）
⭐ **没有任何工作把"纯理论/经验 roofline 上界（不依赖 profiler）"作为结构化先验喂给 LLM agent**
- CudaForge/KernelAgent 依赖 NCU 实测 roofline
- Omniwise 用训练的模型预测，但要训练
- **空白**：用架构 spec 的两个常数（peak FLOPS + peak 带宽）或 benchmark 拟合的经验上界，零 profiler 依赖地给 LLM 物理约束

### 三个候选创新点（已确定 A 主 + B 辅 + C 增强）

**创新点 A（主）— Roofline-Anchored LLM World Model**
用 roofline 物理上界替代缺失的 NCU 反馈。三层：
1. 物理锚点层：MXC500 spec（280 TFLOPS, 1.8 TB/s）+ CSV 拟合经验上界（821 GB/s）
2. 静态强度层：LLM 从代码估算 FLOPs/bytes → 算术强度 → 定位 roofline 图
3. 信念锚定层：roofline 上界 clip LLM 性能预测，防幻觉（physics-informed）

**创新点 B（辅）— Co-Evolving Hardware Belief**
维护 `hardware_belief.md`，每轮 benchmark 抽象"配置→性能"规律，让 agent 自学陌生 MXC500 架构（in-context 架构知识蒸馏，替代对国产 GPU 的零先验）。

**创新点 C（增强）— 双层候选过滤**
roofline 物理过滤 + LLM 相对预测，把昂贵 GPU 评测从 O(N)→O(k)。

### 统一底层逻辑
> "针对国产 GPU 缺 profiler、LLM 无先验、benchmark 昂贵三大痛点，把物理可得的硬约束（roofline）+ LLM 推理能力，组合成不依赖成熟 profiler 的硬件反馈闭环"——NVIDIA 生态之外的独有命题。

### 新增参考文献（报告三发现）
- GPU Forecasters (arXiv:2605.31464)：LLM 作 selective 性能 surrogate
- Omniwise (arXiv:2506.20886)：3B 模型免 profiling 预测 arithmetic intensity
- K-Search (arXiv:2602.19128)：co-evolving world model，平均 2.10× / 最高 14.3×
- KernelAgent (PyTorch 官方博客)：NCU+roofline，最直接对照系（但强依赖 NCU）
