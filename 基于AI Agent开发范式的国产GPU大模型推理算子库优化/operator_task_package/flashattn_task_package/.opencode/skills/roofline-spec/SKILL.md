---
name: roofline-spec
description: Use when analyzing GPU kernel performance limits or judging whether a predicted speedup is physically feasible. Trigger keywords: roofline, bandwidth, memory-bound, compute-bound, gap, peak, 物理极限, 带宽利用率, 理论下限. MXC500 硬件规格与 Roofline 物理模型（创新点 A，替代缺失的 NCU profiler）。
---

# Roofline 物理模型与 MXC500 规格（创新点 A）

## 为什么用 Roofline（核心动机）
MXC500 **没有 per-kernel profiler**（实测 mx-smi 仅功耗监视器，无 NCU 那样的 occupancy/stall/SOL 指标）。
CudaForge/KernelAgent 的"NCU 硬件反馈闭环"在国产 GPU 上失效。
→ 用 **Roofline 物理上界**（只依赖架构 spec 两个常数，零 profiler 依赖）替代。

## MXC500 硬件规格（来源：沐曦官网）
- **显存**：64 GB HBM2e
- **峰值带宽**：1.8 TB/s = 1800 GB/s
- **BF16 算力**：280 TFLOPS
- **INT8**：560 TOPS
- **平衡点算术强度** = 280e12 / 1800e9 ≈ **155 FLOP/Byte**

## FlashAttention decode 的 Roofline 分析
- 算术强度 ≈ **2 FLOP/Byte**（远低于平衡点 155）→ **严格 memory-bound**
- 性能上限 = 带宽（不是算力）
- 官方 baseline 实测峰值仅 821 GB/s = **45.6% 带宽利用率** → 优化空间 ~2×

## 关键公式
```
T_lower_bound = max(FLOPs / peak_TFLOPS, Bytes / peak_BW)
gap_to_roofline = measured_time / T_lower_bound
  gap=1.0 → 已达物理极限
  gap=10  → 还有 10x 空间

bandwidth_utilization = achievable_bw / 1800
物理可行性：predicted_speedup ≤ 1 / baseline_utilization（roofline 上界 clip）
```

## Split-K 启发式
```
num_splits ≈ num_SMs / (batch * num_heads)
batch=1: split≈12（SM 严重欠载）
batch=16: split≈1（已饱和）
```

## 工具调用
```python
from agent_system.roofline_engine import analyze, gap_to_roofline, suggest_split_k
r = analyze(cfg)  # 返回 bound_type, t_lower_bound, arithmetic_intensity
```
