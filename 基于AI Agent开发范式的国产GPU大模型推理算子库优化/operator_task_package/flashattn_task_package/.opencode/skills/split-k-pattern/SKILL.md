---
name: split-k-pattern
description: Use when implementing Split-K (Flash-Decoding) for attention decode kernels to increase parallelism when batch is small. Trigger keywords: split-k, split-kv, FlashDecoding, parallelism, SM utilization, 并行度, 分裂. FlashAttention decode 的 Split-K 并行实现模板。
---

# Split-K (Flash-Decoding) 实现模板

## 适用条件（必读）
- seqlen_q=1（decode 阶段）
- batch * num_heads < num_SMs（GPU 欠载）
- **本题 batch=1,4,16 全部适用**（8 heads → 8/32/128 blocks，C500 有 ~96 SM）

## 核心思想
seqlen_q=1 时 Q 维无并行，沿 **KV 序列维度**增加第二并行轴：
```
grid: (batch, heads) → (batch, heads, num_splits)
每个 split block 处理 seqlen_kv/num_splits 个 token
产出 partial (o_i, m_i, l_i) → reduce kernel 合并
```

## Online Softmax 合并公式（reduce kernel）
处理 N 个 split 的 partial 结果：
```
global_max = max(m_1, m_2, ..., m_N)
global_sum = Σ l_i * exp(m_i - global_max)
global_out = Σ o_i * exp(m_i - global_max) / global_sum
```

## num_splits 选择（启发式）
```python
from agent_system.roofline_engine import suggest_split_k
split = suggest_split_k(cfg, num_sms=96)
# batch=1,seq=8192 → split≈12
# batch=16,seq=1024 → split=1（已饱和）
```
- 太少：SM 欠载（本题现状，batch=1 仅 50 GB/s）
- 太多：reduce 开销 + partial HBM 流量主导
- 经验：split 数让总 block 数略大于 SM 数

## mctlass 实现
```
threadblock::MacaMmaSplitKParallel  # 每 split 独立 MMA
reduction::ReduceSplitK             # 合并 kernel
```
或手写：每个 split block 用 EpilogueVisitorSoftmax 输出 (o_i, m_i, l_i)，第二个 kernel 合并。

## 内存布局
- partial output: (batch, heads, num_splits, headdim) fp32
- partial m, l: (batch, heads, num_splits) fp32
- reduce 后写回 (batch, 1, heads, headdim) bf16

## 已知问题（写入 failure_cases 规避）
- split 数不能整除 seq_kv 时，最后一段要 mask 尾部
- partial 结果的 HBM 中转会增加流量（split 越多流量越大）
- FlashDecoding++ 用"统一 max value"消除 reduce 同步（split≥8 时考虑）

## 预期收益
- batch=1: 50 GB/s → 300+ GB/s（最大单项收益）
- batch=4: 显著
- batch=16: 边际（已接近饱和）
