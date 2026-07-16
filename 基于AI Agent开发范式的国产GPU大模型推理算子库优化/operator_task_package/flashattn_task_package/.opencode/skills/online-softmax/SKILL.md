---
name: online-softmax
description: Use when implementing online softmax (flash-style incremental softmax) in attention kernels. Trigger keywords: online softmax, flash softmax, incremental, rescale, row max, row sum, m_i, l_i. Online softmax 的数学原理与正确实现（避免 NaN/数值错误）。
---

# Online Softmax 正确实现

## 数学原理（FlashAttention 核心）
softmax 的归一化统计量（row max m、row sum l）可以增量式分块累加，无需先看整行。

## 递推公式（处理第 j 个新 K 块时）
```
m_new = max(m_old, max(scores_j))
l_new = l_old * exp(m_old - m_new) + Σ exp(scores_j - m_new)
o_new = o_old * exp(m_old - m_new) + Σ exp(scores_j - m_new) * V_j
```

## Split-K 合并公式（reduce kernel）
```
global_max = max(m_1, ..., m_N)
global_sum = Σ l_i * exp(m_i - global_max)
output = Σ o_i * exp(m_i - global_max) / global_sum
```

## ⚠️ 已知陷阱（从真实失败案例提炼）
1. **score 必须广播到所有线程**：block 级归约后，只有 tid==0 有 score，其它线程 score=0 导致 softmax 状态不一致。修复：所有线程从 shared memory 读归约结果
2. **初始 max 用 -1e30 而非 -inf**：避免 (-inf)-(-inf)=NaN
3. **partial_o 不要预归一化**：compute kernel 写的 partial_o 应是未除 sum 的累加值，由 reduce kernel 统一归一化
4. **sum_exp=0 时输出 0**：空 split（start_tok>=seqlen）要处理

## 实现模板（CUDA）
```cpp
float m_i = -1e30f, l_i = 0.0f, acc = 0.0f;
for (each token t in split) {
    float score = dot(q, k[t]) * scale;
    float m_new = fmaxf(m_i, score);
    float alpha = expf(m_i - m_new);  // 重缩放历史
    float p = expf(score - m_new);
    l_i = l_i * alpha + p;
    acc = acc * alpha + p * v[t];     // 每个 thread 累加自己负责的维度
    m_i = m_new;
}
// 写 partial: o=acc, m=m_i, l=l_i（不要除 l_i！）
```
