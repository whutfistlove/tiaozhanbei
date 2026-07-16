---
name: paged-addressing
description: Use when implementing paged KV-cache addressing in attention kernels. Trigger keywords: paged, block_table, page_idx, page_offset, physical block, blocks_per_batch, num_blocks, 寻址. Paged KV-Cache 的正确寻址公式与边界处理。
---

# Paged KV-Cache 寻址

## 布局
```
k_cache_paged: (num_blocks, page_block_size, num_heads_k, headdim) bf16
block_table:   (batch_size, blocks_per_batch) int32，blocks_per_batch = num_blocks / batch_size
```

## 寻址公式（第 b 个 batch 的第 t 个 KV token）
```cpp
int page_idx = t / page_block_size;       // 逻辑页号
int page_off = t % page_block_size;       // 页内偏移
int phys = block_table[b * blocks_per_batch + page_idx];  // 物理页号
int64_t kv_base = (int64_t)phys * page_block_size * num_heads_k * headdim
                + page_off * num_heads_k * headdim
                + kv_head * headdim;
// k_cache[kv_base + d] 就是第 t 个 token 的第 d 维 key
```

## ⚠️ 边界处理（从真实失败提炼）
1. **phys 可能很大**：block_table 是 randperm，phys 可达 num_blocks-1。**num_blocks 必须 ≥1024**，否则 kv_base 越界 → illegal memory access
2. **blocks_per_batch 计算**：`num_blocks / batch_size`，不是 `num_blocks / batch_size / page_block_size`
3. **cache_seqlens**：实际 KV 长度，可能 < seqlen_k（尾部 page 不满）

## GQA head 映射
```cpp
int groups = max(1, num_heads / num_heads_k);
int kv_head = h / groups;
```
本题 H=HK=8，退化为 MHA（kv_head = h）。

## 验证用 Python
```python
from agent_system.correctness import make_test_inputs
q, k, v, lens, bt = make_test_inputs(cfg, device='cuda')
# bt.shape = (batch, num_blocks//batch)
# k.shape = (num_blocks, 16, 8, 128)
```
