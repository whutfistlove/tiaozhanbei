#include <cstdint>
#include <cmath>
#include "mctlass/bfloat16.h"
typedef mctlass::bfloat16_t __nv_bfloat16;
using mctlass::bfloat16_t;

// 正确的 attention decode：单 warp 处理一个 (batch,head)，headdim<=32
extern "C" __global__ void decode_kernel(
    const bfloat16_t* __restrict__ q,
    const bfloat16_t* __restrict__ k_cache,
    const bfloat16_t* __restrict__ v_cache,
    bfloat16_t* __restrict__ output,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int64_t batch_size, int64_t seqlen_k, int64_t seqlen_q,
    int64_t num_heads, int64_t num_heads_k, int64_t headdim,
    int64_t page_block_size, int64_t blocks_per_batch)
{
    int b = blockIdx.x / num_heads;
    int h = blockIdx.x % num_heads;
    if (b >= batch_size) return;
    int tid = threadIdx.x;  // 0..31
    int seqlen = cache_seqlens[b];
    int kv_h = h * num_heads_k / num_heads;

    // headdim<=32: tid 直接对应维度 idx
    float q_local = (tid < headdim)
        ? float(q[b * seqlen_q * num_heads * headdim + h * headdim + tid]) : 0.0f;

    float scale = 1.0f / sqrtf((float)headdim);
    float max_val = -1e30f, sum_exp = 0.0f, out_local = 0.0f;

    for (int t = 0; t < seqlen; ++t) {
        int page_idx = t / page_block_size;
        int page_off = t % page_block_size;
        int phys = block_table[b * blocks_per_batch + page_idx];
        int64_t kv_base = (int64_t)phys * page_block_size * num_heads_k * headdim
                        + page_off * num_heads_k * headdim + kv_h * headdim;

        float partial = (tid < headdim) ? q_local * float(k_cache[kv_base + tid]) : 0.0f;
        // warp shuffle 归约（32 threads → 完整点积，因为 headdim<=32）
        for (int mask = 16; mask > 0; mask >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, mask);
        float score = partial * scale;

        float new_max = fmaxf(max_val, score);
        float exp_old = expf(max_val - new_max);
        float exp_val = expf(score - new_max);
        sum_exp = sum_exp * exp_old + exp_val;
        max_val = new_max;
        if (tid < headdim) out_local = out_local * exp_old + exp_val * float(v_cache[kv_base + tid]);
    }
    if (tid < headdim && sum_exp > 0)
        output[b * seqlen_q * num_heads * headdim + h * headdim + tid] = bfloat16_t(out_local / sum_exp);
}

extern "C" void run_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache_paged, const __nv_bfloat16* v_cache_paged,
    __nv_bfloat16* output, const int32_t* cache_seqlens, const int32_t* block_table,
    int64_t batch_size, int64_t seqlen_k, int64_t seqlen_q,
    int64_t num_heads, int64_t num_heads_k, int64_t headdim,
    int64_t page_block_size, int64_t num_blocks, int64_t causal)
{
    int64_t blocks_per_batch = num_blocks / batch_size;
    dim3 grid(batch_size * num_heads);
    decode_kernel<<<grid, 32>>>(  // 单 warp
        q, k_cache_paged, v_cache_paged, output,
        cache_seqlens, block_table,
        batch_size, seqlen_k, seqlen_q, num_heads, num_heads_k, headdim,
        page_block_size, blocks_per_batch);
}
