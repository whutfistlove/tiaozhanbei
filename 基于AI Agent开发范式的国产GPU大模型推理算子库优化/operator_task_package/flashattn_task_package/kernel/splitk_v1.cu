#include <cstdint>
#include <cmath>
#include "mctlass/bfloat16.h"
#include "mcr/mc_runtime.h"
typedef mctlass::bfloat16_t __nv_bfloat16;
using mctlass::bfloat16_t;

static constexpr int NUM_SPLITS = 8;

extern "C" __global__ void decode_kernel_splitk(
    const bfloat16_t* __restrict__ q,
    const bfloat16_t* __restrict__ k_cache,
    const bfloat16_t* __restrict__ v_cache,
    bfloat16_t* __restrict__ partial_o,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int64_t batch_size, int64_t seqlen_k, int64_t seqlen_q,
    int64_t num_heads, int64_t num_heads_k, int64_t headdim,
    int64_t page_block_size, int64_t blocks_per_batch)
{
    int split_idx = blockIdx.z;
    int b = blockIdx.x / num_heads;
    int h = blockIdx.x % num_heads;
    if (b >= batch_size) return;
    int tid = threadIdx.x;
    int seqlen = cache_seqlens[b];
    int kv_h = h * num_heads_k / num_heads;

    int tokens_per_split = (seqlen + NUM_SPLITS - 1) / NUM_SPLITS;
    int start_tok = split_idx * tokens_per_split;
    int end_tok = min(start_tok + tokens_per_split, seqlen);
    if (start_tok >= seqlen) return;

    float q_local = (tid < headdim)
        ? float(q[b * seqlen_q * num_heads * headdim + h * headdim + tid]) : 0.0f;

    float scale = 1.0f / sqrtf((float)headdim);
    float max_val = -1e30f, sum_exp = 0.0f, out_local = 0.0f;

    for (int t = start_tok; t < end_tok; ++t) {
        int page_idx = t / page_block_size;
        int page_off = t % page_block_size;
        int phys = block_table[b * blocks_per_batch + page_idx];
        int64_t kv_base = (int64_t)phys * page_block_size * num_heads_k * headdim
                        + page_off * num_heads_k * headdim + kv_h * headdim;

        float partial = (tid < headdim) ? q_local * float(k_cache[kv_base + tid]) : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, mask);
        float score = partial * scale;

        float new_max = fmaxf(max_val, score);
        float exp_old = expf(max_val - new_max);
        float exp_val = expf(score - new_max);
        sum_exp = sum_exp * exp_old + exp_val;
        max_val = new_max;
        if (tid < headdim) out_local = out_local * exp_old + exp_val * float(v_cache[kv_base + tid]);
    }

    int batch_head_idx = b * num_heads + h;
    int flat_idx = batch_head_idx * NUM_SPLITS + split_idx;
    if (tid < headdim) partial_o[flat_idx * headdim + tid] = bfloat16_t(out_local);
    if (tid == 0) {
        partial_max[flat_idx] = max_val;
        partial_sum[flat_idx] = sum_exp;
    }
}

extern "C" __global__ void reduce_kernel_splitk(
    const bfloat16_t* __restrict__ partial_o,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    bfloat16_t* __restrict__ output,
    int64_t batch_size, int64_t num_heads, int64_t headdim)
{
    int b = blockIdx.x / num_heads;
    int h = blockIdx.x % num_heads;
    if (b >= batch_size) return;
    int tid = threadIdx.x;

    int batch_head_idx = b * num_heads + h;
    float global_max = -1e30f;
    float global_sum = 0.0f;

    for (int s = 0; s < NUM_SPLITS; ++s) {
        int flat_idx = batch_head_idx * NUM_SPLITS + s;
        float m = partial_max[flat_idx];
        global_max = fmaxf(global_max, m);
    }
    for (int s = 0; s < NUM_SPLITS; ++s) {
        int flat_idx = batch_head_idx * NUM_SPLITS + s;
        float m = partial_max[flat_idx];
        float l = partial_sum[flat_idx];
        global_sum += l * expf(m - global_max);
    }
    if (tid < headdim) {
        float out_local = 0.0f;
        for (int s = 0; s < NUM_SPLITS; ++s) {
            int flat_idx = batch_head_idx * NUM_SPLITS + s;
            float m = partial_max[flat_idx];
            float reweight = expf(m - global_max);
            float val = float(partial_o[flat_idx * headdim + tid]) * reweight;
            out_local += val;
        }
        int out_idx = b * num_heads * headdim + h * headdim + tid;
        output[out_idx] = (global_sum > 0) ? bfloat16_t(out_local / global_sum) : bfloat16_t(0.0f);
    }
}

static char* d_partial_o = nullptr;
static char* d_partial_max = nullptr;
static char* d_partial_sum = nullptr;
static int64_t last_batch = 0;
static int64_t last_heads = 0;

extern "C" void run_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache_paged, const __nv_bfloat16* v_cache_paged,
    __nv_bfloat16* output, const int32_t* cache_seqlens, const int32_t* block_table,
    int64_t batch_size, int64_t seqlen_k, int64_t seqlen_q,
    int64_t num_heads, int64_t num_heads_k, int64_t headdim,
    int64_t page_block_size, int64_t num_blocks, int64_t causal)
{
    int64_t blocks_per_batch = num_blocks / batch_size;
    int64_t need_partial_o = batch_size * num_heads * NUM_SPLITS * headdim * sizeof(bfloat16_t);
    int64_t need_partial_m = batch_size * num_heads * NUM_SPLITS * sizeof(float);
    int64_t need_partial_l = batch_size * num_heads * NUM_SPLITS * sizeof(float);

    if (batch_size != last_batch || num_heads != last_heads) {
        if (d_partial_o) { mcFree(d_partial_o); }
        if (d_partial_max) { mcFree(d_partial_max); }
        if (d_partial_sum) { mcFree(d_partial_sum); }
        mcMalloc(&d_partial_o, need_partial_o);
        mcMalloc(&d_partial_max, need_partial_m);
        mcMalloc(&d_partial_sum, need_partial_l);
        last_batch = batch_size;
        last_heads = num_heads;
    }

    dim3 grid(batch_size * num_heads, 1, NUM_SPLITS);
    decode_kernel_splitk<<<grid, 32>>>(
        q, k_cache_paged, v_cache_paged,
        (bfloat16_t*)d_partial_o, (float*)d_partial_max, (float*)d_partial_sum,
        cache_seqlens, block_table,
        batch_size, seqlen_k, seqlen_q, num_heads, num_heads_k, headdim,
        page_block_size, blocks_per_batch);

    reduce_kernel_splitk<<<dim3(batch_size * num_heads), 32>>>(
        (bfloat16_t*)d_partial_o, (float*)d_partial_max, (float*)d_partial_sum,
        output, batch_size, num_heads, headdim);
}