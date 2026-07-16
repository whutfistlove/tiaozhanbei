#include <cstdint>
#include <cmath>
#include "mctlass/bfloat16.h"
#include "mcr/mc_runtime.h"
typedef mctlass::bfloat16_t __nv_bfloat16;
using mctlass::bfloat16_t;

static constexpr int BLOCK_DIM = 128;
static constexpr int NUM_SPLITS = 12;
static constexpr int MAX_SEQ_KV = 16384;

extern "C" __global__ void splitk_compute_kernel(
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
    int64_t page_block_size, int64_t blocks_per_batch,
    int num_splits)
{
    extern __shared__ char smem_base[];
    float* shared_q = reinterpret_cast<float*>(smem_base);
    float* shared_s = reinterpret_cast<float*>(smem_base + headdim * sizeof(float));

    int split_idx = blockIdx.y;
    int b = blockIdx.x / num_heads;
    int h = blockIdx.x % num_heads;
    if (b >= batch_size) return;
    int tid = threadIdx.x;

    if (tid < headdim) {
        int q_idx = b * seqlen_q * num_heads * headdim + h * headdim + tid;
        shared_q[tid] = float(q[q_idx]);
    }
    __syncthreads();

    if (tid >= headdim) return;

    int seqlen = cache_seqlens[b];
    int kv_h = h * num_heads_k / num_heads;

    int tokens_per_split = (seqlen + num_splits - 1) / num_splits;
    int start_tok = split_idx * tokens_per_split;
    int end_tok = start_tok + tokens_per_split;
    if (end_tok > seqlen) end_tok = seqlen;
    if (start_tok >= seqlen) return;

    float scale = 1.0f / sqrtf((float)headdim);
    float max_val = -1e30f;
    float sum_exp = 0.0f;
    float out_local = 0.0f;

    for (int t = start_tok; t < end_tok; ++t) {
        int page_idx = t / page_block_size;
        int page_off = t % page_block_size;
        int phys = block_table[b * blocks_per_batch + page_idx];
        int64_t kv_base = (int64_t)phys * page_block_size * num_heads_k * headdim
                        + page_off * num_heads_k * headdim + kv_h * headdim;

        float dot = 0.0f;
        for (int d = tid; d < headdim; d += BLOCK_DIM) {
            dot += shared_q[d] * float(k_cache[kv_base + d]);
        }
        shared_s[tid] = dot;
        __syncthreads();

        for (int active = BLOCK_DIM >> 1; active >= 32; active >>= 1) {
            if (tid < active) shared_s[tid] += shared_s[tid + active];
            __syncthreads();
        }
        if (tid < 32) {
            float val = shared_s[tid];
            for (int mask = 16; mask > 0; mask >>= 1) {
                val += __shfl_xor_sync(0xffffffff, val, mask);
            }
            shared_s[tid] = val;
        }
        __syncthreads();

        // 所有线程从 shared memory 读归约结果（确保一致）
        float score = shared_s[0] * scale;

        float new_max = fmaxf(max_val, score);
        float exp_old = expf(max_val - new_max);
        float exp_val = expf(score - new_max);
        sum_exp = sum_exp * exp_old + exp_val;
        max_val = new_max;

        if (tid < headdim) {
            float v_val = float(v_cache[kv_base + tid]);
            out_local = out_local * exp_old + exp_val * v_val;
        }
    }

    int batch_head_idx = b * num_heads + h;
    int flat_idx = batch_head_idx * num_splits + split_idx;
    if (tid < headdim) {
        partial_o[flat_idx * headdim + tid] = bfloat16_t(out_local);
    }
    if (tid == 0) {
        partial_max[flat_idx] = max_val;
        partial_sum[flat_idx] = sum_exp;
    }
}

extern "C" __global__ void splitk_reduce_kernel(
    const bfloat16_t* __restrict__ partial_o,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    bfloat16_t* __restrict__ output,
    int64_t batch_size, int64_t num_heads, int64_t headdim,
    int num_splits)
{
    int b = blockIdx.x / num_heads;
    int h = blockIdx.x % num_heads;
    if (b >= batch_size) return;
    int tid = threadIdx.x;

    int batch_head_idx = b * num_heads + h;
    float global_max = -1e30f;
    float global_sum = 0.0f;

    for (int s = 0; s < num_splits; ++s) {
        int flat_idx = batch_head_idx * num_splits + s;
        float m = partial_max[flat_idx];
        global_max = fmaxf(global_max, m);
    }
    for (int s = 0; s < num_splits; ++s) {
        int flat_idx = batch_head_idx * num_splits + s;
        float m = partial_max[flat_idx];
        float l = partial_sum[flat_idx];
        global_sum += l * expf(m - global_max);
    }

    if (tid < headdim) {
        float out_local = 0.0f;
        for (int s = 0; s < num_splits; ++s) {
            int flat_idx = batch_head_idx * num_splits + s;
            float m = partial_max[flat_idx];
            float reweight = expf(m - global_max);
            float val = float(partial_o[flat_idx * headdim + tid]) * reweight;
            out_local += val;
        }
        int out_idx = b * num_heads * headdim + h * headdim + tid;
        output[out_idx] = (global_sum > 0.0f) ? bfloat16_t(out_local / global_sum) : bfloat16_t(0.0f);
    }
}

static bfloat16_t* d_partial_o = nullptr;
static float* d_partial_max = nullptr;
static float* d_partial_sum = nullptr;
static int64_t last_batch = 0;
static int64_t last_heads = 0;
static int last_num_splits = 0;

extern "C" void run_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache_paged, const __nv_bfloat16* v_cache_paged,
    __nv_bfloat16* output, const int32_t* cache_seqlens, const int32_t* block_table,
    int64_t batch_size, int64_t seqlen_k, int64_t seqlen_q,
    int64_t num_heads, int64_t num_heads_k, int64_t headdim,
    int64_t page_block_size, int64_t num_blocks, int64_t causal)
{
    int num_splits = (batch_size * num_heads >= 96) ? 1 : std::max(1, std::min(static_cast<int>(96 / (batch_size * num_heads)), NUM_SPLITS));
    int64_t blocks_per_batch = num_blocks / batch_size;

    int64_t need_partial_o = batch_size * num_heads * num_splits * headdim * sizeof(bfloat16_t);
    int64_t need_partial_m = batch_size * num_heads * num_splits * sizeof(float);
    int64_t need_partial_l = batch_size * num_heads * num_splits * sizeof(float);

    if (batch_size != last_batch || num_heads != last_heads || num_splits != last_num_splits) {
        if (d_partial_o) mcFree(d_partial_o);
        if (d_partial_max) mcFree(d_partial_max);
        if (d_partial_sum) mcFree(d_partial_sum);
        mcMalloc((void**)&d_partial_o, need_partial_o);
        mcMalloc((void**)&d_partial_max, need_partial_m);
        mcMalloc((void**)&d_partial_sum, need_partial_l);
        last_batch = batch_size;
        last_heads = num_heads;
        last_num_splits = num_splits;
    }

    dim3 grid(batch_size * num_heads, num_splits);
    int smem_size = headdim * sizeof(float) * 2;

    splitk_compute_kernel<<<grid, BLOCK_DIM, smem_size>>>(
        q, k_cache_paged, v_cache_paged,
        d_partial_o, d_partial_max, d_partial_sum,
        cache_seqlens, block_table,
        batch_size, seqlen_k, seqlen_q, num_heads, num_heads_k, headdim,
        page_block_size, blocks_per_batch, num_splits);

    splitk_reduce_kernel<<<dim3(batch_size * num_heads), BLOCK_DIM>>>(
        d_partial_o, d_partial_max, d_partial_sum,
        output, batch_size, num_heads, headdim, num_splits);
}