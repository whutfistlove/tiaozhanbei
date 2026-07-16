"""
kernel_compiler + kernel_loader 的真实集成测试。

验证完整链路：.cu 源码 → mxcc 编译 → .so → ctypes 加载 → GPU 运行 → 正确性校验。

这是从"原型"到"真实实现"的关键验证。
"""
import os
import pytest
import torch
from pathlib import Path

from agent_system.kernel_compiler import (
    compile_source, compile_file, find_mxcc, classify_error, CompileResult,
)
from agent_system.kernel_loader import load_kernel, call_run_kernel, make_output_tensor
from agent_system.correctness import make_test_inputs, generate_reference, check
from agent_system.roofline_engine import KernelConfig


# 一个已知能编译的简单 kernel（从 mxcc 调试验证得来）
SIMPLE_KERNEL_SRC = r"""#include <cstdint>
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
"""


@pytest.fixture
def workdir(tmp_path):
    return tmp_path


class TestCompiler:
    def test_mxcc_available(self):
        """mxcc 必须可用"""
        mxcc = find_mxcc()
        assert mxcc is not None, "mxcc 未找到，MACA 工具链未配置"

    def test_compile_success(self, workdir):
        """真实编译简单 kernel 为 .so"""
        so_path = str(workdir / "test_kernel.so")
        result = compile_source(SIMPLE_KERNEL_SRC, so_path)
        assert result.success, f"编译失败: {result.error_msg}"
        assert os.path.exists(so_path)
        assert result.compile_time_s > 0

    def test_compile_bad_source_fails(self, workdir):
        """语法错误应编译失败并返回错误信息"""
        bad_src = 'extern "C" void f() { this is not valid }'
        result = compile_source(bad_src, str(workdir / "bad.so"))
        assert not result.success
        assert result.error_msg  # 有错误信息

    def test_compile_file_not_found(self, workdir):
        result = compile_file("/nonexistent/file.cu", str(workdir / "x.so"))
        assert not result.success
        assert "不存在" in result.error_msg

    def test_classify_error(self):
        assert classify_error("no member named 'foo'") == "api_misuse"
        assert classify_error("incomplete type '__maca_bfloat16'") == "type_error"
        assert classify_error("expected ';' after") == "syntax_error"
        assert classify_error("undefined reference to `foo`") == "link_error"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
class TestLoaderAndRun:
    """真实 GPU 加载与运行（核心集成测试）。"""

    def test_load_and_run_correctness(self, workdir):
        """完整链路：编译→加载→运行→正确性校验"""
        cfg = KernelConfig(batch_size=1, seqlen_kv=16, headdim=4, num_heads=1)
        # 编译
        so_path = str(workdir / "kernel.so")
        cres = compile_source(SIMPLE_KERNEL_SRC, so_path)
        assert cres.success, f"编译失败: {cres.error_msg}"
        # 加载
        lres = load_kernel(so_path)
        assert lres.success, f"加载失败: {lres.error_msg}"
        # 生成输入（num_blocks 用默认自动计算，确保足够大）
        q, k, v, lens, bt = make_test_inputs(cfg, device="cuda", seed=42)
        num_blocks = k.shape[0]
        output = make_output_tensor(cfg, device="cuda")
        # 运行
        out = call_run_kernel(lres.run_kernel_fn, q, k, v, output, lens, bt, cfg, num_blocks)
        # 参考输出
        ref = generate_reference(q, k, v, lens, bt, cfg)
        # 正确性校验
        result = check(out, ref, rtol=1e-2, atol=1e-2)
        assert result.passed, f"正确性失败: {result.detail}"

    def test_load_nonexistent_so(self):
        result = load_kernel("/nonexistent/kernel.so")
        assert not result.success

    def test_output_shape(self, workdir):
        """输出 shape 正确"""
        cfg = KernelConfig(batch_size=2, seqlen_kv=16, headdim=4, num_heads=1)
        so_path = str(workdir / "kernel.so")
        assert compile_source(SIMPLE_KERNEL_SRC, so_path).success
        lres = load_kernel(so_path)
        assert lres.success
        q, k, v, lens, bt = make_test_inputs(cfg, device="cuda", seed=1)
        num_blocks = k.shape[0]
        output = make_output_tensor(cfg, device="cuda")
        out = call_run_kernel(lres.run_kernel_fn, q, k, v, output, lens, bt, cfg, num_blocks)
        assert out.shape == (2, 1, 1, 4)

    def test_deterministic_run(self, workdir):
        """相同输入两次运行结果一致"""
        cfg = KernelConfig(batch_size=1, seqlen_kv=16, headdim=4, num_heads=1)
        so_path = str(workdir / "kernel.so")
        compile_source(SIMPLE_KERNEL_SRC, so_path)
        lres = load_kernel(so_path)
        q, k, v, lens, bt = make_test_inputs(cfg, device="cuda", seed=42)
        num_blocks = k.shape[0]
        out1 = make_output_tensor(cfg, "cuda")
        out2 = make_output_tensor(cfg, "cuda")
        call_run_kernel(lres.run_kernel_fn, q, k, v, out1, lens, bt, cfg, num_blocks)
        call_run_kernel(lres.run_kernel_fn, q, k, v, out2, lens, bt, cfg, num_blocks)
        assert torch.allclose(out1.float(), out2.float(), atol=1e-6)
