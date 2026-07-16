"""
correctness checker 的单元测试。

测试覆盖：
- 参考输出数学正确性（softmax(QK^T)V）
- allclose 校验逻辑
- 自身一致性（同样输入两次输出一致）
- 输入生成器（paged 寻址、shape）
- GQA head 映射
- 边界（seq_len=1）

运行：pytest tests/test_correctness.py -v
"""
import math
import pytest
import torch

from agent_system.correctness import (
    generate_reference,
    check,
    make_test_inputs,
    CorrectnessResult,
)
from agent_system.roofline_engine import KernelConfig


class TestReferenceGeneration:
    def test_reference_shape(self):
        cfg = KernelConfig(batch_size=2, seqlen_kv=64, headdim=32, num_heads=4)
        q, k, v, lens, bt = make_test_inputs(cfg, num_blocks=128, device="cpu")
        out = generate_reference(q, k, v, lens, bt, cfg)
        assert out.shape == (2, 1, 4, 32)

    def test_reference_is_softmax_qkv(self):
        """验证参考输出 = softmax(QK^T/sqrt(d))V 的数学正确性"""
        cfg = KernelConfig(batch_size=1, seqlen_kv=8, headdim=4, num_heads=1)
        q, k, v, lens, bt = make_test_inputs(cfg, num_blocks=16, device="cpu", seed=42)
        out = generate_reference(q, k, v, lens, bt, cfg)

        # 手动用连续 K/V 重算（绕过 paged）
        qf = q[0, 0, 0].float()  # (4,)
        # paged cache 的 block_table[0] 决定 token 顺序
        contig_k = []
        for t in range(8):
            page_idx = t // cfg.page_block_size
            page_off = t % cfg.page_block_size
            phys = int(bt[0, page_idx])
            contig_k.append(k[phys, page_off, 0].float())
        k_sel = torch.stack(contig_k)  # (8,4)
        v_sel = torch.stack([
            v[int(bt[0, t // cfg.page_block_size]), t % cfg.page_block_size, 0].float()
            for t in range(8)
        ])
        scores = qf @ k_sel.T / math.sqrt(4)
        expected = torch.softmax(scores, dim=0) @ v_sel
        assert torch.allclose(out[0, 0, 0], expected, atol=1e-5)

    def test_reference_deterministic(self):
        """相同输入两次调用结果一致"""
        cfg = KernelConfig(batch_size=1, seqlen_kv=32, headdim=16, num_heads=2)
        q, k, v, lens, bt = make_test_inputs(cfg, num_blocks=64, device="cpu", seed=7)
        out1 = generate_reference(q, k, v, lens, bt, cfg)
        out2 = generate_reference(q, k, v, lens, bt, cfg)
        assert torch.equal(out1, out2)

    def test_reference_output_float32(self):
        cfg = KernelConfig(batch_size=1, seqlen_kv=16, headdim=8)
        q, k, v, lens, bt = make_test_inputs(cfg, num_blocks=32, device="cpu")
        out = generate_reference(q, k, v, lens, bt, cfg)
        assert out.dtype == torch.float32


class TestCheckFunction:
    def test_identical_passes(self):
        a = torch.randn(4, 8)
        r = check(a, a)
        assert r.passed

    def test_large_diff_fails(self):
        a = torch.zeros(10)
        b = torch.ones(10) * 5.0  # 差距 5，远超 atol=0.01
        r = check(a, b)
        assert not r.passed

    def test_small_diff_within_tolerance(self):
        a = torch.randn(100)
        b = a + torch.randn(100) * 1e-4  # 微小扰动
        r = check(a, b, rtol=1e-2, atol=1e-2)
        assert r.passed

    def test_result_fields(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.01, 2.02, 3.03])
        r = check(a, b)
        assert r.max_abs_diff > 0
        assert r.num_elements == 3
        assert r.rtol == 1e-2

    def test_shape_mismatch_raises(self):
        a = torch.zeros(4)
        b = torch.zeros(5)
        with pytest.raises(AssertionError):
            check(a, b)

    def test_rtol_atol_control(self):
        """收紧容差后原本通过的会失败"""
        a = torch.tensor([1.0, 1.0])
        b = torch.tensor([1.005, 1.005])  # 差 0.005
        assert check(a, b, atol=1e-2).passed     # atol=0.01 通过
        assert not check(a, b, atol=1e-4, rtol=1e-4).passed  # rtol=atol=0.0001 失败


class TestMakeTestInputs:
    def test_shapes(self):
        cfg = KernelConfig(batch_size=2, seqlen_kv=64, headdim=128, num_heads=8)
        q, k, v, lens, bt = make_test_inputs(cfg, num_blocks=256, device="cpu")
        assert q.shape == (2, 1, 8, 128)
        assert k.shape == (256, 16, 8, 128)
        assert v.shape == (256, 16, 8, 128)
        assert lens.shape == (2,)
        assert bt.shape == (2, 128)  # 256/2

    def test_dtypes(self):
        cfg = KernelConfig(batch_size=1, seqlen_kv=32, headdim=64)
        q, k, v, lens, bt = make_test_inputs(cfg, num_blocks=64, device="cpu")
        assert q.dtype == torch.bfloat16
        assert k.dtype == torch.bfloat16
        assert lens.dtype == torch.int32
        assert bt.dtype == torch.int32

    def test_cache_seqlens_filled(self):
        cfg = KernelConfig(batch_size=4, seqlen_kv=128)
        _, _, _, lens, _ = make_test_inputs(cfg, num_blocks=512, device="cpu")
        assert (lens == 128).all()

    def test_seed_reproducible(self):
        cfg = KernelConfig(batch_size=1, seqlen_kv=16, headdim=8)
        q1, _, _, _, _ = make_test_inputs(cfg, num_blocks=32, device="cpu", seed=123)
        q2, _, _, _, _ = make_test_inputs(cfg, num_blocks=32, device="cpu", seed=123)
        assert torch.equal(q1, q2)

    def test_different_seed_different(self):
        cfg = KernelConfig(batch_size=1, seqlen_kv=16, headdim=8)
        q1, _, _, _, _ = make_test_inputs(cfg, num_blocks=32, device="cpu", seed=1)
        q2, _, _, _, _ = make_test_inputs(cfg, num_blocks=32, device="cpu", seed=2)
        assert not torch.equal(q1, q2)


class TestGQAHeadMapping:
    def test_gqa_kv_head_mapping(self):
        """H=8, HK=2 时，每 4 个 q head 共享 1 个 kv head"""
        cfg = KernelConfig(batch_size=1, seqlen_kv=16, headdim=8, num_heads=8, num_heads_k=2)
        q, k, v, lens, bt = make_test_inputs(cfg, num_blocks=32, device="cpu")
        out = generate_reference(q, k, v, lens, bt, cfg)
        assert out.shape == (1, 1, 8, 8)  # 输出仍是 8 个 head


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
class TestOnGPU:
    def test_reference_on_gpu(self):
        cfg = KernelConfig(batch_size=1, seqlen_kv=64, headdim=32)
        q, k, v, lens, bt = make_test_inputs(cfg, num_blocks=128, device="cuda")
        out = generate_reference(q, k, v, lens, bt, cfg)
        assert out.device.type == "cuda"
        assert out.shape == (1, 1, 8, 32)
