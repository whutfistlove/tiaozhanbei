"""
严格的滚动 A/B 实验编排器测试。

覆盖四种结局 + 日志分类：
- KEEP（提速超噪声容限 + 正确）→ 版本链推进，kept_changes.md 有记录
- NOCHANGE（提速在噪声内）→ 不推进版本，rejected_changes.md
- REJECT（正确性失败）→ rejected_changes.md
- ERROR（编译失败/运行崩溃）→ errors.md
- ChangeProposal 单点改动校验

运行：pytest tests/test_ab_orchestrator.py -v
"""
import pytest
import torch
from pathlib import Path

from agent_system.strategy_schema import ChangeProposal, validate_single_change
from agent_system.domain_memory import DomainMemory
from agent_system.optimization_log import OptimizationLog
from agent_system.kernel_version_store import KernelVersionStore
from agent_system.real_orchestrator import run_ab_iteration
from agent_system.roofline_engine import KernelConfig


# ── 一个真正能编译+运行的"慢" kernel 作为 A（baseline）──
# 用 splitk_h128 的同款代码，但 NUM_SPLITS=1（退化，慢），让 B 容易超越
SLOW_KERNEL = r'''
#include <cstdint>
#include <cmath>
#include "mctlass/bfloat16.h"
#include "mcr/mc_runtime.h"
typedef mctlass::bfloat16_t __nv_bfloat16;
using mctlass::bfloat16_t;
static constexpr int BLOCK_DIM = 128;
static constexpr int NUM_SPLITS = 1;   // 退化：只用 1 个 split，SM 欠载
extern "C" __global__ void decode_kernel(
    const bfloat16_t* __restrict__ q, const bfloat16_t* __restrict__ k_cache,
    const bfloat16_t* __restrict__ v_cache, bfloat16_t* __restrict__ output,
    const int32_t* __restrict__ cache_seqlens, const int32_t* __restrict__ block_table,
    int64_t batch_size, int64_t seqlen_k, int64_t seqlen_q,
    int64_t num_heads, int64_t num_heads_k, int64_t headdim,
    int64_t page_block_size, int64_t blocks_per_batch) {
    int b = blockIdx.x / num_heads; int h = blockIdx.x % num_heads;
    if (b >= batch_size) return;
    int tid = threadIdx.x; int seqlen = cache_seqlens[b];
    int kv_h = h * num_heads_k / num_heads;
    extern __shared__ char smem[]; float* sq = (float*)smem; float* ss = (float*)(smem + headdim*4);
    if (tid < headdim) sq[tid] = float(q[b*num_heads*headdim + h*headdim + tid]);
    __syncthreads(); if (tid >= headdim) return;
    float scale = 1.0f/sqrtf((float)headdim), mx=-1e30f, se=0.0f, ol=0.0f;
    for (int t = 0; t < seqlen; ++t) {
        int pi = t/page_block_size, po = t%page_block_size;
        int phys = block_table[b*blocks_per_batch + pi];
        int64_t base = (int64_t)phys*page_block_size*num_heads_k*headdim + po*num_heads_k*headdim + kv_h*headdim;
        float dot=0.0f; for (int d=tid; d<headdim; d+=BLOCK_DIM) dot += sq[d]*float(k_cache[base+d]);
        ss[tid]=dot; __syncthreads();
        for (int a=BLOCK_DIM>>1; a>=32; a>>=1) { if(tid<a) ss[tid]+=ss[tid+a]; __syncthreads(); }
        if(tid<32){ float v=ss[tid]; for(int m=16;m>0;m>>=1) v+=__shfl_xor_sync(0xffffffff,v,m); ss[tid]=v; }
        __syncthreads(); float sc=ss[0]*scale; float nm=fmaxf(mx,sc);
        float eo=expf(mx-nm), ev=expf(sc-nm); se=se*eo+ev; mx=nm;
        if(tid<headdim) ol=ol*eo+ev*float(v_cache[base+tid]);
    }
    if(tid<headdim && se>0) output[b*num_heads*headdim+h*headdim+tid]=bfloat16_t(ol/se);
}
extern "C" void run_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* o, const int32_t* cl, const int32_t* bt,
    int64_t bs, int64_t sk, int64_t sq, int64_t nh, int64_t nhk, int64_t hd,
    int64_t pbs, int64_t nb, int64_t causal) {
    int64_t bpb = nb/bs;
    decode_kernel<<<(int)(bs*nh), BLOCK_DIM, (int)(hd*4*2)>>>(
        q,k,v,o,cl,bt,bs,sk,sq,nh,nhk,hd,pbs,bpb);
}
'''

# B：NUM_SPLITS=12（更快，正确）—— 只改这一处常量
FAST_KERNEL = SLOW_KERNEL.replace("static constexpr int NUM_SPLITS = 1;",
                                  "static constexpr int NUM_SPLITS = 12;")

# ── KEEP 测试用：确定性地 A 慢 / B 快 ──
# 取仓库 splitk_h128.cu。A 注入一个"冗余忙循环"（__syncthreads + dummy 累加），
# 使其在任何 GPU 状态下都确定性地比 B（原版）慢——避免依赖 split 行为的波动。
_SPLITK_SRC_PATH = Path(__file__).resolve().parent.parent / "kernel" / "splitk_h128.cu"
try:
    _SPLITK_FULL = _SPLITK_SRC_PATH.read_text()
    # B = 原版（正确且正常速度）
    SPLITK_FAST_B = _SPLITK_FULL
    # A = 原版 + 冗余忙循环（在 softmax 归一化前插入一段无意义但耗时的循环）→ 确定性变慢
    # 用长依赖链 + 大循环次数，避免被 GPU 并行/流水隐藏
    _BUSY_LOOP = (
        "        // 冗余忙循环：人为制造确定性 slowdown（测试用）\n"
        "        float _busy = 1.0f;\n"
        "        for (int _bi = 0; _bi < 2000; ++_bi) { _busy = sinf(_busy) * 0.5f + cosf(_busy) * 0.5f; }\n"
        "        out_local += _busy * 0.0f;\n"
    )
    # 在 "if (tid < headdim) {" 这行（softmax 之前）插入忙循环
    _ANCHOR = "    if (tid < headdim) {\n        partial_o[flat_idx * headdim + tid] = bfloat16_t(out_local);\n    }"
    _ANCHOR_NEW = _BUSY_LOOP + _ANCHOR
    SPLITK_SLOW_A = _SPLITK_FULL.replace(_ANCHOR, _ANCHOR_NEW)
except Exception:
    SPLITK_SLOW_A = ""
    SPLITK_FAST_B = ""


def make_proposal(target="NUM_SPLITS", pid="p1",
                  before="static constexpr int NUM_SPLITS = 1;",
                  after="static constexpr int NUM_SPLITS = 12;") -> ChangeProposal:
    """构造一个单点改动提案（before/after 是 diff 锚点片段，保证 changed_line_count 小）。

    注意：orchestrator 用 proposal.after 作为改动后的*完整 kernel 源码*，
    所以测试里会在构造后把 proposal.after 覆盖成完整源码（见各 Test 类）。
    """
    return ChangeProposal(
        proposal_id=pid, target=target, change_type="param_tune",
        one_line_summary=f"{target} tune",
        before=before, after=after, hypothesis="填满更多 SM",
    )


@pytest.fixture
def ab_env(tmp_path):
    """A/B 实验环境（小配置，快速）。

    KEEP 测试用 splitk kernel + 较大 seq（split 才有收益）；
    NOCHANGE/ERROR/REJECT 用单 kernel + 小 seq 即可。
    本 fixture 提供环境容器，具体 kernel/seq 由各 Test 指定。
    """
    if not torch.cuda.is_available():
        pytest.skip("A/B orchestrator 需要 GPU 编译运行真实 kernel")
    memory = DomainMemory(base_dir=tmp_path / "mem")
    log = OptimizationLog(log_dir=tmp_path / "log")
    store = KernelVersionStore(tmp_path / "versions")
    return memory, log, store, tmp_path


class TestChangeProposal:
    def test_single_change_accepted(self):
        p = ChangeProposal(proposal_id="p", target="NUM_SPLITS",
                           before="int X = 1;", after="int X = 2;")
        ok, _ = validate_single_change(p)
        assert ok

    def test_multi_change_rejected(self):
        before = "\n".join(f"line {i}" for i in range(10))
        after = "\n".join(f"line {i} CHANGED" for i in range(10))
        p = ChangeProposal(proposal_id="p", target="rewrite", before=before, after=after)
        ok, reason = validate_single_change(p)
        assert not ok
        assert "超过单轮上限" in reason

    def test_empty_rejected(self):
        p = ChangeProposal(proposal_id="p", target="x", before="", after="")
        ok, reason = validate_single_change(p)
        assert not ok

    def test_diff_summary_contains_target(self):
        p = ChangeProposal(proposal_id="p", target="NUM_SPLITS",
                           one_line_summary="tune", before="X=1;", after="X=2;")
        s = p.to_diff_summary()
        assert "NUM_SPLITS" in s
        assert "tune" in s


class TestABKeep:
    def test_keep_when_faster(self, ab_env):
        """单点改动 + B 明显更快 → KEEP，版本链推进。

        用真实 splitk_h128 kernel：A 强制 NUM_SPLITS=1（慢），B 自适应（快）。
        seq=4096 时 split 有显著收益（前面实测 ~10x）。
        """
        if not SPLITK_FAST_B:
            pytest.skip("splitk_h128.cu 源码不可读")
        memory, log, store, tmp = ab_env
        cfg = KernelConfig(batch_size=1, seqlen_kv=4096, headdim=128,
                           num_heads=8, num_heads_k=8)
        v0 = store.add_source("flashattention_kvcache_decode", SPLITK_SLOW_A,
                              "splitk forced NUM_SPLITS=1", verdict="KEEP", metrics={})
        store.promote(v0.version_id)
        proposal = ChangeProposal(
            proposal_id="p1", target="NUM_SPLITS", change_type="param_tune",
            one_line_summary="NUM_SPLITS 1->12 自适应", before="NUM_SPLITS=1",
            after="NUM_SPLITS=12", hypothesis="填满 SM",
            patched_source=SPLITK_FAST_B,
        )

        def gen(cfg, bottleneck, mem):
            return proposal

        result = run_ab_iteration(
            iteration=1, cfg=cfg, current_code=SPLITK_SLOW_A, current_version_id=v0.version_id,
            generate_proposal_fn=gen, memory=memory, log=log, version_store=store,
            workdir=tmp / "w1", noise_margin=0.03, warmup=3, repeats=10,
        )
        assert result.verdict == "KEEP"
        assert result.speedup is not None and result.speedup > 1.0
        assert result.promoted_version is not None
        # 版本链推进
        assert store.current("flashattention_kvcache_decode").version_id == result.promoted_version
        # kept_changes.md 有记录
        kept = (tmp / "log" / "kept_changes.md").read_text()
        assert "KEEP" in kept and "NUM_SPLITS" in kept


class TestABNoChange:
    def test_nochange_when_within_noise(self, ab_env):
        """改动正确但提速不足（在噪声容限内）→ NOCHANGE，不推进版本。

        B = A（同样的 kernel）。用一个很宽的 noise_margin（50%）确保即使
        benchmark 抖动让 B 偶尔略快，也跨不过 50% 这个真实优化的门槛——
        这正是噪声容限的语义：把"小幅波动"和"真实优化"分开。
        """
        memory, log, store, tmp = ab_env
        cfg = KernelConfig(batch_size=1, seqlen_kv=256, headdim=128,
                           num_heads=8, num_heads_k=8)
        v0 = store.add_source("flashattention_kvcache_decode", SLOW_KERNEL,
                              "slow baseline", verdict="KEEP", metrics={})
        store.promote(v0.version_id)
        # B = A（同样的慢 kernel）；50% 噪声容限下不可能被判为真实提速
        proposal = ChangeProposal(
            proposal_id="p2", target="noop", change_type="param_tune",
            one_line_summary="no actual change", before="X=1;", after="X=1;",
            hypothesis="测试噪声容限", patched_source=SLOW_KERNEL,
        )

        def gen(cfg, bottleneck, mem):
            return proposal

        result = run_ab_iteration(
            iteration=1, cfg=cfg, current_code=SLOW_KERNEL, current_version_id=v0.version_id,
            generate_proposal_fn=gen, memory=memory, log=log, version_store=store,
            workdir=tmp / "w2", noise_margin=0.50, warmup=3, repeats=10,
        )
        assert result.verdict == "NOCHANGE"
        assert result.promoted_version is None
        # 版本未推进
        assert store.current("flashattention_kvcache_decode").version_id == v0.version_id


class TestABError:
    def test_error_on_compile_failure(self, ab_env):
        """B 编译失败 → ERROR，errors.md 有记录，记忆库记录失败。"""
        memory, log, store, tmp = ab_env
        cfg = KernelConfig(batch_size=1, seqlen_kv=256, headdim=128,
                           num_heads=8, num_heads_k=8)
        v0 = store.add_source("flashattention_kvcache_decode", SLOW_KERNEL,
                              "slow baseline", verdict="KEEP", metrics={})
        store.promote(v0.version_id)
        broken = SLOW_KERNEL.replace("typedef mctlass::bfloat16_t __nv_bfloat16;",
                                     "typedef BROKEN_TYPE __nv_bfloat16;")
        proposal = ChangeProposal(
            proposal_id="p3", target="bad_include", change_type="api_swap",
            one_line_summary="swap type", before="good_type", after="BROKEN_TYPE",
            hypothesis="", patched_source=broken,
        )

        def gen(cfg, bottleneck, mem):
            return proposal

        result = run_ab_iteration(
            iteration=1, cfg=cfg, current_code=SLOW_KERNEL, current_version_id=v0.version_id,
            generate_proposal_fn=gen, memory=memory, log=log, version_store=store,
            workdir=tmp / "w3", noise_margin=0.03, warmup=3, repeats=10,
        )
        assert result.verdict == "ERROR"
        assert len(memory.failures) >= 1
        err_log = (tmp / "log" / "errors.md").read_text()
        assert "ERROR" in err_log


class TestABReject:
    def test_reject_on_incorrect(self, ab_env):
        """B 编译运行通过但数值错 → REJECT，rejected_changes.md 有记录。

        用一个输出写错地址的 kernel（数值错但不崩溃）。
        """
        memory, log, store, tmp = ab_env
        cfg = KernelConfig(batch_size=1, seqlen_kv=256, headdim=128,
                           num_heads=8, num_heads_k=8)
        v0 = store.add_source("flashattention_kvcache_decode", SLOW_KERNEL,
                              "slow baseline", verdict="KEEP", metrics={})
        store.promote(v0.version_id)
        # 让 decode_kernel 的输出路径写错地址 → 数值错（但能跑完不崩溃）
        wrong = SLOW_KERNEL.replace(
            "output[b*num_heads*headdim+h*headdim+tid]=bfloat16_t(ol/se);",
            "output[0]=bfloat16_t(0.0f);")  # 全写第一个元素，其余为零
        proposal = ChangeProposal(
            proposal_id="p4", target="bad_output", change_type="other",
            one_line_summary="wrong output addr", before="correct out", after="zero out",
            hypothesis="", patched_source=wrong,
        )

        def gen(cfg, bottleneck, mem):
            return proposal

        result = run_ab_iteration(
            iteration=1, cfg=cfg, current_code=SLOW_KERNEL, current_version_id=v0.version_id,
            generate_proposal_fn=gen, memory=memory, log=log, version_store=store,
            workdir=tmp / "w4", noise_margin=0.03, warmup=3, repeats=10,
        )
        assert result.verdict == "REJECT"
        rej_log = (tmp / "log" / "rejected_changes.md").read_text()
        assert "REJECT" in rej_log


class TestLogClassification:
    def test_summary_counts_correctly(self, tmp_path):
        """summary.md 的分类计数与实际条目一致。"""
        from agent_system.optimization_log import OptimizationEntry
        log = OptimizationLog(log_dir=tmp_path / "log")
        for i, (verdict, cat) in enumerate([
            ("KEEP", "keep"), ("KEEP", "keep"), ("REJECT", "reject"),
            ("ERROR", "error_compile"), ("NOCHANGE", "nochange")], 1):
            log.record(OptimizationEntry(
                iteration=i, timestamp="t", change_description=f"ch{i}",
                target_config="cfg", baseline_time_ms=1.0, candidate_time_ms=0.9,
                speedup=1.1 if verdict == "KEEP" else None,
                bandwidth_util_before=0.01, bandwidth_util_after=None,
                correctness_passed=True, verdict=verdict, category=cat,
                proposal_target=f"t{i}",
            ))
        summary = (tmp_path / "log" / "summary.md").read_text()
        # 各桶计数
        assert "| keep | 2 |" in summary
        assert "| reject | 1 |" in summary
        assert "| error | 1 |" in summary
        assert "| nochange | 1 |" in summary

    def test_category_files_isolated(self, tmp_path):
        """分类文件各自只含对应类别的条目。"""
        from agent_system.optimization_log import OptimizationEntry
        log = OptimizationLog(log_dir=tmp_path / "log")
        log.record(OptimizationEntry(iteration=1, timestamp="t", change_description="k",
            target_config="cfg", baseline_time_ms=1.0, candidate_time_ms=0.9, speedup=1.1,
            bandwidth_util_before=0.01, bandwidth_util_after=None, correctness_passed=True,
            verdict="KEEP", category="keep", proposal_target="t1"))
        log.record(OptimizationEntry(iteration=2, timestamp="t", change_description="e",
            target_config="cfg", baseline_time_ms=1.0, candidate_time_ms=None, speedup=None,
            bandwidth_util_before=0.01, bandwidth_util_after=None, correctness_passed=False,
            verdict="ERROR", category="error_compile", proposal_target="t2"))
        kept = (tmp_path / "log" / "kept_changes.md").read_text()
        errs = (tmp_path / "log" / "errors.md").read_text()
        assert "轮次 1" in kept and "轮次 2" not in kept
        assert "轮次 2" in errs and "轮次 1" not in errs
