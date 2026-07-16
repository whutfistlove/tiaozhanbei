"""
Real Coder —— 用真实 LLM（MiniMax-M2.7）生成/优化 kernel 代码。

替代 mock generate_candidates_fn。
Coder 角色的真实实现：读取瓶颈报告 + 领域记忆，让 LLM 生成优化候选。

关键设计：
- 注入 AGENTS.md 的铁律 + mctlass-usage skill + failure_cases（创新点B）
- LLM 生成 .cu 代码 + 候选描述
- 解析 LLM 输出为结构化候选列表
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from agent_system.llm_client import chat, extract_json
from agent_system.llm_cost_model import Candidate
from agent_system.roofline_engine import KernelConfig
from agent_system.domain_memory import DomainMemory


# Coder 的系统提示（注入领域知识）
CODER_SYSTEM = """你是沐曦 MACA 平台的 mctlass 算子工程师，专门优化 FlashAttention decode 的 run_kernel。

核心铁律：
1. 每次只改一处优化
2. 核心矩阵计算用 mctlass，bf16 用 typedef mctlass::bfloat16_t __nv_bfloat16
3. 接口签名严格匹配 extern "C" void run_kernel(...)
4. paged KV 寻址：page_idx=t/page_size, page_off=t%page_size, phys=block_table[b*blocks_per_batch+page_idx]

⚠️ 已知陷阱（必须规避，来自真实编译失败）：
- 不要用 cuda_bf16.h / cuda_runtime.h / mctlass/matrix_utils.h（不存在）
- 只用这些 include：<cstdint> <cmath> "mctlass/bfloat16.h"
- __maca_bfloat16 是不完整类型，必须用 mctlass::bfloat16_t
- bf16 转 float: float(x)，float 转 bf16: bfloat16_t(x)

输出格式（严格遵守）：
对每个候选，输出：
```candidate
description: 一句话描述
params: {"split_k": N, "tile_n": N}
```
```cuda
// 完整的 .cu 代码（含 run_kernel），只用白名单 include
```

MACA 环境确认：
- warp shuffle: __shfl_xor_sync(0xffffffff, val, mask)
- 编译: mxcc -std=c++17 -fPIC -shared -DMACA_ARCH=1000
"""


def _parse_candidates(text: str) -> List[dict]:
    """
    从 LLM 输出解析候选列表（鲁棒，容忍多种格式）。

    策略：
    1. 先尝试严格的 ```candidate + ```cuda 双块格式
    2. 若失败，提取所有 ```cuda/cpp/c++ 代码块，每个作为一个候选
    3. 从代码块前的文字提取描述
    """
    # 先剥离 <think> 标签
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    candidates = []

    # 策略1：严格双块格式
    pattern = re.compile(
        r"```candidate\s*(.*?)\s*```\s*```(?:cuda|cpp|c\+\+)?\s*(.*?)\s*```",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        meta_block, code_block = m.group(1), m.group(2)
        desc, params = _parse_meta(meta_block)
        if code_block.strip() and "run_kernel" in code_block:
            candidates.append({"description": desc, "params": params, "code": code_block.strip()})

    if candidates:
        return candidates

    # 策略2：提取所有含 run_kernel 的代码块
    code_pattern = re.compile(r"```(?:cuda|cpp|c\+\+)?\s*(.*?)\s*```", re.DOTALL)
    blocks = list(code_pattern.finditer(text))
    for i, m in enumerate(blocks):
        code = m.group(1).strip()
        if "run_kernel" not in code:
            continue
        # 描述：取代码块前的文字（最多 200 字符）
        preceding = text[:m.start()].rstrip()
        # 取最后一个非空行作为描述
        lines = [l.strip() for l in preceding.split("\n") if l.strip()]
        desc = lines[-1][:100] if lines else f"候选 {i+1}"
        # 去掉描述里的 markdown 标记
        desc = re.sub(r"^#+\s*", "", desc)
        # 尝试从代码里提取 split_k 等参数
        params = {}
        sk = re.search(r"split_k\s*[=:]\s*(\d+)", code)
        if sk:
            params["split_k"] = int(sk.group(1))
        candidates.append({"description": desc, "params": params, "code": code})

    return candidates


def _parse_meta(meta_block: str) -> tuple:
    """解析候选 meta（description + params）。"""
    desc = ""
    params = {}
    for line in meta_block.split("\n"):
        if line.strip().startswith("description:"):
            desc = line.split(":", 1)[1].strip()
        elif line.strip().startswith("params:"):
            try:
                params = extract_json(line.split(":", 1)[1].strip())
            except Exception:
                params = {}
    return desc, params


def generate_candidates(
    cfg: KernelConfig,
    bottleneck: str,
    memory: DomainMemory,
    current_code: str = "",
    num_candidates: int = 3,
    model: str = "MiniMax-M2.7",
) -> List[Candidate]:
    """
    真实 Coder：让 LLM 生成优化候选。

    返回 Candidate 列表（带 .params，code 存在 ._code 私有属性）。
    """
    # 构造上下文：领域记忆
    mem_ctx = memory.build_context("mctlass split_k") if memory else ""

    prompt = f"""当前 kernel 性能瓶颈分析：
{bottleneck}

当前配置：batch={cfg.batch_size}, seqlen_kv={cfg.seqlen_kv}, headdim={cfg.headdim}, num_heads={cfg.num_heads}
目标硬件：沐曦 C500（1.8TB/s 带宽，memory-bound，无 profiler）

已知经验：
{mem_ctx}

请基于瓶颈分析，生成 {num_candidates} 个优化候选。每个候选只改一处。
优先考虑：Split-K 并行（小 batch 时 SM 欠载，最大收益）。

当前 kernel 代码（供参考修改）：
```cuda
{current_code[:2000] if current_code else "// 无（从零生成）"}
```
"""
    try:
        resp = chat(
            [{"role": "system", "content": CODER_SYSTEM},
             {"role": "user", "content": prompt}],
            model=model, temperature=0.7, max_tokens=6000,
        )
        parsed = _parse_candidates(resp.content)
    except Exception as e:
        # LLM 失败时返回空，由 orchestrator 记录失败
        return []

    candidates = []
    for i, p in enumerate(parsed):
        c = Candidate(
            candidate_id=f"llm_cand_{i}",
            description=p["description"],
            params=p.get("params", {}),
            confidence=0.6,  # 默认置信度，后续 cost model 调整
        )
        # 把代码挂在私有属性上（不污染 dataclass）
        object.__setattr__(c, "_code", p["code"])
        candidates.append(c)
    return candidates


def get_candidate_code(candidate: Candidate) -> str:
    """取候选的 .cu 代码。"""
    return getattr(candidate, "_code", "")
