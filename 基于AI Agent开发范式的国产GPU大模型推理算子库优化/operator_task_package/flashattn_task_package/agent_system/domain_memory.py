"""
领域记忆库 + 协同进化的硬件信念（创新点 B）。

针对 AscendKernelGen 实证的"通用 LLM 在国产硬件正确率≈0%"问题，
用 in-context 的领域记忆替代昂贵的 SFT/RL 微调。

两部分：
1. Skill（正样本）：成功优化经验，Voyager skill library 思想
2. failure_cases（负样本）：错误衍生监督，AscendKernelGen 方法
3. hardware_belief（信念笔记）：每轮 benchmark 抽象"配置→性能"规律，
   让 agent 自学陌生 MXC500 架构（K-Search co-evolving world model）

全部是可序列化的 markdown/json，支持检索与注入。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


MEMORY_DIR = Path(__file__).resolve().parent / "domain_memory"


@dataclass
class FailureCase:
    """单条失败案例（错误衍生监督的素材）。"""
    case_id: str
    timestamp: str
    category: str               # api_misuse | sync_error | boundary_error | compile_error | correctness_error
    symptom: str                # 症状（错误信息摘要）
    root_cause: str             # 根因分析
    fix: str                    # 修正方案
    code_snippet: str = ""      # 出错代码片段
    config: str = ""            # 触发配置

    def to_markdown(self) -> str:
        return (
            f"### {self.case_id} [{self.category}]\n"
            f"- **症状**: {self.symptom}\n"
            f"- **根因**: {self.root_cause}\n"
            f"- **修正**: {self.fix}\n"
            + (f"- **配置**: {self.config}\n" if self.config else "")
            + (f"- **代码**: `{self.code_snippet}`\n" if self.code_snippet else "")
        )


@dataclass
class HardwareBelief:
    """协同进化的硬件信念笔记（创新点 B 核心）。"""
    entries: list = field(default_factory=list)  # [{"observation","rule","timestamp","confidence"}]

    def add(self, observation: str, rule: str, confidence: float = 0.5):
        self.entries.append({
            "observation": observation,
            "rule": rule,
            "confidence": confidence,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    def to_markdown(self) -> str:
        if not self.entries:
            return "# Hardware Belief（暂无）\n\n（agent 尚未学到 MXC500 的架构规律）\n"
        lines = ["# Hardware Belief — 自学 MXC500 架构规律\n"]
        for i, e in enumerate(self.entries, 1):
            lines.append(
                f"## 规律 {i}（置信度 {e['confidence']:.0%}）\n"
                f"- **观察**: {e['observation']}\n"
                f"- **规律**: {e['rule']}\n"
                f"- **时间**: {e['timestamp']}\n"
            )
        return "\n".join(lines)


class DomainMemory:
    """
    领域记忆库管理器。

    负责读写 skill / failure_cases / hardware_belief，
    以及按任务语义检索（简化版，关键词匹配）。
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or MEMORY_DIR
        self.failures: list[FailureCase] = []
        self.belief = HardwareBelief()
        self._load()

    # ── 持久化 ──
    def _failures_path(self) -> Path:
        return self.base_dir / "failure_cases" / "cases.json"

    def _belief_path(self) -> Path:
        return self.base_dir / "hardware_belief.json"

    def _load(self):
        fp = self._failures_path()
        if fp.exists():
            data = json.loads(fp.read_text(encoding="utf-8"))
            self.failures = [FailureCase(**d) for d in data]
        bp = self._belief_path()
        if bp.exists():
            data = json.loads(bp.read_text(encoding="utf-8"))
            self.belief = HardwareBelief(entries=data.get("entries", []))

    def save(self):
        self._failures_path().parent.mkdir(parents=True, exist_ok=True)
        self._failures_path().write_text(
            json.dumps([asdict(f) for f in self.failures], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._belief_path().write_text(
            json.dumps({"entries": self.belief.entries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── 失败案例 ──
    def record_failure(self, category: str, symptom: str, root_cause: str,
                       fix: str, code_snippet: str = "", config: str = "") -> FailureCase:
        """记录一条失败案例（错误衍生监督）。"""
        case = FailureCase(
            case_id=f"fail_{len(self.failures)+1:04d}_{int(time.time())%100000}",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            category=category, symptom=symptom, root_cause=root_cause,
            fix=fix, code_snippet=code_snippet, config=config,
        )
        self.failures.append(case)
        self.save()
        return case

    def query_failures(self, keyword: str, limit: int = 5) -> list[FailureCase]:
        """按关键词检索失败案例（注入 Coder prompt 规避）。"""
        kw = keyword.lower()
        scored = []
        for f in self.failures:
            text = f"{f.category} {f.symptom} {f.root_cause} {f.fix}".lower()
            if kw in text:
                scored.append(f)
        return scored[:limit]

    # ── 硬件信念 ──
    def record_belief(self, observation: str, rule: str, confidence: float = 0.5):
        """记录一条硬件信念（Reflector 角色，每轮 benchmark 后更新）。"""
        self.belief.add(observation, rule, confidence)
        self.save()

    # ── 注入上下文（给 Coder/Analyst）──
    def build_context(self, task_keyword: str = "") -> str:
        """构造注入 LLM 的领域记忆上下文。"""
        parts = []
        if task_keyword:
            fails = self.query_failures(task_keyword)
            if fails:
                parts.append("## ⚠️ 已知陷阱（请规避）\n")
                for f in fails:
                    parts.append(f.to_markdown())
        if self.belief.entries:
            parts.append("\n" + self.belief.to_markdown())
        return "\n".join(parts) if parts else "（暂无领域记忆）"
