"""
Optimization Logger（Logger 角色）。

记录每轮迭代的完整信息——赛题"可复现性"(20%)的核心物料。
每条记录含：版本号、改动描述、配置、baseline/优化前后、加速比、正确性、决策。

设计为 append-only 的结构化 markdown + 可机读 jsonl。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class OptimizationEntry:
    """单轮优化日志条目。"""
    iteration: int
    timestamp: str
    change_description: str
    target_config: str             # 如 "batch=1,seq=4096,headdim=128"
    baseline_time_ms: float
    candidate_time_ms: Optional[float]
    speedup: Optional[float]
    bandwidth_util_before: float
    bandwidth_util_after: Optional[float]
    correctness_passed: Optional[bool]
    verdict: str                   # KEEP / ROLLBACK / REJECT / PENDING / NOCHANGE / ERROR
    gap_to_roofline: Optional[float] = None
    notes: str = ""
    # ── A/B 实验与分类归档新增字段 ──
    category: str = ""             # keep | reject | error_compile | error_runtime | nochange
    proposal_target: str = ""      # 本轮改的什么（对应 ChangeProposal.target）
    diff_summary: str = ""         # 改动摘要（ChangeProposal.to_diff_summary()）
    baseline_version: str = ""     # A 的版本 id（滚动追踪）
    promoted_version: str = ""     # KEEP 时新版本 id
    noise_margin: float = 0.0      # 本轮用的噪声容限
    reject_reason: str = ""        # KEEP 以外的原因（编译错误/数值错/未提速…）
    used_median: bool = False      # 是否用中位数判定

    def to_markdown_row(self) -> str:
        sp = f"{self.speedup:.2f}x" if self.speedup else "—"
        ct = f"{self.candidate_time_ms:.4f}" if self.candidate_time_ms else "—"
        ua = f"{self.bandwidth_util_after:.1%}" if self.bandwidth_util_after else "—"
        cp = "✓" if self.correctness_passed else ("✗" if self.correctness_passed is False else "—")
        return (
            f"| {self.iteration} | {self.change_description[:30]} | "
            f"{self.target_config} | {self.baseline_time_ms:.4f} | {ct} | "
            f"{sp} | {self.bandwidth_util_before:.1%}→{ua} | {cp} | {self.verdict} |"
        )

    def category_or_infer(self) -> str:
        """显式 category 优先；否则从 verdict 推断（向后兼容旧调用）。"""
        if self.category:
            return self.category
        v = self.verdict
        if v == "KEEP":
            return "keep"
        if v == "REJECT":
            return "reject"
        if v == "ERROR":
            return "error_runtime"
        if v == "NOCHANGE":
            return "nochange"
        return v.lower() if v else "unknown"


# ── category → 文件名映射 ──
# 每个分类对应一份独立归档文件，便于按结果检索与审计。
_CATEGORY_FILES = {
    "keep": "kept_changes.md",           # 所有 KEEP（保留的代码）
    "reject": "rejected_changes.md",     # 正确性失败
    "error_patch": "errors.md",          # before/after 补丁无法唯一应用
    "error_compile": "errors.md",        # 编译错误
    "error_runtime": "errors.md",        # 运行崩溃（与编译错误合并在 errors.md）
    "nochange": "rejected_changes.md",   # 在噪声内未提速（归入"未保留"）
}

# category → 用于 summary 统计的桶
_CATEGORY_BUCKETS = ("keep", "reject", "error", "nochange")


def _bucket_of(category: str) -> str:
    """把细分 category 归并到 summary 统计桶。"""
    if category.startswith("error"):
        return "error"
    if category in ("keep", "reject", "nochange"):
        return category
    return "other"


class OptimizationLog:
    """优化日志管理器（append-only）。

    支持两种构造方式：
    - OptimizationLog(log_dir=Path(...))  ← 推荐，启用分类归档
    - OptimizationLog(log_path=Path(...)) ← 旧接口，向后兼容（单文件 + 分类文件同目录）
    """

    def __init__(self, log_path: Optional[Path] = None, log_dir: Optional[Path] = None):
        if log_dir is not None:
            self.log_dir = Path(log_dir)
            self.log_path = self.log_dir / "optimization_log.md"
            self.jsonl_path = self.log_dir / "optimization_log.jsonl"   # 目录模式用规范名
        elif log_path is not None:
            self.log_path = Path(log_path)
            self.log_dir = self.log_path.parent
            self.jsonl_path = self.log_path.with_suffix(".jsonl")      # 旧接口：与 md 同名（向后兼容）
        else:
            self.log_path = Path("optimization_log.md")
            self.log_dir = self.log_path.parent
            self.jsonl_path = self.log_path.with_suffix(".jsonl")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.summary_path = self.log_dir / "summary.md"
        self.entries: list[OptimizationEntry] = []
        self._load_jsonl()

    # ── 分类归档文件路径 ──
    def _category_file(self, category: str) -> Path:
        fname = _CATEGORY_FILES.get(category)
        if fname is None:
            # 未知 category 兜底到 errors.md 不合适，单列一个
            fname = "other.md"
        return self.log_dir / fname

    def _load_jsonl(self):
        if self.jsonl_path.exists():
            for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        self.entries.append(OptimizationEntry(**json.loads(line)))
                    except TypeError:
                        # 极旧格式（缺新字段）兼容：忽略无法解析的行
                        continue

    def record(self, entry: OptimizationEntry):
        self.entries.append(entry)
        # append jsonl（机读）
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        # 重写所有人读 markdown（完整流水 + 分类 + summary）
        self._write_markdown()
        self._write_category_files()
        self._write_summary()

    def _write_markdown(self):
        lines = [
            "# 优化日志（Optimization Log）",
            "",
            "> 赛题 Agent 可复现性(20%)核心物料。每轮迭代的完整记录。",
            "> 分类归档见同目录：kept_changes.md / rejected_changes.md / errors.md / summary.md",
            "",
            "| 轮次 | 改动 | 配置 | baseline(ms) | 优化后(ms) | 加速比 | 带宽利用率 | 正确性 | 决策 |",
            "|------|------|------|-------------|-----------|--------|----------|--------|------|",
        ]
        for e in self.entries:
            lines.append(e.to_markdown_row())
        self.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _entry_detail_block(self, e: OptimizationEntry) -> str:
        """单条记录的详细块（用于分类归档，含 diff/版本/原因）。"""
        sp = f"{e.speedup:.3f}x" if e.speedup else "—"
        ct = f"{e.candidate_time_ms:.4f}" if e.candidate_time_ms else "—"
        ver = e.baseline_version or "—"
        promoted = e.promoted_version or "—"
        lines = [
            f"### 轮次 {e.iteration} · {e.verdict} · {e.category_or_infer()}",
            f"- **改动点**: {e.proposal_target or e.change_description[:40]}",
            f"- **配置**: {e.target_config}",
            f"- **A/B**: baseline={e.baseline_time_ms:.4f}ms → candidate={ct} (speedup {sp})",
            f"- **版本**: A={ver}" + (f" → B={promoted}" if e.promoted_version else ""),
        ]
        if e.noise_margin:
            lines.append(f"- **噪声容限**: {e.noise_margin:.1%}" + ("（中位数判定）" if e.used_median else ""))
        if e.reject_reason:
            lines.append(f"- **拒绝原因**: {e.reject_reason}")
        if e.diff_summary:
            lines.append("")
            lines.append("<details><summary>diff</summary>")
            lines.append("")
            lines.append("```diff")
            lines.append(e.diff_summary)
            lines.append("```")
            lines.append("")
            lines.append("</details>")
        lines.append("")
        return "\n".join(lines)

    def _write_category_files(self):
        """按 category 把条目分别写入对应归档文件。每次全量重写。"""
        # 按"目标文件"聚合
        groups: dict[Path, list[OptimizationEntry]] = {}
        for e in self.entries:
            cat = e.category_or_infer()
            fp = self._category_file(cat)
            groups.setdefault(fp, []).append(e)
        for fp, entries in groups.items():
            cat_set = {e.category_or_infer() for e in entries}
            title = " / ".join(sorted(cat_set))
            lines = [
                f"# 归档：{title}",
                "",
                f"> 共 {len(entries)} 条记录。此文件由 OptimizationLog 自动维护。",
                "",
            ]
            for e in entries:
                lines.append(self._entry_detail_block(e))
            fp.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_summary(self):
        """汇总统计：分类计数 + 最佳加速 + 版本链。"""
        from collections import Counter
        buckets = Counter(_bucket_of(e.category_or_infer()) for e in self.entries)
        lines = [
            "# 优化汇总（Summary）",
            "",
            f"> 共 {len(self.entries)} 轮迭代。",
            "",
            "| 类别 | 数量 |",
            "|------|------|",
        ]
        for b in _CATEGORY_BUCKETS:
            lines.append(f"| {b} | {buckets.get(b, 0)} |")
        if buckets:
            lines.append(f"| other | {buckets.get('other', 0)} |")
        best = self.best_entry()
        if best:
            lines.append("")
            lines.append(f"**最佳单轮加速**: {best.speedup:.2f}x（轮次 {best.iteration}，"
                         f"改动 `{best.proposal_target or best.change_description[:30]}`）")
        # 版本链（按 promoted_version 顺序）
        kept = [e for e in self.entries if e.verdict == "KEEP" and e.promoted_version]
        if kept:
            chain = " → ".join(e.promoted_version for e in kept)
            lines.append("")
            lines.append(f"**版本演进链**: {chain}")
        self.summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def best_entry(self) -> Optional[OptimizationEntry]:
        """返回加速比最大的 KEEP 记录。"""
        kept = [e for e in self.entries if e.verdict == "KEEP" and e.speedup]
        if not kept:
            return None
        return max(kept, key=lambda e: e.speedup)

    def entries_in_category(self, category: str) -> list[OptimizationEntry]:
        """返回某个 category 的所有记录（用于测试与检索）。"""
        return [e for e in self.entries if e.category_or_infer() == category]

    def summary(self) -> str:
        if not self.entries:
            return "（暂无优化记录）"
        n_keep = sum(1 for e in self.entries if e.verdict == "KEEP")
        n_rollback = sum(1 for e in self.entries if e.verdict == "ROLLBACK")
        best = self.best_entry()
        best_str = f"最佳加速 {best.speedup:.2f}x（轮次 {best.iteration}）" if best else "无"
        return f"共 {len(self.entries)} 轮：KEEP {n_keep}，ROLLBACK {n_rollback}，{best_str}"
