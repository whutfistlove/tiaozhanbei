"""
optimization_log 的单元测试。

测试覆盖：
- 条目记录与序列化（md + jsonl）
- best_entry 检索
- summary 统计
- 持久化往返
- markdown 表格格式

运行：pytest tests/test_optimization_log.py -v
"""
import pytest
from pathlib import Path

from agent_system.optimization_log import OptimizationLog, OptimizationEntry


def make_entry(it, verdict="KEEP", speedup=2.0, time_ms=0.5):
    return OptimizationEntry(
        iteration=it,
        timestamp="2026-07-15 12:00:00",
        change_description=f"优化 v{it}",
        target_config="batch=1,seq=4096,d=128",
        baseline_time_ms=1.0,
        candidate_time_ms=time_ms,
        speedup=speedup,
        bandwidth_util_before=0.45,
        bandwidth_util_after=0.6,
        correctness_passed=True,
        verdict=verdict,
        gap_to_roofline=2.0,
    )


class TestOptimizationLog:
    def test_record_creates_files(self, tmp_path):
        log = OptimizationLog(log_path=tmp_path / "log.md")
        log.record(make_entry(1))
        assert (tmp_path / "log.md").exists()
        assert (tmp_path / "log.jsonl").exists()

    def test_markdown_has_header(self, tmp_path):
        log = OptimizationLog(log_path=tmp_path / "log.md")
        log.record(make_entry(1))
        content = (tmp_path / "log.md").read_text()
        assert "优化日志" in content
        assert "| 轮次 |" in content

    def test_jsonl_roundtrip(self, tmp_path):
        log = OptimizationLog(log_path=tmp_path / "log.md")
        log.record(make_entry(1, speedup=2.5))
        log.record(make_entry(2, speedup=3.0))
        # 重新加载
        log2 = OptimizationLog(log_path=tmp_path / "log.md")
        assert len(log2.entries) == 2
        assert log2.entries[0].speedup == 2.5
        assert log2.entries[1].speedup == 3.0

    def test_best_entry(self, tmp_path):
        log = OptimizationLog(log_path=tmp_path / "log.md")
        log.record(make_entry(1, speedup=1.5))
        log.record(make_entry(2, speedup=3.0))
        log.record(make_entry(3, speedup=2.0))
        best = log.best_entry()
        assert best is not None
        assert best.speedup == 3.0

    def test_best_entry_none_when_all_rollback(self, tmp_path):
        log = OptimizationLog(log_path=tmp_path / "log.md")
        log.record(make_entry(1, verdict="ROLLBACK", speedup=None, time_ms=None))
        assert log.best_entry() is None

    def test_summary(self, tmp_path):
        log = OptimizationLog(log_path=tmp_path / "log.md")
        log.record(make_entry(1, verdict="KEEP", speedup=2.0))
        log.record(make_entry(2, verdict="ROLLBACK", speedup=None, time_ms=None))
        s = log.summary()
        assert "共 2 轮" in s
        assert "KEEP 1" in s

    def test_summary_empty(self, tmp_path):
        log = OptimizationLog(log_path=tmp_path / "log.md")
        assert "暂无" in log.summary()

    def test_markdown_row_format(self, tmp_path):
        e = make_entry(1)
        row = e.to_markdown_row()
        assert "| 1 |" in row
        assert "2.00x" in row
        assert "KEEP" in row

    def test_append_multiple(self, tmp_path):
        log = OptimizationLog(log_path=tmp_path / "log.md")
        for i in range(5):
            log.record(make_entry(i+1))
        assert len(log.entries) == 5
