"""
domain_memory 的单元测试。

测试覆盖：
- FailureCase 记录/序列化/检索
- HardwareBelief 信念更新
- 持久化（save/load 往返）
- 上下文注入
- 临时目录隔离（不污染真实记忆库）

运行：pytest tests/test_domain_memory.py -v
"""
import json
import pytest
import tempfile
from pathlib import Path

from agent_system.domain_memory import (
    DomainMemory, FailureCase, HardwareBelief,
)


@pytest.fixture
def mem(tmp_path):
    """每个测试用独立临时目录，互不污染。"""
    return DomainMemory(base_dir=tmp_path)


class TestFailureCase:
    def test_record_failure(self, mem):
        case = mem.record_failure(
            category="api_misuse",
            symptom="mctlass GEMM 编译报错",
            root_cause="InstructionShape 写成 16x16x8 但 bf16 必须 16x16x16",
            fix="改为 InstructionShape<16,8,16>",
        )
        assert case.case_id.startswith("fail_")
        assert case.category == "api_misuse"
        assert len(mem.failures) == 1

    def test_record_persists(self, mem, tmp_path):
        mem.record_failure("compile_error", "s1", "r1", "f1")
        # 重新加载应能读回
        mem2 = DomainMemory(base_dir=tmp_path)
        assert len(mem2.failures) == 1
        assert mem2.failures[0].symptom == "s1"

    def test_to_markdown(self, mem):
        case = mem.record_failure("sync_error", "race condition", "missing __syncthreads", "add sync")
        md = case.to_markdown()
        assert "sync_error" in md
        assert "race condition" in md

    def test_query_by_keyword(self, mem):
        mem.record_failure("api_misuse", "mctlass GEMM error", "shape wrong", "fix shape")
        mem.record_failure("boundary_error", "page offset overflow", "no boundary check", "add check")
        results = mem.query_failures("mctlass")
        assert len(results) == 1
        assert results[0].category == "api_misuse"

    def test_query_no_match(self, mem):
        mem.record_failure("api_misuse", "xxx", "yyy", "zzz")
        assert mem.query_failures("nonexistent_keyword") == []

    def test_multiple_failures(self, mem):
        for i in range(5):
            mem.record_failure("api_misuse", f"symptom_{i}", f"cause_{i}", f"fix_{i}")
        assert len(mem.failures) == 5
        all_md = [f.to_markdown() for f in mem.failures]
        assert all("symptom" in m or "symptom_" in m for m in all_md)


class TestHardwareBelief:
    def test_add_belief(self, mem):
        mem.record_belief("split_k=4 在 seq=4096 比 split_k=2 快 1.8x", "长序列小batch用大split", 0.8)
        assert len(mem.belief.entries) == 1
        assert mem.belief.entries[0]["confidence"] == 0.8

    def test_belief_persists(self, mem, tmp_path):
        mem.record_belief("obs1", "rule1", 0.6)
        mem2 = DomainMemory(base_dir=tmp_path)
        assert len(mem2.belief.entries) == 1
        assert mem2.belief.entries[0]["rule"] == "rule1"

    def test_belief_markdown_empty(self):
        b = HardwareBelief()
        md = b.to_markdown()
        assert "暂无" in md

    def test_belief_markdown_with_entries(self, mem):
        mem.record_belief("obs", "rule", 0.9)
        md = mem.belief.to_markdown()
        assert "规律" in md
        assert "90%" in md


class TestContextInjection:
    def test_build_context_empty(self, mem):
        ctx = mem.build_context()
        assert "暂无" in ctx

    def test_build_context_with_failures(self, mem):
        mem.record_failure("api_misuse", "mctlass shape error", "wrong shape", "fix")
        ctx = mem.build_context("mctlass")
        assert "已知陷阱" in ctx
        assert "mctlass" in ctx

    def test_build_context_with_belief(self, mem):
        mem.record_belief("obs", "rule", 0.7)
        ctx = mem.build_context()
        assert "Hardware Belief" in ctx

    def test_build_context_filters_relevant(self, mem):
        mem.record_failure("api_misuse", "mctlass error", "c1", "f1")
        mem.record_failure("boundary_error", "page overflow", "c2", "f2")
        ctx = mem.build_context("page")
        assert "page overflow" in ctx
        assert "mctlass error" not in ctx


class TestSerialization:
    def test_json_roundtrip(self, tmp_path):
        mem = DomainMemory(base_dir=tmp_path)
        mem.record_failure("api_misuse", "s", "r", "f", code_snippet="x=1", config="b=1")
        mem.record_belief("o", "rule", 0.5)
        mem.save()

        # 直接读 JSON 验证结构
        fj = json.loads((tmp_path / "failure_cases" / "cases.json").read_text())
        assert fj[0]["code_snippet"] == "x=1"
        bj = json.loads((tmp_path / "hardware_belief.json").read_text())
        assert bj["entries"][0]["rule"] == "rule"

    def test_code_snippet_and_config_optional(self, mem):
        case = mem.record_failure("c", "s", "r", "f")  # 不传可选字段
        assert case.code_snippet == ""
        assert case.config == ""
