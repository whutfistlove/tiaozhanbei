"""
llm_client 的单元测试。

测试策略：
- extract_json：纯字符串解析，不调 API（快）
- is_available：检查 key 存在
- 真实 API 调用：标记 slow，手动触发

运行：pytest tests/test_llm_client.py -v
"""
import json
import pytest

from agent_system.llm_client import extract_json, is_available, chat_simple, chat_json


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_markdown_wrapped(self):
        text = '结果：\n```json\n{"speedup": 2.0}\n```\n完成'
        assert extract_json(text) == {"speedup": 2.0}

    def test_code_block_no_lang(self):
        text = '```\n{"x": 1}\n```'
        assert extract_json(text) == {"x": 1}

    def test_with_think_tag(self):
        """MiniMax-M2.7 带 <think> 标签"""
        text = '<think>let me think</think>\n{"speedup": 1.5, "confidence": 0.8}'
        result = extract_json(text)
        assert result["speedup"] == 1.5

    def test_json_with_surrounding_text(self):
        text = 'Here is the result: {"a": 1, "b": 2} thanks'
        assert extract_json(text) == {"a": 1, "b": 2}

    def test_nested_json(self):
        text = 'output: {"outer": {"inner": 1}}'
        assert extract_json(text)["outer"]["inner"] == 1

    def test_multiple_braces(self):
        """多个花括号时应取平衡的第一个"""
        text = '{"a": {"b": 1}} and {"c": 2}'
        result = extract_json(text)
        assert "a" in result

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            extract_json("no json here")

    def test_think_then_markdown(self):
        text = '<think>分析中</think>\n```json\n{"ok": true}\n```'
        assert extract_json(text) == {"ok": True}


class TestAvailability:
    def test_is_available_returns_bool(self):
        assert isinstance(is_available(), bool)


@pytest.mark.slow
@pytest.mark.skipif(not is_available(), reason="MOARK_API_KEY 未设置")
class TestRealAPICall:
    """真实 API 调用（手动 --runslow 触发，消耗 token）。"""

    def test_simple_chat(self):
        r = chat_simple("回复数字1", temperature=0.1, max_tokens=10)
        assert isinstance(r, str)
        assert "1" in r

    def test_json_chat(self):
        r = chat_json('只返回这个JSON，不要其它内容：{"status": "ok", "value": 42}',
                      max_tokens=50)
        assert r["status"] == "ok"
        assert r["value"] == 42
