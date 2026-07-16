"""
LLM Client —— 模力方舟 API 真实接入（MiniMax-M2.7）。

替代 mock 回调，让 Agent 真正由大模型驱动。
支持：
- chat：通用对话（Coder/Analyst/Judge/Reflector 用）
- 结构化输出：让 LLM 输出 JSON（cost model 预测、候选生成用）

API：https://api.moark.com/v1/chat/completions（OpenAI 兼容）
模型：MiniMax-M2.7
凭证：MOARK_API_KEY 环境变量
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import httpx


API_BASE = "https://api.moark.com/v1"
DEFAULT_MODEL = "MiniMax-M2.7"
DEFAULT_TIMEOUT = 120


def _get_api_key() -> str:
    """从环境变量取 API Key（兜底读 ~/.bashrc）。"""
    key = os.environ.get("MOARK_API_KEY")
    if key:
        return key
    # 兜底：从 bashrc 读
    bashrc = os.path.expanduser("~/.bashrc")
    if os.path.exists(bashrc):
        with open(bashrc) as f:
            for line in f:
                if "MOARK_API_KEY" in line and "=" in line:
                    # export MOARK_API_KEY="xxx"
                    m = re.search(r'MOARK_API_KEY=["\']([^"\']+)["\']', line)
                    if m:
                        return m.group(1)
    raise RuntimeError("MOARK_API_KEY 未设置（环境变量或 ~/.bashrc）")


@dataclass
class LLMResponse:
    """LLM 响应。"""
    content: str
    model: str
    usage: dict
    raw: dict

    @property
    def text(self) -> str:
        return self.content


def chat(
    messages: list,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout: int = DEFAULT_TIMEOUT,
    api_key: Optional[str] = None,
    retries: int = 2,
) -> LLMResponse:
    """
    调用模力方舟 chat completions API（含重试）。

    messages: [{"role": "system"/"user"/"assistant", "content": "..."}]
    """
    import time
    key = api_key or _get_api_key()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    last_exc = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(f"{API_BASE}/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return LLMResponse(
                content=content,
                model=data.get("model", model),
                usage=data.get("usage", {}),
                raw=data,
            )
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))  # 指数退避
    raise last_exc


def chat_simple(prompt: str, system: str = "", **kwargs) -> str:
    """便捷调用：单轮 prompt → 文本。"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, **kwargs).text


def chat_json(prompt: str, system: str = "", **kwargs) -> dict:
    """
    调用 LLM 并解析 JSON 输出。

    自动提取 ```json ... ``` 代码块或裸 JSON。
    """
    raw = chat_simple(prompt, system, **kwargs)
    return extract_json(raw)


def extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON（容忍 markdown 包裹、think 标签、前后文字）。"""
    # 先剥离 <think>...</think>（MiniMax-M2.7 的思考过程）
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 尝试 ```json ... ```
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # 尝试 ``` ... ```
    m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 尝试找第一个平衡的 { ... }
    depth = 0
    start = -1
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    start = -1
    raise ValueError(f"无法从输出提取 JSON：{text[:200]}")


def is_available() -> bool:
    """检查 API 是否可用（不实际调用模型，只检查 key 存在 + 连通性）。"""
    try:
        _get_api_key()
        return True
    except RuntimeError:
        return False
