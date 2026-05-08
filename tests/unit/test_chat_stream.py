"""chat_service 流式 token 透传测试。

直接 mock `app.agent.graph.stream_events`，验证 chat_service:
- generator 节点的 LLM token 被包成 type=token 的 SSE chunk 实时透出
- token 的 meta.node 字段携带节点名（前端用来做缓冲切换）
- planner 节点的 token 不被透传
- 最后会发送 type=done
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator
from unittest.mock import patch

import pytest


class _FakeChunk:
    """模拟 langchain_core 的 AIMessageChunk —— 只用到 .content 字段。"""

    def __init__(self, content: str) -> None:
        self.content = content


async def _fake_stream(_user_input: str, **_kwargs) -> AsyncIterator[tuple[str, Any]]:
    """伪造 stream_events 输出：依次发出 plan / retrieval / token×3(generator) /
    token×1(planner=应该被过滤) / yaml / validation / 最后什么都不发。"""
    yield "updates", {
        "planner": {
            "intent": "mob",
            "search_queries": ["frost boss"],
            "name": "DeepSeaBoss",
        }
    }
    yield "updates", {
        "retriever": {
            "retrieved": [{"source": "Skills/Mechanics/freeze.md", "title": "Freeze"}]
        }
    }
    # planner 节点的 token，应该被过滤掉
    yield "messages", (_FakeChunk('{"intent":'), {"langgraph_node": "planner"})
    # generator 节点的 token，应该被透传
    yield "messages", (_FakeChunk("## YAML\n"), {"langgraph_node": "generator"})
    yield "messages", (_FakeChunk("```yaml\nDeepSeaBoss:\n"), {"langgraph_node": "generator"})
    yield "messages", (_FakeChunk("  Type: GUARDIAN\n```\n"), {"langgraph_node": "generator"})
    # generator 节点完成后的 yaml 提取
    yield "updates", {
        "generator": {
            "yaml_text": "DeepSeaBoss:\n  Type: GUARDIAN\n",
            "raw_output": "## YAML\n```yaml\nDeepSeaBoss:\n  Type: GUARDIAN\n```\n",
        }
    }
    yield "updates", {
        "validator": {
            "yaml_text": "DeepSeaBoss:\n  Type: GUARDIAN\n",
            "validation_errors": [],
            "validation_warnings": [],
        }
    }


def _parse_sse(line: str) -> dict:
    """把 'data: {...}\n\n' 解出 dict。"""
    assert line.startswith("data: ")
    return json.loads(line[len("data: "):].strip())


@pytest.mark.asyncio
async def test_chat_stream_emits_tokens_incrementally() -> None:
    from app.services import chat_service

    with patch.object(chat_service, "stream_events", _fake_stream):
        chunks = []
        async for raw in chat_service.chat_stream("生成深海 boss"):
            chunks.append(_parse_sse(raw))

    types_in_order = [c["type"] for c in chunks]

    # plan / retrieval 在最前
    assert types_in_order[0] == "plan"
    assert types_in_order[1] == "retrieval"

    # token 事件来自 generator，且按顺序累加
    tokens = [c for c in chunks if c["type"] == "token"]
    assert len(tokens) == 3, f"expect 3 generator tokens, got {len(tokens)}: {tokens}"
    assert all(t["meta"]["node"] == "generator" for t in tokens)
    assert tokens[0]["content"] == "## YAML\n"
    assert tokens[1]["content"].startswith("```yaml")
    assert tokens[2]["content"].endswith("```\n")

    # planner 的 JSON token 必须被过滤
    assert all('"intent"' not in t["content"] for t in tokens)

    # 至少有一次 yaml 与一次 validation
    assert any(c["type"] == "yaml" for c in chunks)
    assert any(c["type"] == "validation" for c in chunks)

    # 末尾必为 done
    assert types_in_order[-1] == "done"


@pytest.mark.asyncio
async def test_chat_stream_token_before_yaml() -> None:
    """关键属性：第一个 token 必须早于该节点结束的 yaml 事件。"""
    from app.services import chat_service

    with patch.object(chat_service, "stream_events", _fake_stream):
        first_token_idx = None
        first_yaml_idx = None
        idx = 0
        async for raw in chat_service.chat_stream("x"):
            ev = _parse_sse(raw)
            if ev["type"] == "token" and first_token_idx is None:
                first_token_idx = idx
            if ev["type"] == "yaml" and first_yaml_idx is None:
                first_yaml_idx = idx
            idx += 1

    assert first_token_idx is not None
    assert first_yaml_idx is not None
    assert first_token_idx < first_yaml_idx, (
        "token 必须在 yaml 之前到达——这是流式体验的核心约束"
    )
