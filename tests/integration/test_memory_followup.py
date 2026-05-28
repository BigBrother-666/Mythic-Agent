"""Memory + 多轮对话集成测试。"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from app.agent import graph as graph_module
from app.agent import memory as memory_module
from app.agent.memory import InMemoryBackend, MemoryStore
from app.services import chat_service


class _StubMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = []


class _RecordingLLM:
    """把每次 ainvoke 收到的 user prompt 文本完整记录下来，按预设响应轮转返回。"""

    def __init__(self, responses: list[str]) -> None:
        self.calls = 0
        self.responses = responses
        self.user_prompts: list[str] = []

    async def ainvoke(self, messages):  # type: ignore[no-untyped-def]
        try:
            self.user_prompts.append(str(messages[1].content))
        except Exception:  # noqa: BLE001
            self.user_prompts.append("")
        idx = self.calls
        self.calls += 1
        if idx >= len(self.responses):
            return _StubMessage(self.responses[-1])
        return _StubMessage(self.responses[idx])

    def bind_tools(self, tools):  # type: ignore[no-untyped-def]
        return self


async def _fake_search(q, top_k=4, category=None, wiki=None):  # type: ignore[no-untyped-def]
    return [
        {
            "score": 1.0,
            "text": "Skills: - projectile @target",
            "title": "Projectile",
            "source": "MythicMobs.wiki/Skills/Mechanics/projectile.md",
            "category": "mechanics",
            "tags": ["projectile"],
            "wiki": "mythicmobs",
        }
    ]


def _fresh_memory(monkeypatch) -> MemoryStore:
    """每个测试拿到全新的 MemoryStore（强制 InMemoryBackend），避免不同测试相互污染。"""
    store = MemoryStore(backend=InMemoryBackend())
    monkeypatch.setattr(memory_module, "_store", store)
    return store


def _reset_graph() -> None:
    graph_module._compiled = None


@pytest.mark.asyncio
async def test_first_turn_writes_memory_and_does_not_use_history(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """第一轮：history 为空 → planner prompt 里 context_block 应空；YAML 与回复入 memory。"""
    mem = _fresh_memory(monkeypatch)

    stub = _RecordingLLM(
        responses=[
            # planner JSON
            json.dumps({
                "intent": "skill",
                "search_queries": ["frost nova"],
                "name": "FrostNova",
                "notes": "area freeze",
                "is_followup": False,
            }),
            # generator
            "## YAML\n```yaml\n"
            "FrostNova:\n  Cooldown: 8\n  Skills:\n  - freeze{ticks=60} @EIR{r=5}\n"
            "```\n\n## 解释\n- Cooldown 防止过度施放\n\n## 建议\n- 加音效\n- 加视觉\n",
        ]
    )
    monkeypatch.setattr(graph_module, "get_llm", lambda: stub)
    with patch("app.agent.graph.wiki_search_dict", side_effect=_fake_search):
        _reset_graph()
        chunks = []
        async for raw in chat_service.chat_stream("写一个范围冻结技能", session_id="s1"):
            chunks.append(raw)

    # 第一轮的 planner prompt 不应出现「历史对话」「上一次生成的 YAML」字样
    planner_user = stub.user_prompts[0]
    assert "历史对话" not in planner_user
    assert "上一次生成的 YAML" not in planner_user

    # memory 内已写入 user + assistant + last_yaml
    s1 = await mem.get("s1")
    assert s1.last_yaml is not None and "FrostNova" in s1.last_yaml
    roles = [m["role"] for m in s1.history]
    assert roles == ["user", "assistant"]
    assert "Cooldown 防止过度施放" in s1.history[1]["content"]


@pytest.mark.asyncio
async def test_second_turn_injects_history_and_last_yaml_into_prompt(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """第二轮：planner 与 generator 的 prompt 必须看到上一轮的 YAML 与回复。"""
    mem = _fresh_memory(monkeypatch)
    # 预填 memory，模拟第一轮已经发生过
    await mem.append_message("s1", "user", "写一个范围冻结技能")
    await mem.append_message("s1", "assistant", "Cooldown 防止过度施放")
    await mem.set_last_yaml(
        "s1",
        "FrostNova:\n  Cooldown: 8\n  Skills:\n  - freeze{ticks=60} @EIR{r=5}\n",
    )

    stub = _RecordingLLM(
        responses=[
            # 第二轮 planner: 应判定为 followup 并复用 FrostNova 名字
            json.dumps({
                "intent": "skill",
                "search_queries": ["damage mechanic"],
                "name": "FrostNova",
                "notes": "在原 metaskill 上加伤害",
                "is_followup": True,
            }),
            # 第二轮 generator: 在原 YAML 基础上加 damage
            "## YAML\n```yaml\n"
            "FrostNova:\n  Cooldown: 8\n  Skills:\n"
            "  - freeze{ticks=60} @EIR{r=5}\n"
            "  - damage{a=8} @EIR{r=5}\n"
            "```\n\n## 解释\n- 新增 damage 机制\n\n## 建议\n- 调参\n- 加 cooldown\n",
        ]
    )
    monkeypatch.setattr(graph_module, "get_llm", lambda: stub)
    with patch("app.agent.graph.wiki_search_dict", side_effect=_fake_search):
        _reset_graph()
        async for _ in chat_service.chat_stream("再加一点伤害", session_id="s1"):
            pass

    # 关键断言 1：planner prompt 里包含 history + last_yaml 两个标记
    planner_user = stub.user_prompts[0]
    assert "历史对话" in planner_user
    assert "上一次生成的 YAML" in planner_user
    assert "FrostNova" in planner_user
    assert "freeze{ticks=60}" in planner_user

    # 关键断言 2：generator prompt 里 is_followup=true，并能看见旧 YAML
    generator_user = stub.user_prompts[1]
    assert "is_followup" in generator_user
    assert "true" in generator_user.lower()
    assert "FrostNova:\n  Cooldown: 8" in generator_user

    # 关键断言 3：本轮 user 消息也写进了 history（在调用 stream_events 之后）
    s1 = await mem.get("s1")
    user_msgs = [m for m in s1.history if m["role"] == "user"]
    assert any("再加一点伤害" in m["content"] for m in user_msgs)

    # 第二轮的产物覆盖了 last_yaml
    assert s1.last_yaml is not None and "damage{a=8}" in s1.last_yaml


@pytest.mark.asyncio
async def test_history_does_not_contain_current_user_message_during_planner(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """关键边界：第二轮的 planner prompt 中，history 只能含历史消息，不能把本轮 user 也算进去。"""
    mem = _fresh_memory(monkeypatch)
    await mem.append_message("s1", "user", "第一条问题")
    await mem.append_message("s1", "assistant", "第一条回复")
    await mem.set_last_yaml("s1", "Foo:\n  Type: ZOMBIE\n")

    stub = _RecordingLLM(
        responses=[
            json.dumps({"intent": "chat", "search_queries": [], "name": "", "is_followup": False}),
            "好的，我已知道你的偏好",
        ]
    )
    monkeypatch.setattr(graph_module, "get_llm", lambda: stub)
    with patch("app.agent.graph.wiki_search_dict", side_effect=_fake_search):
        _reset_graph()
        async for _ in chat_service.chat_stream("第二条问题", session_id="s1"):
            pass

    planner_user = stub.user_prompts[0]
    assert "第一条问题" in planner_user  # history 含
    assert "第一条回复" in planner_user
    # 但本轮的 user message 应该出现在「用户输入：」段，而不是出现在 history block 里两次
    assert planner_user.count("第二条问题") == 1


@pytest.mark.asyncio
async def test_no_session_id_does_not_touch_memory(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """没传 session_id 时，memory 不应被写入。"""
    mem = _fresh_memory(monkeypatch)

    stub = _RecordingLLM(
        responses=[
            json.dumps({"intent": "chat", "search_queries": [], "name": "", "is_followup": False}),
            "你好",
        ]
    )
    monkeypatch.setattr(graph_module, "get_llm", lambda: stub)
    with patch("app.agent.graph.wiki_search_dict", side_effect=_fake_search):
        _reset_graph()
        async for _ in chat_service.chat_stream("hi", session_id=None):
            pass

    assert await mem.backend.size() == 0  # 完全不写
