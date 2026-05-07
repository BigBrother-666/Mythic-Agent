"""集成测试：Agent 流程在 mock 掉 LLM 与 RAG 后能跑通。

不依赖真实的 Milvus / OpenAI；用 monkeypatch 替换。
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.agent import graph as graph_module


class _StubMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubLLM:
    """两次调用：第一次 planner，第二次 generator。"""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            # planner: 输出 JSON
            return _StubMessage(
                '{"intent":"mob","search_queries":["frost boss skill","summon minion"],'
                '"name":"DeepSeaBoss","notes":"three phases"}'
            )
        # generator: 输出标准三段
        return _StubMessage(
            "## YAML\n"
            "```yaml\n"
            "DeepSeaBoss:\n"
            "  Type: GUARDIAN\n"
            "  Health: 500\n"
            "  Skills:\n"
            "  - skill{s=FrostNova} ~onTimer:60\n"
            "```\n\n"
            "## 解释\n- Type 决定底层实体；GUARDIAN 适合水下 Boss\n"
            "- Health 500 提供较高血量\n\n"
            "## 建议\n- 可以加入三阶段血量阈值技能\n- 加 BossBar 增强体验\n"
        )


@pytest.mark.asyncio
async def test_agent_flow_mocked(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """端到端：Agent 跑完后产出合法 YAML。"""
    stub = _StubLLM()
    monkeypatch.setattr(graph_module, "get_llm", lambda: stub)

    async def fake_search(q, top_k=4):  # type: ignore[no-untyped-def]
        return [
            {
                "score": 1.0,
                "text": "Skills: - projectile @target",
                "title": "Projectile",
                "source": "Skills/Mechanics/projectile.md",
                "category": "mechanics",
                "tags": ["projectile"],
            }
        ]

    with patch("app.agent.graph.wiki_search_dict", side_effect=fake_search):
        # 强制重建 graph，确保使用 stub
        graph_module._compiled = None
        result = await graph_module.run_agent(
            "帮我设计一个深海 Boss，有三阶段、冰冻技能、召唤小怪、狂暴机制"
        )

    assert result.get("intent") == "mob"
    assert "DeepSeaBoss" in (result.get("yaml_text") or "")
    assert result.get("validation_errors") == []
    assert result.get("retrieved")  # 检索结果非空
    assert any(s for s in (result.get("suggestions") or []))
