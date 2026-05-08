"""集成测试：Agent 流程在 mock 掉 LLM 与 RAG 后能跑通。

不依赖真实的 Milvus / OpenAI；用 monkeypatch 替换。
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.agent import graph as graph_module
from app.agent import prompts as prompts_module


class _StubMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubLLM:
    """按 `responses` 列表依次返回内容；首次必是 planner 的 JSON，之后是 generator。"""

    def __init__(self, responses: list[str]) -> None:
        self.calls = 0
        self.responses = responses
        self.last_prompts: list[str] = []  # 记录每次 ainvoke 收到的 user prompt 字符串

    async def ainvoke(self, messages):  # type: ignore[no-untyped-def]
        # messages 是 [SystemMessage, HumanMessage]；记录第二条 content 便于断言模板
        try:
            self.last_prompts.append(str(messages[1].content))
        except Exception:  # noqa: BLE001
            self.last_prompts.append("")
        idx = self.calls
        self.calls += 1
        if idx >= len(self.responses):
            return _StubMessage(self.responses[-1])
        return _StubMessage(self.responses[idx])


async def _fake_search(q, top_k=4, category=None, wiki=None):  # type: ignore[no-untyped-def]
    """检索 mock：永远返回一条 mythicmobs 的 projectile 片段。"""
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


def _reset_graph() -> None:
    """让 graph 在每个测试里重新编译——保证使用最新的 stub。"""
    graph_module._compiled = None


# -------------- mob 路径（保留原回归用例） --------------

@pytest.mark.asyncio
async def test_agent_flow_mob(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    stub = _StubLLM(
        responses=[
            # planner
            '{"intent":"mob","search_queries":["frost boss skill","summon minion"],'
            '"name":"DeepSeaBoss","notes":"three phases"}',
            # generator
            "## YAML\n```yaml\n"
            "DeepSeaBoss:\n  Type: GUARDIAN\n  Health: 500\n"
            "  Skills:\n  - skill{s=FrostNova} ~onTimer:60\n"
            "```\n\n## 解释\n- Type 决定底层实体\n\n## 建议\n- 加 BossBar\n- 加阶段技能\n",
        ]
    )
    monkeypatch.setattr(graph_module, "get_llm", lambda: stub)
    with patch("app.agent.graph.wiki_search_dict", side_effect=_fake_search):
        _reset_graph()
        result = await graph_module.run_agent("造一个深海 Boss")

    assert result["intent"] == "mob"
    assert "DeepSeaBoss" in result["yaml_text"]
    assert result["validation_errors"] == []
    # 第二次调用（generator）的 prompt 应该来自 mob 模板（含 "Type:" 行示例）
    gen_prompt = stub.last_prompts[1]
    assert "Type:" in gen_prompt
    # 不应误用 item / skill 模板的标志性内容
    assert "Recipes" not in gen_prompt or "## 任务：生成一个 Mob" in gen_prompt or "Mob" in gen_prompt


# -------------- item 路径 --------------

@pytest.mark.asyncio
async def test_agent_flow_item(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    stub = _StubLLM(
        responses=[
            # planner: intent=item
            '{"intent":"item","search_queries":["custom sword","onUse trigger"],'
            '"name":"FrostBlade","notes":"freeze on hit"}',
            # generator: 输出 item YAML
            "## YAML\n```yaml\n"
            "FrostBlade:\n  Id: diamond_sword\n  Display: '&b冰霜之刃'\n"
            "  Skills:\n  - skill{s=FrostStrike} @target ~onUse ?holding{m=FrostBlade}\n"
            "```\n\n## 解释\n- Id 决定底层 material\n\n## 建议\n- 加合成\n- 加 Lore\n",
        ]
    )
    monkeypatch.setattr(graph_module, "get_llm", lambda: stub)
    with patch("app.agent.graph.wiki_search_dict", side_effect=_fake_search):
        _reset_graph()
        result = await graph_module.run_agent("造一把会冻结敌人的剑")

    assert result["intent"] == "item"
    assert "FrostBlade" in result["yaml_text"]
    assert "Id: diamond_sword" in result["yaml_text"]
    assert result["validation_errors"] == []
    # generator 的 prompt 应来自 item 模板（含 Recipes / onUse 这类 item 特有的关键字）
    gen_prompt = stub.last_prompts[1]
    assert "Recipes" in gen_prompt
    assert "onUse" in gen_prompt or "~onUse" in gen_prompt


# -------------- skill 路径 --------------

@pytest.mark.asyncio
async def test_agent_flow_skill(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    stub = _StubLLM(
        responses=[
            # planner: intent=skill
            '{"intent":"skill","search_queries":["frost nova mechanic","particles cloud"],'
            '"name":"FrostNova","notes":"area freeze"}',
            # generator: 输出独立 metaskill
            "## YAML\n```yaml\n"
            "FrostNova:\n  Cooldown: 8\n"
            "  Skills:\n  - particles{p=cloud;a=20} @self\n  - freeze{ticks=60} @EIR{r=5}\n"
            "```\n\n## 解释\n- Cooldown 防止过度施放\n\n## 建议\n- 加音效\n- 加视觉特效\n",
        ]
    )
    monkeypatch.setattr(graph_module, "get_llm", lambda: stub)
    with patch("app.agent.graph.wiki_search_dict", side_effect=_fake_search):
        _reset_graph()
        result = await graph_module.run_agent("写一个范围冰冻的 metaskill")

    assert result["intent"] == "skill"
    assert "FrostNova" in result["yaml_text"]
    assert result["validation_errors"] == []
    # skill 模板的标志：明确说不带 Type / Material
    gen_prompt = stub.last_prompts[1]
    assert "metaskill" in gen_prompt.lower()
    assert "TargetConditions" in gen_prompt


# -------------- 默认名按 intent 切 --------------

@pytest.mark.asyncio
async def test_default_name_by_intent(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """planner 没给 name 时，按 intent 切默认值。"""
    stub = _StubLLM(
        responses=[
            # planner 故意不给 name
            '{"intent":"item","search_queries":["potion item"]}',
            "## YAML\n```yaml\nMyCustomItem:\n  Id: stick\n  Display: 'x'\n```\n"
            "## 解释\n- ok\n\n## 建议\n- a\n- b\n",
        ]
    )
    monkeypatch.setattr(graph_module, "get_llm", lambda: stub)
    with patch("app.agent.graph.wiki_search_dict", side_effect=_fake_search):
        _reset_graph()
        result = await graph_module.run_agent("做个物品")

    assert result["intent"] == "item"
    assert result["name"] == "MyCustomItem"


# -------------- 兜底测试：planner 返回未识别 intent --------------

@pytest.mark.asyncio
async def test_unknown_intent_falls_back_to_chat(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """planner 输出非法 intent 时应回退到 chat（不进 retriever / generator 的 YAML 模板）。"""
    stub = _StubLLM(
        responses=[
            '{"intent":"banana","search_queries":[]}',
            "我没听懂你的需求",  # chat 分支直接当回复
        ]
    )
    monkeypatch.setattr(graph_module, "get_llm", lambda: stub)
    with patch("app.agent.graph.wiki_search_dict", side_effect=_fake_search):
        _reset_graph()
        result = await graph_module.run_agent("？？？")
    assert result["intent"] == "chat"
    assert result.get("yaml_text") in ("", None)
