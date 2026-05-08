"""prompts 单元测试——保证关键约束词存在，避免无意识漂移。"""
from __future__ import annotations

from app.agent.prompts import (
    GENERATOR_PROMPT,
    ITEM_GENERATOR_PROMPT,
    MOB_GENERATOR_PROMPT,
    PLANNER_PROMPT,
    SKILL_GENERATOR_PROMPT,
    SYSTEM_PROMPT,
    get_generator_prompt,
)


def test_system_prompt_contains_key_rules() -> None:
    assert "优先参考" in SYSTEM_PROMPT
    assert "不得编造" in SYSTEM_PROMPT
    assert "ruamel" in SYSTEM_PROMPT or "yaml" in SYSTEM_PROMPT.lower()
    # 现在 system prompt 必须解释 mythicmobs vs crucible 的区别
    assert "crucible" in SYSTEM_PROMPT.lower()


def test_planner_prompt_outputs_json() -> None:
    assert "JSON" in PLANNER_PROMPT
    assert "intent" in PLANNER_PROMPT
    assert "search_queries" in PLANNER_PROMPT
    # 新版 planner 必须列出 4 类 intent 的判定规则
    for kind in ("mob", "item", "skill", "chat"):
        assert kind in PLANNER_PROMPT


def test_each_generator_prompt_has_three_sections() -> None:
    for name, prompt in [
        ("mob", MOB_GENERATOR_PROMPT),
        ("item", ITEM_GENERATOR_PROMPT),
        ("skill", SKILL_GENERATOR_PROMPT),
    ]:
        assert "## YAML" in prompt, f"{name} prompt missing ## YAML"
        assert "## 解释" in prompt, f"{name} prompt missing ## 解释"
        assert "## 建议" in prompt, f"{name} prompt missing ## 建议"


def test_item_prompt_mentions_crucible_triggers() -> None:
    """物品 prompt 必须提到 Crucible 与 ~onUse / ~onLeftclick 这类触发器。"""
    assert "crucible" in ITEM_GENERATOR_PROMPT.lower()
    assert "~onUse" in ITEM_GENERATOR_PROMPT or "onUse" in ITEM_GENERATOR_PROMPT
    assert "Recipes" in ITEM_GENERATOR_PROMPT


def test_skill_prompt_does_not_force_type_or_material() -> None:
    """skill 模板应明确「不带 Type / Material」。"""
    assert "不带 Type" in SKILL_GENERATOR_PROMPT or "Material / Id" in SKILL_GENERATOR_PROMPT


def test_get_generator_prompt_dispatches() -> None:
    assert get_generator_prompt("mob") is MOB_GENERATOR_PROMPT
    assert get_generator_prompt("item") is ITEM_GENERATOR_PROMPT
    assert get_generator_prompt("skill") is SKILL_GENERATOR_PROMPT
    # 未知 intent 兜底到 mob
    assert get_generator_prompt("anything-else") is MOB_GENERATOR_PROMPT


def test_legacy_generator_prompt_alias() -> None:
    """老代码 / 老测试还在用 GENERATOR_PROMPT；它现在是 mob 模板的别名。"""
    assert GENERATOR_PROMPT is MOB_GENERATOR_PROMPT
