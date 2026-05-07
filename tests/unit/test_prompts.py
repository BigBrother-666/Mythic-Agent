"""prompts 单元测试——保证关键约束词存在，避免无意识漂移。"""
from __future__ import annotations

from app.agent.prompts import GENERATOR_PROMPT, PLANNER_PROMPT, SYSTEM_PROMPT


def test_system_prompt_contains_key_rules() -> None:
    assert "优先参考" in SYSTEM_PROMPT
    assert "不得编造" in SYSTEM_PROMPT
    assert "ruamel" in SYSTEM_PROMPT or "yaml" in SYSTEM_PROMPT.lower()


def test_planner_prompt_outputs_json() -> None:
    assert "JSON" in PLANNER_PROMPT
    assert "intent" in PLANNER_PROMPT
    assert "search_queries" in PLANNER_PROMPT


def test_generator_prompt_has_sections() -> None:
    assert "## YAML" in GENERATOR_PROMPT
    assert "## 解释" in GENERATOR_PROMPT
    assert "## 建议" in GENERATOR_PROMPT
