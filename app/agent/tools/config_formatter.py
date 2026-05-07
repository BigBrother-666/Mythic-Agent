"""ConfigFormatterTool —— 统一 YAML 风格、修复缩进、排序顶层字段。"""
from __future__ import annotations

import io
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from ruamel.yaml import YAML

# 顶层字段排序优先级（MythicMobs 社区常见约定）
_MOB_FIELD_ORDER = [
    "Type",
    "Display",
    "Health",
    "Damage",
    "Armor",
    "PowerLevel",
    "Faction",
    "Disguise",
    "Mount",
    "Options",
    "Modules",
    "AIGoalSelectors",
    "AITargetSelectors",
    "Equipment",
    "DamageModifiers",
    "ImmunityTables",
    "LevelModifiers",
    "Variables",
    "Skills",
    "Drops",
    "Trades",
    "KillMessages",
    "BossBar",
]


class ConfigFormatterInput(BaseModel):
    """工具输入。"""

    yaml_text: str = Field(...)


def _sort_fields(value: Any) -> Any:
    """递归排序顶层 mob 字段。"""
    if not isinstance(value, dict):
        return value
    ordered: dict[str, Any] = {}
    remaining = dict(value)
    for k in _MOB_FIELD_ORDER:
        if k in remaining:
            ordered[k] = remaining.pop(k)
    for k in sorted(remaining.keys()):
        ordered[k] = remaining[k]
    return ordered


def format_yaml(yaml_text: str) -> str:
    """格式化 YAML 字符串：缩进 2、排序顶层字段。"""
    parser = YAML()
    parser.indent(mapping=2, sequence=4, offset=2)
    parser.preserve_quotes = True
    data = parser.load(io.StringIO(yaml_text))
    if isinstance(data, dict):
        new = {}
        for k, v in data.items():
            new[k] = _sort_fields(v)
        data = new
    out = io.StringIO()
    parser.dump(data, out)
    return out.getvalue().rstrip() + "\n"


@tool("config_formatter", args_schema=ConfigFormatterInput)
def config_formatter_tool(yaml_text: str) -> str:
    """格式化 MythicMobs YAML。"""
    try:
        return format_yaml(yaml_text)
    except Exception as e:  # noqa: BLE001
        return f"格式化失败: {e}\n原文:\n{yaml_text}"
