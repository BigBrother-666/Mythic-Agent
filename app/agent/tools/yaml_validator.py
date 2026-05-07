"""YAMLValidatorTool —— 校验 MythicMobs YAML 的格式与结构。"""
from __future__ import annotations

import io
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

# 已知顶层字段（经常出现的；非穷尽）
_KNOWN_MOB_KEYS = {
    "Type",
    "Display",
    "Health",
    "Damage",
    "Armor",
    "Faction",
    "Disguise",
    "Mount",
    "Options",
    "Modules",
    "AIGoalSelectors",
    "AITargetSelectors",
    "Skills",
    "Drops",
    "Equipment",
    "DamageModifiers",
    "Variables",
    "KillMessages",
    "BossBar",
    "ImmunityTables",
    "LevelModifiers",
    "Scoreboard",
    "Trades",
    "Threat",
    "PowerLevel",
}

_KNOWN_ITEM_KEYS = {
    "Id",
    "Material",
    "Display",
    "Lore",
    "Amount",
    "CustomModelData",
    "Attributes",
    "Enchantments",
    "Hide",
    "Options",
    "Skills",
    "Group",
    "PotionEffects",
}

_KNOWN_SKILL_KEYS = {
    "Skills",
    "Conditions",
    "TargetConditions",
    "TriggerConditions",
    "Cooldown",
    "OnCooldownSkill",
    "FailedConditionsSkill",
    "CancelIfNoTargets",
}

# 已知废弃 / 旧字段；提示替换
_DEPRECATED = {
    "MaxHealth": "请改用 Health（MythicMobs 5.x 起）",
    "Mana": "5.x 不再使用 Mana 字段；请用 Variables 自管",
}


class YAMLValidateInput(BaseModel):
    """校验工具输入。"""

    yaml_text: str = Field(..., description="MythicMobs YAML 字符串")
    kind: str = Field(default="auto", description="mob/item/skill/auto；auto 会自动推断")


def _detect_kind(data: Any) -> str:
    """根据顶层字段推断对象类型。"""
    if not isinstance(data, dict) or not data:
        return "unknown"
    first_value = next(iter(data.values()))
    if not isinstance(first_value, dict):
        return "unknown"
    keys = set(first_value.keys())
    if "Material" in keys:
        return "item"
    if "Type" in keys:
        return "mob"
    if "Skills" in keys and "Type" not in keys and "Material" not in keys:
        return "skill"
    return "unknown"


def _validate_structure(data: Any, kind: str) -> tuple[list[str], list[str]]:
    """返回 (errors, warnings)。"""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(data, dict):
        errors.append("顶层必须是 mapping（{name: {...}}），但实际是 " + type(data).__name__)
        return errors, warnings
    if not data:
        errors.append("YAML 内容为空")
        return errors, warnings
    for name, body in data.items():
        if not isinstance(body, dict):
            errors.append(f"对象 '{name}' 的值必须是 mapping，实际为 {type(body).__name__}")
            continue

        known_keys: set[str] = set()
        if kind == "mob":
            known_keys = _KNOWN_MOB_KEYS
            if "Type" not in body:
                errors.append(f"Mob '{name}' 缺少必填字段 Type")
        elif kind == "item":
            known_keys = _KNOWN_ITEM_KEYS
            if "Material" not in body and "Id" not in body:
                errors.append(f"Item '{name}' 必须至少有 Material 或 Id")
        elif kind == "skill":
            known_keys = _KNOWN_SKILL_KEYS
            if "Skills" not in body:
                warnings.append(f"Skill '{name}' 没有 Skills 列表")

        for key in body.keys():
            if key in _DEPRECATED:
                warnings.append(f"'{name}.{key}': {_DEPRECATED[key]}")
            if known_keys and key not in known_keys:
                # 仅警告，不报错——MythicMobs 的字段集很大且持续扩展
                warnings.append(f"'{name}.{key}' 不在常见 {kind} 字段列表中（可能是新字段或拼写）")

        if kind == "mob":
            for list_key in ("AIGoalSelectors", "AITargetSelectors", "Skills", "Drops", "Equipment"):
                if list_key in body and not isinstance(body[list_key], list):
                    errors.append(f"'{name}.{list_key}' 必须是列表")
    return errors, warnings


def validate_yaml(yaml_text: str, kind: str = "auto") -> dict[str, Any]:
    """同步校验主函数（也供 service 层直接调用）。"""
    parser = YAML(typ="safe")
    try:
        data = parser.load(io.StringIO(yaml_text))
    except YAMLError as e:
        return {
            "valid": False,
            "errors": [f"YAML 解析失败: {e}"],
            "warnings": [],
            "kind": "unknown",
            "data": None,
        }

    detected = _detect_kind(data) if kind == "auto" else kind
    errors, warnings = _validate_structure(data, detected)
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "kind": detected,
        "data": data,
    }


@tool("yaml_validator", args_schema=YAMLValidateInput)
def yaml_validator_tool(yaml_text: str, kind: str = "auto") -> str:
    """校验 MythicMobs YAML；返回多行文本格式的报告。"""
    res = validate_yaml(yaml_text, kind=kind)
    lines = [
        f"valid: {res['valid']}",
        f"kind: {res['kind']}",
    ]
    if res["errors"]:
        lines.append("errors:")
        lines.extend(f"  - {e}" for e in res["errors"])
    if res["warnings"]:
        lines.append("warnings:")
        lines.extend(f"  - {w}" for w in res["warnings"])
    return "\n".join(lines)
