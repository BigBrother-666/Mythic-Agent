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


def _detect_object_kind(body: Any) -> str:
    """根据单个对象的字段推断 kind。

    规则（按优先级）：
    - 含 `Material` 或 `Id` → item
    - 含 `Type` → mob
    - 含 `Skills` 但不含 `Type/Material/Id` → skill (metaskill)
    - 其余 → unknown
    """
    if not isinstance(body, dict):
        return "unknown"
    keys = set(body.keys())
    if "Material" in keys or "Id" in keys:
        return "item"
    if "Type" in keys:
        return "mob"
    if "Skills" in keys:
        return "skill"
    return "unknown"


def _detect_kind(data: Any) -> str:
    """从一份完整 YAML 推断「主体」kind（仅对外报告，不参与逐对象校验）。

    现在采用「多数投票」：扫描所有顶层对象、按各自 kind 计数，返回出现最多的 kind。
    这样 mob + 多个 metaskill 混写时主体仍报告 mob，不会因「第一个 value」误判。
    """
    if not isinstance(data, dict) or not data:
        return "unknown"
    counts: dict[str, int] = {}
    for body in data.values():
        k = _detect_object_kind(body)
        if k != "unknown":
            counts[k] = counts.get(k, 0) + 1
    if not counts:
        return "unknown"
    return max(counts, key=lambda k: counts[k])


def _validate_one(name: str, body: Any, kind: str) -> tuple[list[str], list[str]]:
    """校验单个顶层对象。kind 必须是 mob / item / skill / unknown 中的一个。"""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(body, dict):
        errors.append(f"对象 '{name}' 的值必须是 mapping，实际为 {type(body).__name__}")
        return errors, warnings

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
    else:
        # 无法识别的对象——只做最基本的废弃字段提示，不强制结构。
        warnings.append(
            f"对象 '{name}' 类型未识别（既无 Type/Material/Id/Skills），可能是配置片段或新字段"
        )

    for key in body.keys():
        if key in _DEPRECATED:
            warnings.append(f"'{name}.{key}': {_DEPRECATED[key]}")
        if known_keys and key not in known_keys:
            warnings.append(f"'{name}.{key}' 不在常见 {kind} 字段列表中（可能是新字段或拼写）")

    if kind == "mob":
        for list_key in ("AIGoalSelectors", "AITargetSelectors", "Skills", "Drops", "Equipment"):
            if list_key in body and not isinstance(body[list_key], list):
                errors.append(f"'{name}.{list_key}' 必须是列表")
    elif kind == "skill":
        if "Skills" in body and not isinstance(body["Skills"], list):
            errors.append(f"'{name}.Skills' 必须是列表")

    return errors, warnings


def _validate_structure(data: Any, primary_kind: str) -> tuple[list[str], list[str]]:
    """返回 (errors, warnings)。

    `primary_kind` 是调用方传入的「期望 kind」（来自 agent intent）。我们不强求所有对象都是这个
    kind——LLM 经常在同一份 YAML 里混写 mob 主体 + 多个 metaskill，这是合法且常见的。

    校验策略：
    1. 顶层必须是 mapping。
    2. 对每个顶层对象 **按它自己的形态独立判断 kind 并校验**。
    3. 如果一个对象的形态与 `primary_kind` 矛盾（例如 caller 说要校验 item，但里面有 Type=...），
       仅作 warning 不作 error——避免把合法的「mob + metaskill 一起写」误判为「mob 缺 Type」。
    4. `primary_kind == "auto"` 时，对每个对象都用其自身推断的 kind 校验。
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(data, dict):
        errors.append("顶层必须是 mapping（{name: {...}}），但实际是 " + type(data).__name__)
        return errors, warnings
    if not data:
        errors.append("YAML 内容为空")
        return errors, warnings

    for name, body in data.items():
        own_kind = _detect_object_kind(body)
        # 决定用哪个 kind 来校验这个对象：优先用对象自己的 kind；
        # 仅当对象自身无法识别 (unknown) 时才退回到 primary_kind。
        check_kind = own_kind if own_kind != "unknown" else primary_kind
        if check_kind == "auto":
            check_kind = "unknown"
        e, w = _validate_one(name, body, check_kind)
        errors.extend(e)
        warnings.extend(w)
        # primary_kind 与对象 kind 不一致时给一条信息性 warning（仅在 caller 显式指定 kind 时）
        if (
            primary_kind in {"mob", "item", "skill"}
            and own_kind in {"mob", "item", "skill"}
            and own_kind != primary_kind
        ):
            warnings.append(
                f"对象 '{name}' 看起来是 {own_kind}，与请求的 {primary_kind} 不同——"
                f"已按 {own_kind} 校验（这通常是 mob 与 metaskill 一起写的正常情况）"
            )

    return errors, warnings


def validate_yaml(yaml_text: str, kind: str = "auto") -> dict[str, Any]:
    """同步校验主函数（也供 service 层直接调用）。

    `kind` ∈ {"auto", "mob", "item", "skill"}：
    - `auto` —— 对每个顶层对象按自身形态独立校验
    - 其他 —— 把 `kind` 作为「主体类型」记录到返回值；但每个对象仍按自身形态校验，
              不会因为「混写 mob + metaskill」就报缺字段错误（这是 LLM 输出常见形态）。
    """
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
    errors, warnings = _validate_structure(data, primary_kind=kind)
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
