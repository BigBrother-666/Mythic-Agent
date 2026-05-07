"""VersionCompatibilityTool —— 简单的版本兼容性检查。"""
from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# (废弃字段, 替代提示, 引入版本)
_DEPRECATIONS = {
    "MaxHealth": "改用 Health（5.x+）",
    "Mana": "5.x 不再支持；请用 Variables 自实现",
    "Difficulty": "5.x 已移除",
    "Glow": "请使用 Options.Glowing",
}

# (字段, 最低需要版本)
_FEATURE_VERSION = {
    "Modules": "5.0",
    "MannequinOptions": "5.5",
    "Mannequin": "5.5",
}


class VersionCompatInput(BaseModel):
    """工具输入。"""

    yaml_text: str = Field(...)
    target_version: str = Field(default="5.6", description="目标 MythicMobs 主版本")


@tool("version_compat", args_schema=VersionCompatInput)
def version_compat_tool(yaml_text: str, target_version: str = "5.6") -> str:
    """检查 YAML 中是否使用了与目标版本不兼容的字段。"""
    issues: list[str] = []
    for keyword, hint in _DEPRECATIONS.items():
        if keyword in yaml_text:
            issues.append(f"[deprecated] 出现 '{keyword}'：{hint}")
    for keyword, min_ver in _FEATURE_VERSION.items():
        if keyword in yaml_text and target_version < min_ver:
            issues.append(f"[version] '{keyword}' 需要 MythicMobs ≥ {min_ver}（目标 {target_version}）")
    if not issues:
        return f"OK：与 {target_version} 兼容"
    return "\n".join(issues)
