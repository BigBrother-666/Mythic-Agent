"""生成接口 schema。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """生成请求体——三种类型 (mob/item/skill) 复用同一结构。"""

    description: str = Field(..., min_length=1, max_length=8000, description="自然语言需求描述")
    name: str | None = Field(default=None, description="可选的对象名（生成的 YAML 顶层 key）")
    constraints: list[str] = Field(default_factory=list, description="额外硬性约束条件")


class GenerateResponse(BaseModel):
    """生成结果。"""

    kind: Literal["mob", "item", "skill"]
    name: str
    yaml: str
    explanation: str
    suggestions: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class ValidateRequest(BaseModel):
    """YAML 校验请求体。"""

    yaml_text: str = Field(..., min_length=1, max_length=20000)
    kind: Literal["mob", "item", "skill", "auto"] = "auto"


class ValidateResponse(BaseModel):
    """YAML 校验结果。"""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    formatted: str | None = None
