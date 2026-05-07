"""生成服务：把 GenerateRequest 包装成 Agent 输入并组装响应。"""
from __future__ import annotations

from typing import Literal

from app.agent.graph import run_agent
from app.agent.tools import format_yaml, validate_yaml
from app.core.exceptions import ValidationError
from app.schemas.generate import GenerateRequest, GenerateResponse, ValidateResponse


def _build_user_input(req: GenerateRequest, kind: Literal["mob", "item", "skill"]) -> str:
    """把结构化请求拼成自然语言 prompt。"""
    name = req.name or {"mob": "MyCustomMob", "item": "MyCustomItem", "skill": "MyCustomSkill"}[kind]
    constraint_text = ""
    if req.constraints:
        constraint_text = "\n硬性约束:\n- " + "\n- ".join(req.constraints)
    return (
        f"请生成一个 MythicMobs {kind}。名称: {name}。\n"
        f"需求描述:\n{req.description}{constraint_text}"
    )


async def generate_object(
    req: GenerateRequest,
    kind: Literal["mob", "item", "skill"],
) -> GenerateResponse:
    """主生成函数。"""
    user_input = _build_user_input(req, kind)
    state = await run_agent(user_input)
    yaml_text = state.get("yaml_text") or ""
    name = state.get("name") or req.name or "MyCustom"
    sources = [r["source"] for r in (state.get("retrieved") or [])][:5]
    return GenerateResponse(
        kind=kind,
        name=name,
        yaml=yaml_text,
        explanation=state.get("explanation") or "",
        suggestions=state.get("suggestions") or [],
        sources=sources,
    )


def validate_only(yaml_text: str, kind: str = "auto") -> ValidateResponse:
    """仅做校验与格式化。"""
    res = validate_yaml(yaml_text, kind=kind)
    formatted: str | None = None
    if res["valid"]:
        try:
            formatted = format_yaml(yaml_text)
        except Exception as e:  # noqa: BLE001
            raise ValidationError(f"格式化失败: {e}") from e
    return ValidateResponse(
        valid=res["valid"],
        errors=res["errors"],
        warnings=res["warnings"],
        formatted=formatted,
    )
