"""POST /generate/{mob,item,skill}。"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Path

from app.schemas.common import APIResponse
from app.schemas.generate import GenerateRequest, GenerateResponse
from app.services.generation_service import generate_object

router = APIRouter()


@router.post("/generate/{kind}", response_model=APIResponse[GenerateResponse])
async def generate(
    req: GenerateRequest,
    kind: Literal["mob", "item", "skill"] = Path(...),
) -> APIResponse[GenerateResponse]:
    """统一生成接口。"""
    data = await generate_object(req, kind=kind)
    return APIResponse.ok(data)
