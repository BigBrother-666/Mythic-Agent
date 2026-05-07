"""POST /validate。"""
from __future__ import annotations

from fastapi import APIRouter

from app.schemas.common import APIResponse
from app.schemas.generate import ValidateRequest, ValidateResponse
from app.services.generation_service import validate_only

router = APIRouter()


@router.post("/validate", response_model=APIResponse[ValidateResponse])
async def validate_endpoint(req: ValidateRequest) -> APIResponse[ValidateResponse]:
    """对外的纯校验接口。"""
    data = validate_only(req.yaml_text, kind=req.kind)
    return APIResponse.ok(data)
