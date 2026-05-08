"""GET /health。"""
from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.rag.vector_store import get_milvus_store
from app.schemas.common import APIResponse

router = APIRouter()


@router.get("/health", response_model=APIResponse[dict])
async def health() -> APIResponse[dict]:
    """简单健康检查；尝试 ping milvus。"""
    settings = get_settings()
    milvus_ok = False
    detail = ""
    try:
        store = get_milvus_store()
        milvus_ok = store.has_collection()
    except Exception as e:  # noqa: BLE001
        detail = str(e)
    return APIResponse.ok(
        {
            "milvus": milvus_ok,
            "milvus_collection": settings.milvus_collection,
            "embed_model": settings.embed_model,
            "llm_model": settings.llm_model,
            "rerank": {
                "enabled": settings.enable_rerank,
                "model": settings.rerank_model if settings.enable_rerank else None,
                "pool_factor": settings.rerank_pool_factor if settings.enable_rerank else None,
            },
            "detail": detail,
        }
    )
