"""POST /rag/search —— 暴露底层 RAG 检索。"""
from __future__ import annotations

from fastapi import APIRouter

from app.rag.pipeline import search_wiki
from app.schemas.common import APIResponse
from app.schemas.rag import RAGHit, RAGSearchRequest, RAGSearchResponse

router = APIRouter()


@router.post("/rag/search", response_model=APIResponse[RAGSearchResponse])
async def rag_search(req: RAGSearchRequest) -> APIResponse[RAGSearchResponse]:
    """对外的 RAG 检索接口。"""
    hits = await search_wiki(req.query, top_k=req.top_k, category=req.category)
    return APIResponse.ok(
        RAGSearchResponse(
            query=req.query,
            hits=[
                RAGHit(
                    score=h.score,
                    text=h.text,
                    title=h.title,
                    source=h.source,
                    category=h.category,
                    tags=h.tags,
                )
                for h in hits
            ],
        )
    )
