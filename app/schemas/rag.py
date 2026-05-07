"""RAG 检索接口 schema。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RAGSearchRequest(BaseModel):
    """RAG 检索请求体。"""

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=30)
    category: str | None = Field(default=None, description="可选分类过滤，如 'mechanics'")


class RAGHit(BaseModel):
    """单条命中结果。"""

    score: float
    text: str
    title: str
    source: str
    category: str
    tags: list[str] = Field(default_factory=list)


class RAGSearchResponse(BaseModel):
    """RAG 检索响应。"""

    hits: list[RAGHit]
    query: str
