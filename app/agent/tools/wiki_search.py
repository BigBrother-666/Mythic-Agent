"""WikiSearchTool —— 调 RAG 检索 MythicMobs Wiki。"""
from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.rag.pipeline import search_wiki


class WikiSearchInput(BaseModel):
    """工具输入。"""

    query: str = Field(..., description="自然语言查询，可中可英")
    top_k: int = Field(default=6, ge=1, le=20)
    category: str | None = Field(default=None, description="可选过滤：mechanics/mobs/items/...")


@tool("wiki_search", args_schema=WikiSearchInput)
async def wiki_search(query: str, top_k: int = 6, category: str | None = None) -> str:
    """检索 MythicMobs Wiki，返回与查询最相关的若干文档片段。"""
    hits = await search_wiki(query, top_k=top_k, category=category)
    if not hits:
        return "(no results)"
    parts: list[str] = []
    for i, h in enumerate(hits, 1):
        parts.append(
            f"[{i}] source={h.source} | title={h.title} | category={h.category} "
            f"| score={h.score:.3f}\n{h.text}"
        )
    return "\n\n---\n\n".join(parts)


async def wiki_search_dict(query: str, top_k: int = 6, category: str | None = None) -> list[dict]:
    """非 LangChain 调用方式；service 层用。"""
    hits = await search_wiki(query, top_k=top_k, category=category)
    return [
        {
            "score": h.score,
            "text": h.text,
            "title": h.title,
            "source": h.source,
            "category": h.category,
            "tags": h.tags,
        }
        for h in hits
    ]
