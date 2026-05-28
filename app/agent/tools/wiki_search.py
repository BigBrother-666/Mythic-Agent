"""WikiSearchTool —— 调 RAG 检索 MythicMobs Wiki，支持 tags 预过滤。"""
from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.rag.pipeline import search_wiki


class WikiSearchInput(BaseModel):
    """工具输入。"""

    query: str = Field(..., description="自然语言查询，可中可英")
    top_k: int = Field(default=6, ge=1, le=20)
    tags: list[str] = Field(
        default_factory=list,
        description="MythicMobs 标识符列表，用于预过滤。如 ['projectile', 'orbital']",
    )


@tool("wiki_search", args_schema=WikiSearchInput)
async def wiki_search(query: str, top_k: int = 6, tags: list[str] | None = None) -> str:
    """用自然语言检索 MythicMobs/Crucible Wiki 文档。

    参数说明：
    - query: 自然语言描述你想查找的内容，如 "projectile 的 highAccuracyMode 参数"
    - tags: 可选的 MythicMobs 标识符列表，用于缩小检索范围，可以是任何mechanics/targeters/conditions/triggers标识，比如Projectile、PlayersInRing、altitude、onAttack等（大小写不敏感）。不确定时可以不传，直接用 query 检索。
      传入 tags 时，向量检索会限定在包含这些 tag 的文档范围内，
      相当于"在这些 tag 相关的文档中做语义搜索"。
      例如：tags=["Projectile", "Orbital"] + query="如何设置弹道速度"
    - top_k: 返回的最大文档数

    使用场景：需要了解某个功能的语法、属性、用法说明。
    """
    hits = await search_wiki(query, top_k=top_k, tags=tags or None)
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
    """非 LangChain 调用方式；retriever_node / service 层用。"""
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
