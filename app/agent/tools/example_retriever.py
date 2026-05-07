"""ExampleRetrieverTool —— 在 RAG 上加 category=examples 过滤，返回示例 YAML。"""
from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.rag.pipeline import search_wiki


class ExampleRetrieverInput(BaseModel):
    """工具输入。"""

    query: str = Field(..., description="想找的示例描述")
    top_k: int = Field(default=4, ge=1, le=10)


@tool("example_retriever", args_schema=ExampleRetrieverInput)
async def example_retriever(query: str, top_k: int = 4) -> str:
    """检索官方/社区示例配置。"""
    hits = await search_wiki(query, top_k=top_k, category="examples")
    if not hits:
        # fallback：放宽到 mobs 目录
        hits = await search_wiki(query, top_k=top_k, category="mobs")
    if not hits:
        return "(no examples found)"
    return "\n\n---\n\n".join(
        f"[{i}] {h.source} ({h.title})\n{h.text}" for i, h in enumerate(hits, 1)
    )
