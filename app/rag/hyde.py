"""HyDE (Hypothetical Document Embeddings) —— 用 LLM 生成假设性文档来增强检索。

原理：短查询的 embedding 往往不够精确（如"freeze"可能匹配到很多无关内容）。
HyDE 让 LLM 先生成一段"假设性的 wiki 文档片段"来回答查询，
再用这段文档的 embedding 去检索——因为文档与文档之间的语义距离比查询与文档更近。

启用条件：ENABLE_HYDE=true。会增加一次 LLM 调用，换取更精准的向量检索。
"""
from __future__ import annotations

import asyncio

from app.core.config import get_settings
from app.core.logging import logger

_HYDE_PROMPT = """\
你是 MythicMobs Wiki 的作者。根据下面的用户查询，写一段简短的 wiki 文档片段（100-200字）来回答这个问题。
直接输出文档内容，不要加任何前缀或解释。用与查询相同的语言回答。

用户查询：{query}"""


async def generate_hypothetical_document(query: str) -> str:
    """调用 LLM 生成假设性文档；失败时返回原始 query。"""
    settings = get_settings()

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.hyde_model or settings.llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.7,
            max_tokens=settings.hyde_max_tokens,
        )
        prompt = _HYDE_PROMPT.format(query=query)
        response = await llm.ainvoke(prompt)
        doc = response.content.strip()
        if doc:
            logger.debug("HyDE generated {} chars for query: {}", len(doc), query[:60])
            return doc
    except Exception as e:  # noqa: BLE001
        logger.warning("HyDE generation failed; fallback to raw query: {}", e)

    return query
