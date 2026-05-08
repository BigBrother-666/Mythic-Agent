"""Hybrid Retriever：向量检索 + BM25 关键词检索，使用 RRF 融合。"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.core.logging import logger
from app.rag.embeddings import get_embedding_service
from app.rag.vector_store import VectorHit, get_milvus_store


@dataclass
class FusedHit:
    """融合后的命中项。"""

    score: float
    text: str
    title: str
    source: str
    category: str
    tags: list[str]
    wiki: str = "mythicmobs"


def _tokenize(text: str) -> list[str]:
    """简单的中英文混合分词：英文按 alnum 切，中文逐字。"""
    out: list[str] = []
    buf: list[str] = []
    for ch in text.lower():
        if ch.isalnum() and ord(ch) < 128:
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
            if ch.strip() and ord(ch) >= 0x4E00:
                out.append(ch)
    if buf:
        out.append("".join(buf))
    return out


class HybridRetriever:
    """向量 + BM25 混合检索。"""

    _instance: "HybridRetriever | None" = None

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._corpus_hits: list[VectorHit] = []
        self._lock = asyncio.Lock()

    @classmethod
    def instance(cls) -> "HybridRetriever":
        """全局单例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def warmup(self) -> None:
        """预热 BM25 索引（FastAPI lifespan 调用）。"""
        async with self._lock:
            if self._bm25 is not None:
                return
            store = get_milvus_store()
            if not store.has_collection():
                logger.warning("Milvus collection missing; BM25 not built")
                return
            try:
                hits = await store.fetch_all_for_bm25()
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to fetch corpus for BM25: {}", e)
                return
            if not hits:
                logger.warning("Empty corpus; BM25 not built")
                return
            tokenized = [_tokenize(h.title + " " + h.text) for h in hits]
            self._bm25 = BM25Okapi(tokenized)
            self._corpus_hits = hits
            logger.info("BM25 index built over {} chunks", len(hits))

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """BM25 检索，返回 (corpus_index, score)。"""
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, float(s)) for i, s in ranked[:top_k] if s > 0]

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        category: str | None = None,
        wiki: str | None = None,
    ) -> list[FusedHit]:
        """对外检索接口；返回 RRF 融合后的命中。

        wiki ∈ {"mythicmobs","crucible","local",None}。None 表示不过滤（默认）。
        """
        settings = get_settings()
        if top_k is None:
            top_k = settings.rag_top_k

        embedder = get_embedding_service()
        store = get_milvus_store()

        vector = await embedder.embed_query(query)
        vec_hits = await store.search(vector, top_k=top_k * 3, category=category, wiki=wiki)

        bm25_pairs = self._bm25_search(query, top_k=top_k * 2)

        # ---------- RRF 融合 ----------
        rrf_k = 60
        scores: dict[str, float] = {}
        keep: dict[str, VectorHit] = {}

        for rank, h in enumerate(vec_hits):
            key = f"{h.source}::{h.title}::{h.text[:64]}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            keep.setdefault(key, h)

        for rank, (idx, _) in enumerate(bm25_pairs):
            if idx >= len(self._corpus_hits):
                continue
            h = self._corpus_hits[idx]
            if category and h.category != category:
                continue
            if wiki and h.wiki != wiki:
                continue
            key = f"{h.source}::{h.title}::{h.text[:64]}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            keep.setdefault(key, h)

        ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]

        fused = [
            FusedHit(
                score=scores[k],
                text=keep[k].text,
                title=keep[k].title,
                source=keep[k].source,
                category=keep[k].category,
                tags=keep[k].tags,
                wiki=keep[k].wiki,
            )
            for k in ranked_keys
        ]

        if settings.enable_rerank:
            fused = await self._rerank(query, fused)

        return fused

    async def _rerank(self, query: str, hits: list[FusedHit]) -> list[FusedHit]:
        """Rerank 钩子；当前是 noop，留给后续接 bge-reranker-large。"""
        # 这里保持接口稳定；真正实现时引入 sentence-transformers CrossEncoder。
        _ = query
        return hits


def get_retriever() -> HybridRetriever:
    """全局 retriever。"""
    return HybridRetriever.instance()
