"""Hybrid Retriever：向量检索 + BM25 关键词检索，使用 RRF 融合。"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.core.logging import logger
from app.rag.embeddings import get_embedding_service
from app.rag.vector_store import VectorHit, get_milvus_store

# jieba 静默模式，避免启动时输出日志到 stderr
jieba.setLogLevel(jieba.logging.INFO)

_STOPWORDS_DIR = Path(__file__).parent / "stopwords"


def _load_stopwords() -> frozenset[str]:
    """从 stopwords/ 目录加载所有 txt 文件。"""
    words: set[str] = set()
    for f in _STOPWORDS_DIR.glob("*.txt"):
        for line in f.read_text(encoding="utf-8").splitlines():
            w = line.strip()
            if w:
                words.add(w)
    return frozenset(words)


_STOPWORDS = _load_stopwords()

_MYTHIC_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*")


def _tokenize(text: str) -> list[str]:
    """中英文混合分词：中文用 jieba 切词，英文按字母数字边界切。

    针对 MythicMobs 领域优化：
    - 保留 camelCase / snake_case 标识符完整（如 projectile、onDamaged、EIR）
    - 中文用 jieba 词级分词而非逐字，提升 BM25 区分度
    - 使用 hit/baidu/scu/cn 四份停用词表过滤噪音
    """
    text = text.lower()
    out: list[str] = []

    # 先提取英文/数字 token（含下划线，保留 MythicMobs 标识符完整性）
    en_tokens = _MYTHIC_PATTERN.findall(text)
    for tok in en_tokens:
        if tok not in _STOPWORDS and len(tok) > 1:
            out.append(tok)

    # 提取中文片段并用 jieba 分词
    zh_segments = re.findall(r"[\u4e00-\u9fff]+", text)
    for seg in zh_segments:
        for word in jieba.cut(seg):
            if word not in _STOPWORDS and len(word) > 1:
                out.append(word)

    return out


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
        """对外检索接口；返回 RRF 融合后（如启用则再 rerank）的命中。

        wiki ∈ {"mythicmobs","crucible","local",None}。None 表示不过滤（默认）。

        启用 HyDE 时流程：
            query → LLM 生成假设性文档 → embedding → 向量检索
        启用 rerank 时流程：
            vector top-(k*pool_factor*3) ∪ BM25 top-(k*pool_factor*2)
              → RRF 融合 → 取前 k*pool_factor 个候选
              → CrossEncoder 重排序 → 截到 top_k
        """
        settings = get_settings()
        if top_k is None:
            top_k = settings.rag_top_k

        # rerank 启用时取更大的候选池，否则旧逻辑保持 top_k * 3 / top_k * 2 不变
        pool_factor = settings.rerank_pool_factor if settings.enable_rerank else 1
        candidate_size = top_k * max(pool_factor, 1)

        embedder = get_embedding_service()
        store = get_milvus_store()

        # HyDE：用 LLM 生成假设性文档，用其 embedding 替代原始 query 做向量检索
        embed_text = query
        self.last_hyde_doc = ""
        if settings.enable_hyde:
            from app.rag.hyde import generate_hypothetical_document

            embed_text = await generate_hypothetical_document(query)
            self.last_hyde_doc = embed_text

        vector = await embedder.embed_query(embed_text)
        vec_hits = await store.search(
            vector, top_k=candidate_size * 3, category=category, wiki=wiki
        )

        # BM25 始终用原始 query（关键词匹配不需要扩展）
        bm25_pairs = self._bm25_search(query, top_k=candidate_size * 2)

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

        # 取候选池
        ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

        # Source 级去重：同一 source 最多保留 max_per_source 条，避免大文档霸占结果
        max_per_source = settings.rag_max_per_source
        source_count: dict[str, int] = {}
        pool_keys: list[str] = []
        for k in ranked_keys:
            src = keep[k].source
            if "changelog" in src.lower():
                continue
            cnt = source_count.get(src, 0)
            if cnt >= max_per_source:
                continue
            source_count[src] = cnt + 1
            pool_keys.append(k)
            if len(pool_keys) >= candidate_size:
                break

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
            for k in pool_keys
        ]

        if settings.enable_rerank and fused:
            fused = await self._rerank(query, fused)

        return fused[:top_k]

    async def _rerank(self, query: str, hits: list[FusedHit]) -> list[FusedHit]:
        """用 CrossEncoder 重排序；失败时退回 RRF 顺序，不影响主流程可用性。"""
        from app.rag.rerank import get_rerank_service

        try:
            service = get_rerank_service()
            # rerank 时把 title 拼到 text 前面，让模型看到层级语义
            payloads = [f"{h.title}\n\n{h.text}" if h.title else h.text for h in hits]
            scores = await service.score(query, payloads)
        except Exception as e:  # noqa: BLE001
            logger.warning("Rerank failed; fallback to RRF order: {}", e)
            return hits

        if len(scores) != len(hits):
            logger.warning(
                "Rerank score count mismatch ({} vs {}); fallback", len(scores), len(hits)
            )
            return hits

        scored = list(zip(hits, scores, strict=True))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        # 用 rerank 分数覆盖 FusedHit.score，方便上游观察
        return [
            FusedHit(
                score=float(s),
                text=h.text,
                title=h.title,
                source=h.source,
                category=h.category,
                tags=h.tags,
                wiki=h.wiki,
            )
            for h, s in scored
        ]


def get_retriever() -> HybridRetriever:
    """全局 retriever。"""
    return HybridRetriever.instance()
