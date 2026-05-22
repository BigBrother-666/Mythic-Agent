"""Embedding 模型封装；带 Redis 缓存避免重复计算。

底层 HF embedding 库通过惰性 import 引入，避免在仅测试 chunker / validator 等纯逻辑时被强制安装。
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from app.core.cache import get_cache
from app.core.config import get_settings
from app.core.logging import logger


class EmbeddingService:
    """同步 HF embedding 模型 + 异步缓存层。"""

    _instance: "EmbeddingService | None" = None

    def __init__(self) -> None:
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings

        settings = get_settings()
        self._model_name = settings.embed_model
        self._dim = settings.embed_dim
        self._device = settings.embed_device
        self._batch_size = settings.embed_batch_size
        logger.info("Loading embedding model {} on {}", self._model_name, self._device)
        self._embedder = HuggingFaceBgeEmbeddings(
            model_name=self._model_name,
            model_kwargs={"device": self._device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": self._batch_size},
        )

    @classmethod
    def instance(cls) -> "EmbeddingService":
        """获取全局单例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def dim(self) -> int:
        """向量维度。"""
        return self._dim

    @property
    def model_name(self) -> str:
        """模型名（用作缓存 key 的一部分）。"""
        return self._model_name

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha1(f"{self._model_name}::{text}".encode()).hexdigest()
        return f"emb:{digest}"

    async def embed_query(self, text: str) -> list[float]:
        """对单条查询进行 embedding，带缓存。"""
        cache = await get_cache()
        key = self._cache_key(text)
        cached = await cache.get(key)
        if cached is not None:
            try:
                import orjson

                return orjson.loads(cached)
            except Exception:  # noqa: BLE001
                pass
        vec = await asyncio.to_thread(self._embedder.embed_query, text)
        try:
            import orjson

            await cache.set(key, orjson.dumps(vec).decode(), ttl_seconds=86400)
        except Exception:  # noqa: BLE001
            logger.debug("Cache write failed; ignored")
        return vec

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量 embedding（不走缓存——批量入库时数据本来就是新的）。"""
        return await asyncio.to_thread(self._embedder.embed_documents, texts)

    def to_langchain(self) -> Any:
        """暴露底层 LangChain Embeddings，给 LlamaIndex / 其他工具复用。"""
        return self._embedder


def get_embedding_service() -> EmbeddingService:
    """获取全局 embedding service。"""
    return EmbeddingService.instance()
