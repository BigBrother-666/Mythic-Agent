"""Milvus 向量库封装；负责 collection 创建、写入、检索。

`pymilvus` 通过惰性 import 引入，避免单元测试时强制安装。
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings
from app.core.logging import logger
from app.rag.chunker import Chunk


@dataclass
class VectorHit:
    """检索命中项；与底层 Milvus 解耦。"""

    score: float
    text: str
    title: str
    source: str
    category: str
    tags: list[str]
    wiki: str = "mythicmobs"


class MilvusStore:
    """对 MilvusClient 的薄封装。"""

    _instance: "MilvusStore | None" = None

    def __init__(self) -> None:
        from pymilvus import MilvusClient

        settings = get_settings()
        self._collection = settings.milvus_collection
        self._dim = settings.embed_dim
        logger.info("Connecting Milvus at {}", settings.milvus_uri)
        self._client = MilvusClient(uri=settings.milvus_uri)

    @classmethod
    def instance(cls) -> "MilvusStore":
        """获取全局 Milvus store 单例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def collection_name(self) -> str:
        return self._collection

    def has_collection(self) -> bool:
        """判断 collection 是否存在。"""
        return self._client.has_collection(self._collection)

    def create_collection(self, drop_existing: bool = False) -> None:
        """按预定义 schema 创建 collection。"""
        from pymilvus import DataType

        if self._client.has_collection(self._collection):
            if not drop_existing:
                logger.info("Collection {} already exists", self._collection)
                return
            logger.warning("Dropping existing collection {}", self._collection)
            self._client.drop_collection(self._collection)

        schema = self._client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self._dim)
        # Milvus VARCHAR 的 max_length 按 UTF-8 字节算（不是字符）。中文 3 byte/字符，
        # 32000 字节大约能装 10000+ 汉字或更多英文，对单 chunk 完全够用。
        schema.add_field("text", DataType.VARCHAR, max_length=32000)
        schema.add_field("title", DataType.VARCHAR, max_length=512)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("category", DataType.VARCHAR, max_length=64)
        schema.add_field("tags", DataType.VARCHAR, max_length=2048)
        schema.add_field("has_yaml", DataType.BOOL)
        schema.add_field("wiki", DataType.VARCHAR, max_length=32)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="IP",
        )
        self._client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Created collection {} (dim={})", self._collection, self._dim)

    def drop(self) -> None:
        """删除 collection。"""
        if self._client.has_collection(self._collection):
            self._client.drop_collection(self._collection)
            logger.info("Dropped collection {}", self._collection)

    @staticmethod
    def _truncate_bytes(value: str, max_bytes: int) -> str:
        """按 UTF-8 字节安全截断（不切坏多字节字符）。

        Milvus VARCHAR 的 max_length 是字节数。直接 `s[:n]` 按字符切，遇到中文会膨胀 3 倍。
        这里用 `encode('utf-8')[:n]` 截字节、再 `decode(errors='ignore')` 去掉被切坏的尾部。
        """
        encoded = value.encode("utf-8")
        if len(encoded) <= max_bytes:
            return value
        return encoded[:max_bytes].decode("utf-8", errors="ignore")

    def _to_row(self, chunk: Chunk, vector: list[float]) -> dict[str, Any]:
        meta = chunk.metadata
        return {
            "embedding": vector,
            "text": self._truncate_bytes(chunk.text, 32000),
            "title": self._truncate_bytes(str(meta.get("title", "")), 512),
            "source": self._truncate_bytes(str(meta.get("source", "")), 512),
            "category": self._truncate_bytes(str(meta.get("category", "")), 64),
            "tags": self._truncate_bytes(
                json.dumps(meta.get("tags", []), ensure_ascii=False), 2048
            ),
            "has_yaml": bool(meta.get("has_yaml", False)),
            "wiki": self._truncate_bytes(str(meta.get("wiki", "mythicmobs")), 32),
        }

    async def insert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """批量插入 chunks。"""
        rows = [self._to_row(c, v) for c, v in zip(chunks, vectors, strict=True)]
        result = await asyncio.to_thread(
            self._client.insert,
            collection_name=self._collection,
            data=rows,
        )
        count = int(result.get("insert_count", len(rows)))
        return count

    async def search(
        self,
        vector: list[float],
        top_k: int = 8,
        category: str | None = None,
        wiki: str | None = None,
    ) -> list[VectorHit]:
        """按向量检索；可选 category / wiki 过滤。"""
        clauses: list[str] = []
        if category:
            clauses.append(f'category == "{category}"')
        if wiki:
            clauses.append(f'wiki == "{wiki}"')
        filter_expr = " and ".join(clauses)
        result = await asyncio.to_thread(
            self._client.search,
            collection_name=self._collection,
            data=[vector],
            limit=top_k,
            output_fields=["text", "title", "source", "category", "tags", "wiki"],
            filter=filter_expr,
        )
        hits: list[VectorHit] = []
        for row in result[0]:
            entity = row.get("entity", {})
            try:
                tags = json.loads(entity.get("tags") or "[]")
            except json.JSONDecodeError:
                tags = []
            hits.append(
                VectorHit(
                    score=float(row.get("distance", 0.0)),
                    text=entity.get("text", ""),
                    title=entity.get("title", ""),
                    source=entity.get("source", ""),
                    category=entity.get("category", ""),
                    tags=tags,
                    wiki=entity.get("wiki", "mythicmobs"),
                )
            )
        return hits

    async def fetch_all_for_bm25(self, batch: int = 1000) -> list[VectorHit]:
        """把整个 collection 的文本拉到内存里，用于 BM25 索引。

        Wiki 规模约 10k chunks，几十 MB；可接受。生产环境如果文档更大需要换成分布式实现。
        """
        out: list[VectorHit] = []
        cursor = 0
        while True:
            res = await asyncio.to_thread(
                self._client.query,
                collection_name=self._collection,
                filter="",
                output_fields=["text", "title", "source", "category", "tags", "wiki"],
                limit=batch,
                offset=cursor,
            )
            if not res:
                break
            for entity in res:
                try:
                    tags = json.loads(entity.get("tags") or "[]")
                except json.JSONDecodeError:
                    tags = []
                out.append(
                    VectorHit(
                        score=0.0,
                        text=entity.get("text", ""),
                        title=entity.get("title", ""),
                        source=entity.get("source", ""),
                        category=entity.get("category", ""),
                        tags=tags,
                        wiki=entity.get("wiki", "mythicmobs"),
                    )
                )
            if len(res) < batch:
                break
            cursor += batch
        return out

    async def count(self) -> int:
        """估算 collection 中的实体数。"""
        res = await asyncio.to_thread(
            self._client.get_collection_stats,
            collection_name=self._collection,
        )
        return int(res.get("row_count", 0))


def get_milvus_store() -> MilvusStore:
    """获取全局 Milvus store 单例。"""
    return MilvusStore.instance()
