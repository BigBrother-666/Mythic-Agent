"""RAG ingest / query 协调入口。"""
from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.logging import logger
from app.rag.chunker import Chunk, chunk_markdown, iter_wiki_files
from app.rag.embeddings import get_embedding_service
from app.rag.retriever import FusedHit, get_retriever
from app.rag.vector_store import get_milvus_store


async def ingest_wiki(
    wiki_root: Path,
    *,
    drop_existing: bool = False,
    batch_size: int = 64,
    progress_every: int = 10,
) -> dict[str, int]:
    """把整个 wiki 切分 → embedding → 写入 Milvus。"""
    store = get_milvus_store()
    embedder = get_embedding_service()
    store.create_collection(drop_existing=drop_existing)

    files = iter_wiki_files(wiki_root)
    logger.info("Discovered {} markdown files under {}", len(files), wiki_root)

    all_chunks: list[Chunk] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = f.read_text(encoding="latin-1")
        rel = str(f.relative_to(wiki_root))
        chunks = chunk_markdown(text, source=rel)
        all_chunks.extend(chunks)
    logger.info("Chunked into {} pieces", len(all_chunks))

    written = 0
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        vectors = await embedder.embed_documents([c.text for c in batch])
        await store.insert(batch, vectors)
        written += len(batch)
        if (i // batch_size) % progress_every == 0:
            logger.info("Embedded & inserted {}/{}", written, len(all_chunks))
    logger.info("Ingestion done: {} chunks", written)
    return {"files": len(files), "chunks": len(all_chunks), "written": written}


async def search_wiki(
    query: str,
    top_k: int | None = None,
    category: str | None = None,
) -> list[FusedHit]:
    """高层检索入口。"""
    retriever = get_retriever()
    return await retriever.search(query, top_k=top_k, category=category)


async def warmup() -> None:
    """预热 retriever（构建 BM25 索引）。"""
    retriever = get_retriever()
    await retriever.warmup()


__all__ = ["ingest_wiki", "search_wiki", "warmup"]


async def main_cli() -> None:
    """脚本入口；scripts/ingest.py 复用。"""
    import argparse

    from app.core.config import get_settings
    from app.core.logging import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki", type=Path, default=None, help="Wiki 目录路径")
    parser.add_argument("--drop", action="store_true", help="删除已有 collection 重建")
    args = parser.parse_args()
    settings = get_settings()
    wiki_root = args.wiki or settings.wiki_root
    result = await ingest_wiki(wiki_root, drop_existing=args.drop)
    logger.info("Result: {}", result)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main_cli())
