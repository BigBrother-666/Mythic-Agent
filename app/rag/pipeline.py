"""RAG ingest / query 协调入口。

支持多源入库：
- `<root>/MythicMobs.wiki/**/*.md`   → wiki=mythicmobs
- `<root>/mythiccrucible.wiki/**/*.md` → wiki=crucible
- `<project_root>/examples/**/*.yml`  → wiki=local（以 YAML 文件粒度切）

入库时 source 路径**包含 wiki 子目录前缀**（如 `MythicMobs.wiki/Skills/Mechanics/projectile.md`），
以便 chunker 能据此推断 `wiki` metadata。
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import logger
from app.rag.chunker import (
    Chunk,
    chunk_markdown,
    chunk_yaml_file,
    iter_wiki_files,
    iter_yaml_files,
)
from app.rag.embeddings import get_embedding_service
from app.rag.retriever import FusedHit, get_retriever
from app.rag.vector_store import get_milvus_store


def _read_text(path: Path) -> str:
    """容错读取文件文本。"""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _collect_chunks_for_wiki(wiki_dir: Path, wiki_label: str) -> list[Chunk]:
    """扫描一个 wiki 子目录，切分所有 markdown 为 chunks。

    `wiki_label` 仅用于日志；真正写入 metadata 的 wiki 字段由 chunker 从 source 路径推断。
    """
    if not wiki_dir.exists():
        logger.warning("wiki dir not found: {}", wiki_dir)
        return []
    files = iter_wiki_files(wiki_dir)
    logger.info("[{}] discovered {} markdown files", wiki_label, len(files))
    chunks: list[Chunk] = []
    for f in files:
        text = _read_text(f)
        # source 路径形如 "MythicMobs.wiki/Skills/Mechanics/projectile.md"
        rel = wiki_dir.name + "/" + str(f.relative_to(wiki_dir))
        chunks.extend(chunk_markdown(text, source=rel))
    logger.info("[{}] chunked into {} pieces", wiki_label, len(chunks))
    return chunks


def _collect_chunks_for_examples(examples_dir: Path) -> list[Chunk]:
    """扫描 examples 目录下的 .yml/.yaml，每个文件切成 chunks（按顶层 key）。"""
    if not examples_dir.exists():
        logger.info("examples dir not found, skip: {}", examples_dir)
        return []
    files = iter_yaml_files(examples_dir)
    logger.info("[examples] discovered {} yaml files", len(files))
    chunks: list[Chunk] = []
    for f in files:
        text = _read_text(f)
        rel = "examples/" + str(f.relative_to(examples_dir))
        chunks.extend(chunk_yaml_file(text, source=rel))
    logger.info("[examples] chunked into {} pieces", len(chunks))
    return chunks


async def ingest_wiki(
    wiki_root: Path,
    *,
    drop_existing: bool = False,
    batch_size: int = 64,
    progress_every: int = 10,
    examples_dir: Path | None = None,
) -> dict[str, int]:
    """把 wiki 与 examples 切分 → embedding → 写入 Milvus。

    - `wiki_root`：包含 `MythicMobs.wiki/` 和（可选）`mythiccrucible.wiki/` 的父目录。
      也兼容旧调用：直接传 `MythicMobs.wiki` 单一目录。
    - `examples_dir`：本地 YAML 示例目录；不传则跳过。
    """
    store = get_milvus_store()
    embedder = get_embedding_service()
    store.create_collection(drop_existing=drop_existing)

    all_chunks: list[Chunk] = []

    # 模式一：wiki_root 直接是个 wiki 子目录（旧用法）
    if (wiki_root / "Mobs").is_dir() or (wiki_root / "Skills").is_dir():
        logger.info("Detected single-wiki layout at {}", wiki_root)
        all_chunks.extend(_collect_chunks_for_wiki(wiki_root, wiki_root.name))
    else:
        # 模式二：wiki_root 是父目录，下面有 MythicMobs.wiki / mythiccrucible.wiki
        for sub_name, label in (
            ("MythicMobs.wiki", "mythicmobs"),
            ("mythiccrucible.wiki", "crucible"),
        ):
            sub_dir = wiki_root / sub_name
            all_chunks.extend(_collect_chunks_for_wiki(sub_dir, label))

    if examples_dir is not None:
        all_chunks.extend(_collect_chunks_for_examples(examples_dir))

    if not all_chunks:
        logger.warning("No chunks produced; nothing to ingest")
        return {"chunks": 0, "written": 0}

    written = 0
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        vectors = await embedder.embed_documents([c.text for c in batch])
        await store.insert(batch, vectors)
        written += len(batch)
        if (i // batch_size) % progress_every == 0:
            logger.info("Embedded & inserted {}/{}", written, len(all_chunks))
    logger.info("Ingestion done: {} chunks", written)

    # 简单按 wiki 维度统计
    by_wiki: dict[str, int] = {}
    for c in all_chunks:
        w = str(c.metadata.get("wiki", "unknown"))
        by_wiki[w] = by_wiki.get(w, 0) + 1
    logger.info("Chunks by wiki: {}", by_wiki)

    return {"chunks": len(all_chunks), "written": written, **{f"wiki.{k}": v for k, v in by_wiki.items()}}


async def search_wiki(
    query: str,
    top_k: int | None = None,
    category: str | None = None,
    wiki: str | None = None,
    tags: list[str] | None = None,
) -> list[FusedHit]:
    """高层检索入口。"""
    retriever = get_retriever()
    return await retriever.search(query, top_k=top_k, category=category, wiki=wiki, tags=tags)


async def search_by_tags(tags: list[str], top_k: int = 4) -> list[FusedHit]:
    """按 tags 标量过滤检索示例配置。"""
    from app.rag.vector_store import VectorHit

    store = get_milvus_store()
    hits = await store.query_by_tags(tags, top_k=top_k)
    return [
        FusedHit(
            score=h.score,
            text=h.text,
            title=h.title,
            source=h.source,
            category=h.category,
            tags=h.tags,
        )
        for h in hits
    ]


def get_last_hyde_doc() -> str:
    """获取最近一次检索中 HyDE 生成的假设文档（未启用时为空字符串）。"""
    retriever = get_retriever()
    return getattr(retriever, "last_hyde_doc", "")


async def warmup() -> None:
    """预热 retriever（构建 BM25 索引）。"""
    retriever = get_retriever()
    await retriever.warmup()


__all__ = ["ingest_wiki", "search_wiki", "search_by_tags", "warmup"]


async def main_cli() -> None:
    """脚本入口；scripts/ingest.py 复用。"""
    import argparse

    from app.core.logging import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki", type=Path, default=None, help="Wiki 父目录路径（包含 MythicMobs.wiki 等）")
    parser.add_argument(
        "--examples",
        type=Path,
        default=None,
        help="examples 目录路径（默认 <项目根>/examples，不存在则跳过）",
    )
    parser.add_argument("--drop", action="store_true", help="删除已有 collection 重建")
    args = parser.parse_args()

    settings = get_settings()
    wiki_root = args.wiki or settings.wiki_root

    examples_dir = args.examples
    if examples_dir is None:
        # 默认尝试容器内 /examples（docker-compose 会把项目 examples 挂进去），其次 ./examples
        for candidate in (Path("/examples"), Path("examples")):
            if candidate.exists():
                examples_dir = candidate
                break

    logger.info("Ingest config: wiki_root={}, examples_dir={}, drop={}", wiki_root, examples_dir, args.drop)
    result = await ingest_wiki(wiki_root, drop_existing=args.drop, examples_dir=examples_dir)
    logger.info("Result: {}", result)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main_cli())
