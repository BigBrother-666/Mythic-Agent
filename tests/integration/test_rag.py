"""RAG chunker + Milvus path 集成测试（mock 掉 Milvus）。"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.rag.chunker import chunk_markdown


@pytest.mark.asyncio
async def test_search_pipeline_with_mocks() -> None:
    """search_wiki 在 mock 掉 store 与 embedder 时仍能跑。"""
    from app.rag import retriever as retriever_module
    from app.rag.retriever import HybridRetriever
    from app.rag.vector_store import VectorHit

    fake_hits = [
        VectorHit(score=0.9, text="frost mechanic", title="Freeze", source="Skills/Mechanics/freeze.md", category="mechanics", tags=[]),
    ]

    HybridRetriever._instance = None
    retriever = HybridRetriever.instance()

    class FakeStore:
        def has_collection(self) -> bool:
            return True

        async def search(self, vector, top_k=8, category=None):
            return fake_hits

        async def fetch_all_for_bm25(self):
            return fake_hits

    class FakeEmbedder:
        async def embed_query(self, text):
            return [0.1] * 4

    with patch("app.rag.retriever.get_milvus_store", return_value=FakeStore()), patch(
        "app.rag.retriever.get_embedding_service", return_value=FakeEmbedder()
    ):
        await retriever.warmup()
        results = await retriever.search("freeze attack")

    assert results
    assert results[0].title == "Freeze"


def test_full_wiki_chunker_runs(tmp_path: Path) -> None:
    """对一个简化 wiki 目录跑 chunker 不报错。"""
    sample = tmp_path / "Skills" / "Mechanics"
    sample.mkdir(parents=True)
    (sample / "freeze.md").write_text(
        "# Freeze\n\n## Description\nFreezes the target.\n\n```yaml\nfreeze @target\n```\n"
    )
    text = (sample / "freeze.md").read_text()
    chunks = chunk_markdown(text, source="Skills/Mechanics/freeze.md")
    assert chunks
    assert any(c.has_yaml for c in chunks)
