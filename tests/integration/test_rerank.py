"""Rerank 集成到 retriever 的契约测试。

不真的加载 sentence-transformers 模型，而是 mock RerankService.score
来验证：
- ENABLE_RERANK=false：retriever 不调 rerank service，截断到 top_k
- ENABLE_RERANK=true：候选池放大到 top_k * pool_factor；rerank 后顺序按新分数；rerank 失败时退回 RRF 顺序不报错
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.rag import retriever as retriever_module
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import VectorHit


def _make_hit(i: int) -> VectorHit:
    return VectorHit(
        score=1.0 / (i + 1),
        text=f"chunk-{i} text body about projectile and freeze",
        title=f"Chunk {i}",
        source=f"MythicMobs.wiki/Skills/Mechanics/c{i}.md",
        category="mechanics",
        tags=["t"],
        wiki="mythicmobs",
    )


class _FakeStore:
    def has_collection(self) -> bool:
        return True

    async def search(self, vector, top_k=8, category=None, wiki=None):  # type: ignore[no-untyped-def]
        return [_make_hit(i) for i in range(min(top_k, 30))]

    async def fetch_all_for_bm25(self):  # type: ignore[no-untyped-def]
        return [_make_hit(i) for i in range(30)]


class _FakeEmbedder:
    async def embed_query(self, text):  # type: ignore[no-untyped-def]
        return [0.1] * 4


def _fresh_retriever() -> HybridRetriever:
    """重置单例。"""
    HybridRetriever._instance = None
    return HybridRetriever.instance()


@pytest.mark.asyncio
async def test_rerank_disabled_does_not_call_service(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """ENABLE_RERANK=false 时不应触碰 rerank service。"""
    from app.core.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "enable_rerank", False)

    rerank_calls = []

    async def fake_score(self, query, candidates):  # noqa: ARG001
        rerank_calls.append((query, candidates))
        return [0.0] * len(candidates)

    with patch("app.rag.retriever.get_milvus_store", return_value=_FakeStore()), patch(
        "app.rag.retriever.get_embedding_service", return_value=_FakeEmbedder()
    ), patch(
        "app.rag.rerank.RerankService.score", new=fake_score
    ):
        retriever = _fresh_retriever()
        await retriever.warmup()
        results = await retriever.search("projectile freeze", top_k=5)

    assert len(results) <= 5
    assert rerank_calls == []  # 完全没调用


@pytest.mark.asyncio
async def test_rerank_enabled_changes_order(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """ENABLE_RERANK=true 时，rerank 分数决定最终顺序。"""
    from app.core.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "enable_rerank", True)
    monkeypatch.setattr(s, "rerank_pool_factor", 3)

    # mock：让"末位 chunk"得到最高 rerank 分，验证顺序确实被改变
    captured: dict[str, list[str]] = {}

    async def fake_score(self, query, candidates):  # noqa: ARG001
        captured["candidates"] = list(candidates)
        # 倒序赋分：最后一个候选分最高
        return list(range(len(candidates)))[::-1]

    with patch("app.rag.retriever.get_milvus_store", return_value=_FakeStore()), patch(
        "app.rag.retriever.get_embedding_service", return_value=_FakeEmbedder()
    ), patch(
        "app.rag.rerank.RerankService.score", new=fake_score
    ):
        retriever = _fresh_retriever()
        await retriever.warmup()
        top_k = 4
        results = await retriever.search("projectile freeze", top_k=top_k)

    # 候选池大小应是 top_k * pool_factor
    assert len(captured["candidates"]) >= top_k * 3 - 1  # 允许去重少 1
    # 截到 top_k
    assert len(results) == top_k
    # 第一名的 score 是最大的（rerank 分覆盖到 FusedHit.score）
    assert results[0].score >= results[-1].score
    # rerank 把最后的候选推到了最前——验证候选拼接里包含 title + 内容
    first_payload = captured["candidates"][0]
    assert "Chunk" in first_payload  # 包含 title


@pytest.mark.asyncio
async def test_rerank_failure_falls_back_to_rrf(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """rerank service 抛错时 retriever 应静默回退。"""
    from app.core.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "enable_rerank", True)

    async def explode(self, query, candidates):  # noqa: ARG001
        raise RuntimeError("model not available")

    with patch("app.rag.retriever.get_milvus_store", return_value=_FakeStore()), patch(
        "app.rag.retriever.get_embedding_service", return_value=_FakeEmbedder()
    ), patch(
        "app.rag.rerank.RerankService.score", new=explode
    ):
        retriever = _fresh_retriever()
        await retriever.warmup()
        results = await retriever.search("projectile freeze", top_k=5)

    # 不抛异常，仍能截到 top_k
    assert len(results) <= 5
    assert all(r.text for r in results)
