"""eval/run_eval.py 单元测试 —— 锁定指标计算 + 命中判定逻辑。

不连真 Milvus / Reranker；mock search_wiki 给固定结果。
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.rag.retriever import FusedHit
from eval.run_eval import (
    CaseResult,
    EvalCase,
    _match,
    evaluate_all,
    metrics,
    metrics_by_tag,
)


# ---------- 命中判定 ----------


def test_match_substring_with_correct_wiki() -> None:
    case = EvalCase(
        id="X",
        query="q",
        expected_sources=["Skills/Mechanics/projectile.md"],
        expected_wiki="mythicmobs",
        tags=["mechanic"],
    )
    assert _match(case, "MythicMobs.wiki/Skills/Mechanics/projectile.md", "mythicmobs")
    assert not _match(case, "MythicMobs.wiki/Skills/Mechanics/projectile.md", "crucible")
    assert not _match(case, "MythicMobs.wiki/Skills/Mechanics/freeze.md", "mythicmobs")


def test_match_any_of_multiple_expected() -> None:
    case = EvalCase(
        id="X", query="q",
        expected_sources=["a.md", "b.md", "c.md"],
        expected_wiki="any",
        tags=[],
    )
    assert _match(case, "wiki/foo/b.md", "mythicmobs")
    assert _match(case, "wiki/foo/c.md", "crucible")  # any 接受任何 wiki
    assert not _match(case, "wiki/foo/d.md", "mythicmobs")


def test_match_wiki_any_skips_wiki_filter() -> None:
    case = EvalCase(id="X", query="q", expected_sources=["x.md"], expected_wiki="any", tags=[])
    assert _match(case, "x.md", "crucible")
    assert _match(case, "x.md", "mythicmobs")


# ---------- 指标计算 ----------


def _r(rank: int, cid: str = "x") -> CaseResult:
    return CaseResult(id=cid, query="q", rank=rank, top_sources=[], rerank_enabled=False)


def test_recall_and_mrr_simple() -> None:
    # 4 题：rank 分别 1, 3, 0(miss), 5
    results = [_r(1), _r(3), _r(0), _r(5)]
    m = metrics(results, ks=[1, 3, 5, 8])
    assert m["n"] == 4.0
    assert m["recall@1"] == 1 / 4
    assert m["recall@3"] == 2 / 4  # rank 1 + rank 3
    assert m["recall@5"] == 3 / 4  # + rank 5
    assert m["recall@8"] == 3 / 4  # rank 0 永远不算
    expected_mrr = (1.0 + 1 / 3 + 0.0 + 1 / 5) / 4
    assert m["mrr"] == pytest.approx(expected_mrr)


def test_metrics_empty_returns_empty() -> None:
    assert metrics([], ks=[1, 5]) == {}


def test_metrics_by_tag_groups_correctly() -> None:
    cases = [
        EvalCase(id="a", query="", expected_sources=[], expected_wiki="any", tags=["zh", "mob"]),
        EvalCase(id="b", query="", expected_sources=[], expected_wiki="any", tags=["zh"]),
        EvalCase(id="c", query="", expected_sources=[], expected_wiki="any", tags=["en", "mob"]),
    ]
    results = [_r(1, "a"), _r(0, "b"), _r(2, "c")]
    by_tag = metrics_by_tag(results, cases, k=5)
    assert by_tag["zh"] == pytest.approx(1 / 2)  # a hit, b miss
    assert by_tag["en"] == pytest.approx(1.0)
    assert by_tag["mob"] == pytest.approx(1.0)  # a + c 都命中


# ---------- evaluate_all 行为（不真的搜索） ----------


@pytest.mark.asyncio
async def test_evaluate_all_uses_first_match_rank() -> None:
    """命中位置应取**第一个**命中位置；后续命中不重复算。"""
    cases = [
        EvalCase(
            id="X", query="dummy",
            expected_sources=["projectile.md"],
            expected_wiki="mythicmobs",
            tags=["m"],
        ),
    ]
    fake_hits = [
        FusedHit(score=1.0, text="t", title="A", source="x/freeze.md", category="m", tags=[], wiki="mythicmobs"),
        FusedHit(score=0.5, text="t", title="B", source="x/projectile.md", category="m", tags=[], wiki="mythicmobs"),
        FusedHit(score=0.3, text="t", title="C", source="x/projectile.md", category="m", tags=[], wiki="mythicmobs"),
    ]

    async def fake_search(q, top_k=8, **kw):  # noqa: ARG001
        return fake_hits

    with patch("eval.run_eval.search_wiki", side_effect=fake_search):
        results = await evaluate_all(cases, top_k=8)
    assert results[0].rank == 2
    assert results[0].top_sources == ["x/freeze.md", "x/projectile.md", "x/projectile.md"]


@pytest.mark.asyncio
async def test_evaluate_all_miss_when_wiki_mismatch() -> None:
    """期望 wiki=mythicmobs 时，crucible 命中也算 miss。"""
    cases = [
        EvalCase(
            id="X", query="q",
            expected_sources=["onUse.md"],
            expected_wiki="mythicmobs",  # 故意错配
            tags=[],
        ),
    ]
    fake_hits = [
        FusedHit(score=1.0, text="t", title="A", source="x/onUse.md", category="m", tags=[], wiki="crucible"),
    ]

    async def fake_search(q, top_k=8, **kw):  # noqa: ARG001
        return fake_hits

    with patch("eval.run_eval.search_wiki", side_effect=fake_search):
        results = await evaluate_all(cases, top_k=8)
    assert results[0].rank == 0  # miss
