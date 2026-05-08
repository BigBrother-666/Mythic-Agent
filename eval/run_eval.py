"""RAG 评估 runner —— 跑 eval/queries.yaml，输出 Recall@K + MRR。

用法（在容器内）：
    docker compose exec fastapi python -m eval.run_eval
    docker compose exec fastapi python -m eval.run_eval --toggle-rerank
    docker compose exec fastapi python -m eval.run_eval --top-k 8 --rerank on
    docker compose exec fastapi python -m eval.run_eval --query-file eval/queries.yaml --out eval/results

输出：
- 终端打印汇总表格
- JSON 详情写入 eval/results/{timestamp}_{label}.json，per-query 命中明细可复盘

「命中」判定：
- expected_sources 是 substring 列表，**任意一条**出现在 hit.source 中即视为命中（容忍路径前缀差异）
- expected_wiki 必须匹配（除非声明 "any"）
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from app.core.config import get_settings
from app.core.logging import logger, setup_logging
from app.rag.pipeline import search_wiki, warmup


# ---------- 数据结构 ----------


@dataclass
class EvalCase:
    id: str
    query: str
    expected_sources: list[str]
    expected_wiki: str  # mythicmobs / crucible / any
    tags: list[str]


@dataclass
class CaseResult:
    id: str
    query: str
    rank: int  # 第一个命中的位置（1-based）；0 表示未命中
    top_sources: list[str]  # 前 K 个 hit 的 source，便于复盘
    rerank_enabled: bool


# ---------- 加载 ----------


def load_cases(path: Path) -> list[EvalCase]:
    """读 eval/queries.yaml。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    out: list[EvalCase] = []
    for c in raw["cases"]:
        out.append(
            EvalCase(
                id=str(c["id"]),
                query=str(c["query"]),
                expected_sources=list(c.get("expected_sources") or []),
                expected_wiki=str(c.get("expected_wiki") or "any"),
                tags=list(c.get("tags") or []),
            )
        )
    return out


# ---------- 命中判定 ----------


def _match(case: EvalCase, hit_source: str, hit_wiki: str) -> bool:
    """单个 hit 是否满足 case 的期望。"""
    if case.expected_wiki not in {"any", hit_wiki}:
        return False
    return any(exp in hit_source for exp in case.expected_sources)


# ---------- 评估 ----------


async def evaluate_one(case: EvalCase, top_k: int) -> CaseResult:
    """对单条 case 跑一次检索，返回命中位置。"""
    hits = await search_wiki(case.query, top_k=top_k)
    rank = 0
    top_sources = []
    for i, h in enumerate(hits, 1):
        top_sources.append(h.source)
        if rank == 0 and _match(case, h.source, h.wiki):
            rank = i
    return CaseResult(
        id=case.id,
        query=case.query,
        rank=rank,
        top_sources=top_sources,
        rerank_enabled=get_settings().enable_rerank,
    )


async def evaluate_all(cases: list[EvalCase], top_k: int) -> list[CaseResult]:
    """串行评估（避免一次性把模型 OOM）。"""
    results: list[CaseResult] = []
    for c in cases:
        r = await evaluate_one(c, top_k=top_k)
        marker = "+" if r.rank else "-"
        logger.info("[{}] {} rank={} | {}", marker, c.id, r.rank or "miss", c.query[:60])
        results.append(r)
    return results


# ---------- 指标 ----------


def metrics(results: list[CaseResult], ks: list[int]) -> dict[str, float]:
    """Recall@K + MRR 标量指标。"""
    n = len(results)
    if n == 0:
        return {}
    out: dict[str, float] = {}
    for k in ks:
        hit = sum(1 for r in results if 0 < r.rank <= k)
        out[f"recall@{k}"] = hit / n
    rrs = [(1.0 / r.rank) if r.rank > 0 else 0.0 for r in results]
    out["mrr"] = statistics.mean(rrs)
    out["n"] = float(n)
    return out


def metrics_by_tag(results: list[CaseResult], cases: list[EvalCase], k: int) -> dict[str, float]:
    """按 tag 分组的 recall@k；用于发现「中文短查询命中差」之类局部弱点。"""
    by_tag: dict[str, list[bool]] = {}
    by_id = {c.id: c for c in cases}
    for r in results:
        c = by_id.get(r.id)
        if not c:
            continue
        hit = 0 < r.rank <= k
        for t in c.tags:
            by_tag.setdefault(t, []).append(hit)
    return {tag: sum(v) / len(v) for tag, v in by_tag.items()}


# ---------- 输出 ----------


def format_table(metrics_map: dict[str, dict[str, float]]) -> str:
    """把多次实验的指标排成 ASCII 表"""
    if not metrics_map:
        return "(no metrics)"
    labels = list(metrics_map.keys())
    metric_keys = ["recall@1", "recall@3", "recall@5", "recall@8", "mrr"]
    rows = ["| metric | " + " | ".join(labels) + " |"]
    rows.append("|" + "---|" * (len(labels) + 1))
    for mk in metric_keys:
        line = [mk]
        for lb in labels:
            val = metrics_map[lb].get(mk, float("nan"))
            line.append(f"{val:.3f}")
        rows.append("| " + " | ".join(line) + " |")
    n = next(iter(metrics_map.values())).get("n", 0)
    rows.append(f"\nN = {int(n)}")
    return "\n".join(rows)


def write_results(out_dir: Path, label: str, cases: list[EvalCase], results: list[CaseResult]) -> Path:
    """把详细结果写到 JSON。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = out_dir / f"{ts}_{label}.json"
    payload = {
        "label": label,
        "timestamp": ts,
        "settings": {
            "embed_model": get_settings().embed_model,
            "rerank_enabled": get_settings().enable_rerank,
            "rerank_model": get_settings().rerank_model,
            "rerank_pool_factor": get_settings().rerank_pool_factor,
            "rag_top_k": get_settings().rag_top_k,
        },
        "metrics": metrics(results, ks=[1, 3, 5, 8]),
        "metrics_by_tag_at_5": metrics_by_tag(results, cases, k=5),
        "cases": [asdict(r) for r in results],
    }
    fname.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return fname


# ---------- 主入口 ----------


async def _run(
    cases: list[EvalCase],
    top_k: int,
    label: str,
    out_dir: Path,
) -> dict[str, float]:
    """跑一次 + 落盘 + 返回指标。"""
    await warmup()
    results = await evaluate_all(cases, top_k=top_k)
    m = metrics(results, ks=[1, 3, 5, 8])
    fname = write_results(out_dir, label, cases, results)
    logger.info("Wrote {}", fname)
    return m


async def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-file", type=Path, default=Path("eval/queries.yaml"))
    parser.add_argument("--out", type=Path, default=Path("eval/results"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--rerank",
        choices=["on", "off", "auto"],
        default="auto",
        help="auto=按 .env 跑一次；on/off=临时覆盖；--toggle-rerank 优先级更高",
    )
    parser.add_argument(
        "--toggle-rerank",
        action="store_true",
        help="同一进程内跑两次：rerank=off 与 rerank=on，输出对比表",
    )
    args = parser.parse_args()

    cases = load_cases(args.query_file)
    logger.info(
        "Loaded {} cases; embed_model={}",
        len(cases),
        get_settings().embed_model,
    )
    settings = get_settings()
    metrics_map: dict[str, dict[str, float]] = {}

    async def run_with_rerank(flag: bool, label: str) -> None:
        # 临时切换 settings.enable_rerank。Settings 是 pydantic-settings BaseModel，
        # 直接赋值 + 触发 retriever 重建即可。
        settings.enable_rerank = flag
        # 重建 retriever 单例，让候选池大小按新设置生效；BM25 corpus 已经在内存里，warmup 是 noop。
        from app.rag.retriever import HybridRetriever

        HybridRetriever._instance = None
        m = await _run(cases, top_k=args.top_k, label=label, out_dir=args.out)
        metrics_map[label] = m

    if args.toggle_rerank:
        await run_with_rerank(False, "rerank_off")
        await run_with_rerank(True, "rerank_on")
    elif args.rerank == "off":
        await run_with_rerank(False, "rerank_off")
    elif args.rerank == "on":
        await run_with_rerank(True, "rerank_on")
    else:
        # auto：按 .env 当前值跑一次
        label = "rerank_on" if settings.enable_rerank else "rerank_off"
        await run_with_rerank(settings.enable_rerank, label)

    print("\n========= EVAL SUMMARY =========")
    print(format_table(metrics_map))
    if "rerank_off" in metrics_map and "rerank_on" in metrics_map:
        off = metrics_map["rerank_off"]
        on = metrics_map["rerank_on"]

        def delta(k: str) -> str:
            d = (on[k] - off[k]) * 100
            sign = "+" if d >= 0 else ""
            return f"{sign}{d:.1f}pp"

        print("\nDelta (rerank_on - rerank_off):")
        for k in ["recall@1", "recall@3", "recall@5", "recall@8", "mrr"]:
            print(f"  {k:10s}  {delta(k)}")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
