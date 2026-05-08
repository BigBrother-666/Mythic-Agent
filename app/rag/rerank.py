"""Cross-encoder 重排序服务（可选启用）。

工作流：
- vector + BM25 → RRF 融合 → 取 top_k * pool_factor 个候选
- 对每个 (query, candidate.text) 用 CrossEncoder 算相关性分
- 按新分数排序、截到 top_k

依赖：
- `sentence-transformers` 提供的 CrossEncoder（项目 requirements 已含）
- 模型从 HF Hub 拉取，复用 `HF_HOME` 缓存（与 embedding 共享）

惰性加载：
- 进程启动时不加载；首次调用 `rerank()` 时才装载（首次 ~10s + 网络）
- 之后保持在内存（~280MB for bge-reranker-base）
- 设置 `ENABLE_RERANK=false` 时整个模型不会被加载
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.core.config import get_settings
from app.core.logging import logger


class RerankService:
    """对 cross-encoder 模型的薄封装。"""

    _instance: "RerankService | None" = None

    def __init__(self) -> None:
        self._model: Any | None = None
        self._lock = asyncio.Lock()

    @classmethod
    def instance(cls) -> "RerankService":
        """全局单例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def _ensure_loaded(self) -> Any:
        """惰性加载 CrossEncoder；并发安全。"""
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is not None:
                return self._model
            settings = get_settings()
            logger.info(
                "Loading reranker {} on {} (lazy first call)",
                settings.rerank_model,
                settings.rerank_device,
            )
            # 惰性 import：未启用 rerank 的部署不必装 sentence-transformers 的全部依赖
            from sentence_transformers import CrossEncoder

            def _load() -> Any:
                return CrossEncoder(
                    settings.rerank_model,
                    max_length=settings.rerank_max_length,
                    device=settings.rerank_device,
                )

            self._model = await asyncio.to_thread(_load)
            logger.info("Reranker ready")
            return self._model

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        """对 (query, candidate) 对批量打分，返回与 candidates 同序的分数列表。

        bge-reranker 输出原始 logits（未归一化），数值范围大致在 [-10, 10]，越大越相关。
        我们不做归一化——下游只用相对排序，不依赖绝对值。
        """
        if not candidates:
            return []
        model = await self._ensure_loaded()
        settings = get_settings()
        pairs = [[query, c] for c in candidates]

        def _predict() -> list[float]:
            scores = model.predict(
                pairs,
                batch_size=settings.rerank_batch_size,
                show_progress_bar=False,
            )
            return [float(s) for s in scores]

        return await asyncio.to_thread(_predict)


def get_rerank_service() -> RerankService:
    """获取全局 rerank service 单例。"""
    return RerankService.instance()
