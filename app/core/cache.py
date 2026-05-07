"""异步 Redis 客户端封装；带降级到内存字典的能力（Redis 不可用时仍可工作）。"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from app.core.config import get_settings
from app.core.logging import logger


class CacheClient:
    """统一缓存接口，优先用 Redis，失败时回退到内存。"""

    def __init__(self) -> None:
        self._redis: Any | None = None
        self._memory: dict[str, tuple[float, str]] = {}
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """建立 Redis 连接；失败时静默降级。"""
        settings = get_settings()
        try:
            import redis.asyncio as redis_async  # 惰性 import，缺包时直接降级

            client = redis_async.from_url(settings.redis_url, decode_responses=True)
            await client.ping()
            self._redis = client
            logger.info("Redis connected at {}", settings.redis_url)
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis unavailable ({}), fallback to in-memory cache", e)
            self._redis = None

    async def close(self) -> None:
        """关闭 Redis 连接。"""
        if self._redis is not None:
            await self._redis.aclose()

    async def get(self, key: str) -> str | None:
        """读缓存。"""
        if self._redis is not None:
            return await self._redis.get(key)
        async with self._lock:
            entry = self._memory.get(key)
            if entry is None:
                return None
            expire, value = entry
            if expire and expire < time.time():
                self._memory.pop(key, None)
                return None
            return value

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        """写缓存。"""
        if self._redis is not None:
            await self._redis.set(key, value, ex=ttl_seconds)
            return
        async with self._lock:
            expire = time.time() + ttl_seconds if ttl_seconds else 0.0
            self._memory[key] = (expire, value)

    async def incr_window(self, key: str, window_seconds: int) -> int:
        """简单滑动窗口计数（限流用）。"""
        if self._redis is not None:
            pipe = self._redis.pipeline()
            pipe.incr(key, 1)
            pipe.expire(key, window_seconds)
            count, _ = await pipe.execute()
            return int(count)
        async with self._lock:
            entry = self._memory.get(key)
            now = time.time()
            if entry is None or entry[0] < now:
                self._memory[key] = (now + window_seconds, "1")
                return 1
            new_count = int(entry[1]) + 1
            self._memory[key] = (entry[0], str(new_count))
            return new_count


_cache: CacheClient | None = None


async def get_cache() -> CacheClient:
    """获取全局缓存客户端。"""
    global _cache
    if _cache is None:
        _cache = CacheClient()
        await _cache.connect()
    return _cache


def set_cache(client: CacheClient) -> None:
    """注入缓存客户端（测试用）。"""
    global _cache
    _cache = client


async def close_cache() -> None:
    """关闭全局缓存客户端。"""
    global _cache
    if _cache is not None:
        await _cache.close()
        _cache = None
