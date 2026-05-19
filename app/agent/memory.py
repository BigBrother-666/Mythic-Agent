"""短期会话记忆 —— 支持两种后端：进程内 dict（默认）与 Redis（持久化跨重启）。

存储 schema（仅 Redis 后端）
-----------------------------
键: ``{prefix}{session_id}``（prefix 由 `MEMORY_REDIS_PREFIX` 控制，默认 ``mma:sess:``）
值: JSON 序列化的 `SessionState`，即
    ``{"history": [{role, content}, ...], "last_yaml": "...", "updated_at": 1234.5}``
TTL: 每次写入用 ``SET ... EX session_ttl_seconds`` 重置，实现"活跃即续期"。

线程/并发安全
-------------
- InMemoryBackend 使用 `asyncio.Lock` 串行化每个 session 的读改写。
- RedisBackend 通过 Lua 脚本保证 append_message / set_last_yaml 的原子性
  （GET→decode→mutate→SET EX 在单次 EVAL 中完成，无竞态窗口）。

API 一致性
----------
所有公共方法均为 async；即使 InMemoryBackend 实际是同步实现，也保持 async 签名，
让调用方（chat_service）不必关心后端。
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from app.core.config import get_settings
from app.core.logging import logger


# ---------- 数据结构 ----------


@dataclass
class SessionState:
    """单个会话的状态；序列化后即可入 Redis。"""

    history: list[dict[str, str]] = field(default_factory=list)
    last_yaml: str | None = None
    updated_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        """落 Redis 用。"""
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "SessionState":
        """Redis 读出后还原；旧字段兼容性容错。"""
        data = json.loads(raw)
        return cls(
            history=list(data.get("history") or []),
            last_yaml=data.get("last_yaml"),
            updated_at=float(data.get("updated_at") or time.time()),
        )


# 最近若干轮历史窗口；老消息直接丢弃，避免 prompt 爆 context
_HISTORY_WINDOW = 12


# ---------- Redis Lua 脚本（原子读改写） ----------

_LUA_APPEND_MESSAGE = """
local key = KEYS[1]
local role = ARGV[1]
local content = ARGV[2]
local window = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])
local now = ARGV[5]

local raw = redis.call('GET', key)
local state
if raw then
    state = cjson.decode(raw)
else
    state = {history = {}, last_yaml = false, updated_at = 0}
end

table.insert(state.history, {role = role, content = content})
local n = #state.history
if n > window then
    local trimmed = {}
    for i = n - window + 1, n do
        trimmed[#trimmed + 1] = state.history[i]
    end
    state.history = trimmed
end

state.updated_at = tonumber(now)
redis.call('SET', key, cjson.encode(state), 'EX', ttl)
return 1
"""

_LUA_SET_LAST_YAML = """
local key = KEYS[1]
local yaml_text = ARGV[1]
local ttl = tonumber(ARGV[2])
local now = ARGV[3]

local raw = redis.call('GET', key)
local state
if raw then
    state = cjson.decode(raw)
else
    state = {history = {}, last_yaml = false, updated_at = 0}
end

state.last_yaml = yaml_text
state.updated_at = tonumber(now)
redis.call('SET', key, cjson.encode(state), 'EX', ttl)
return 1
"""


# ---------- 后端协议 ----------


class MemoryBackend(Protocol):
    """会话状态的存取抽象。所有方法异步以兼容 Redis。"""

    async def load(self, session_id: str) -> SessionState | None: ...

    async def save(self, session_id: str, state: SessionState) -> None: ...

    async def delete(self, session_id: str) -> None: ...

    async def size(self) -> int: ...


# ---------- 进程内后端（默认） ----------


class InMemoryBackend:
    """LRU + TTL 的纯进程内实现；零依赖、最快、容器重启即丢。"""

    def __init__(self, max_sessions: int = 1024) -> None:
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._max = max_sessions
        self._lock = asyncio.Lock()

    def _gc_locked(self) -> None:
        """淘汰过期 + 超出容量的会话；调用方持锁。"""
        ttl = get_settings().session_ttl_seconds
        now = time.time()
        for sid in list(self._sessions.keys()):
            if now - self._sessions[sid].updated_at > ttl:
                del self._sessions[sid]
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)

    async def load(self, session_id: str) -> SessionState | None:
        async with self._lock:
            self._gc_locked()
            state = self._sessions.get(session_id)
            if state is None:
                return None
            self._sessions.move_to_end(session_id)
            return state

    async def save(self, session_id: str, state: SessionState) -> None:
        async with self._lock:
            state.updated_at = time.time()
            self._sessions[session_id] = state
            self._sessions.move_to_end(session_id)
            self._gc_locked()

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def size(self) -> int:
        async with self._lock:
            self._gc_locked()
            return len(self._sessions)


# ---------- Redis 后端 ----------


class RedisBackend:
    """Redis 持久化后端；存为 JSON 字符串 + EX TTL。

    连接失败时**降级到 InMemoryBackend**（保证服务可用），并记录 warning。
    """

    def __init__(self, redis_url: str, prefix: str = "mma:sess:") -> None:
        self._url = redis_url
        self._prefix = prefix
        self._redis: Any | None = None
        self._fallback: InMemoryBackend | None = None
        self._lock = asyncio.Lock()

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    async def _client(self) -> Any:
        """惰性连接；失败时切到内存 fallback。"""
        if self._redis is not None or self._fallback is not None:
            return self._redis
        async with self._lock:
            if self._redis is not None or self._fallback is not None:
                return self._redis
            try:
                import redis.asyncio as redis_async

                client = redis_async.from_url(self._url, decode_responses=True)
                await client.ping()
                self._redis = client
                logger.info("Memory backend: Redis @ {} (prefix={})", self._url, self._prefix)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Redis memory backend unavailable ({}); falling back to in-memory", e
                )
                self._fallback = InMemoryBackend()
            return self._redis

    async def load(self, session_id: str) -> SessionState | None:
        client = await self._client()
        if client is None:
            assert self._fallback is not None
            return await self._fallback.load(session_id)
        raw = await client.get(self._key(session_id))
        if raw is None:
            return None
        try:
            return SessionState.from_json(raw)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("Corrupt session {} in Redis ({}); dropping", session_id, e)
            await client.delete(self._key(session_id))
            return None

    async def save(self, session_id: str, state: SessionState) -> None:
        client = await self._client()
        if client is None:
            assert self._fallback is not None
            await self._fallback.save(session_id, state)
            return
        state.updated_at = time.time()
        ttl = get_settings().session_ttl_seconds
        await client.set(self._key(session_id), state.to_json(), ex=ttl)

    async def delete(self, session_id: str) -> None:
        client = await self._client()
        if client is None:
            assert self._fallback is not None
            await self._fallback.delete(session_id)
            return
        await client.delete(self._key(session_id))

    async def size(self) -> int:
        """Redis 后端不便统计全量大小（SCAN 慢），返回 -1 表示不支持。"""
        client = await self._client()
        if client is None:
            assert self._fallback is not None
            return await self._fallback.size()
        return -1

    async def append_message_atomic(
        self, session_id: str, role: str, content: str
    ) -> bool:
        """Lua 原子 append：GET→decode→append→trim→SET EX，单次 RTT。"""
        client = await self._client()
        if client is None:
            return False
        ttl = get_settings().session_ttl_seconds
        await client.eval(
            _LUA_APPEND_MESSAGE,
            1,
            self._key(session_id),
            role,
            content,
            str(_HISTORY_WINDOW),
            str(ttl),
            str(time.time()),
        )
        return True

    async def set_last_yaml_atomic(self, session_id: str, yaml_text: str) -> bool:
        """Lua 原子 set_last_yaml：GET→decode→update→SET EX，单次 RTT。"""
        client = await self._client()
        if client is None:
            return False
        ttl = get_settings().session_ttl_seconds
        await client.eval(
            _LUA_SET_LAST_YAML,
            1,
            self._key(session_id),
            yaml_text,
            str(ttl),
            str(time.time()),
        )
        return True



class MemoryStore:
    """提供面向业务的 append_message / set_last_yaml / snapshot 等操作；委派给 backend。"""

    def __init__(self, backend: MemoryBackend | None = None) -> None:
        if backend is None:
            backend = _build_default_backend()
        self._backend: MemoryBackend = backend

    @property
    def backend(self) -> MemoryBackend:
        """暴露给测试和监控用。"""
        return self._backend

    async def get(self, session_id: str) -> SessionState:
        """取/创建会话。"""
        state = await self._backend.load(session_id)
        if state is None:
            state = SessionState()
            await self._backend.save(session_id, state)
        return state

    async def append_message(self, session_id: str, role: str, content: str) -> None:
        """追加一条消息到历史，并自动截断到最近 _HISTORY_WINDOW 条。"""
        if hasattr(self._backend, "append_message_atomic"):
            ok = await self._backend.append_message_atomic(session_id, role, content)
            if ok:
                return
        state = await self.get(session_id)
        state.history.append({"role": role, "content": content})
        state.history = state.history[-_HISTORY_WINDOW:]
        await self._backend.save(session_id, state)

    async def set_last_yaml(self, session_id: str, yaml_text: str) -> None:
        """记录最近一次生成的 YAML。"""
        if hasattr(self._backend, "set_last_yaml_atomic"):
            ok = await self._backend.set_last_yaml_atomic(session_id, yaml_text)
            if ok:
                return
        state = await self.get(session_id)
        state.last_yaml = yaml_text
        await self._backend.save(session_id, state)

    async def snapshot(self, session_id: str, max_turns: int = 3) -> dict[str, object]:
        """读取会话快照，供 graph 注入到状态。

        返回最近 max_turns 轮 (user/assistant 配对) 历史 + last_yaml。
        history 截到 user 消息打头的边界，避免发半截 assistant 给 LLM。
        """
        state = await self._backend.load(session_id)
        if state is None:
            return {"history": [], "last_yaml": ""}
        recent = list(state.history[-2 * max_turns :])
        if recent and recent[0]["role"] == "assistant":
            recent = recent[1:]
        return {"history": recent, "last_yaml": state.last_yaml or ""}

    async def clear(self, session_id: str) -> None:
        """删除整个会话"""
        await self._backend.delete(session_id)


# ---------- 工厂 / 单例 ----------


def _build_default_backend() -> MemoryBackend:
    """根据 settings 选后端。"""
    s = get_settings()
    if s.memory_backend == "redis":
        return RedisBackend(redis_url=s.redis_url, prefix=s.memory_redis_prefix)
    return InMemoryBackend()


_store: MemoryStore | None = None


def get_memory() -> MemoryStore:
    """全局 memory 单例。"""
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


def reset_memory() -> None:
    """重置单例（测试用）。"""
    global _store
    _store = None
