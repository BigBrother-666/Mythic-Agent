"""Memory 后端单元测试 —— InMemoryBackend + MemoryStore facade。

不依赖 Redis；RedisBackend 的降级行为也在此验证（连接失败 → fallback 到内存）。
"""
from __future__ import annotations

import time

import pytest

from app.agent.memory import (
    InMemoryBackend,
    MemoryStore,
    RedisBackend,
    SessionState,
)


# ---------- SessionState 序列化 ----------


def test_session_state_roundtrip() -> None:
    s = SessionState(
        history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        last_yaml="Foo:\n  Type: ZOMBIE\n",
        updated_at=1700000000.0,
    )
    raw = s.to_json()
    restored = SessionState.from_json(raw)
    assert restored.history == s.history
    assert restored.last_yaml == s.last_yaml
    assert restored.updated_at == s.updated_at


def test_session_state_from_json_tolerates_missing_fields() -> None:
    restored = SessionState.from_json('{"history": []}')
    assert restored.history == []
    assert restored.last_yaml is None
    assert restored.updated_at > 0


# ---------- InMemoryBackend ----------


@pytest.mark.asyncio
async def test_inmemory_load_returns_none_for_unknown() -> None:
    backend = InMemoryBackend()
    assert await backend.load("nonexistent") is None


@pytest.mark.asyncio
async def test_inmemory_save_and_load() -> None:
    backend = InMemoryBackend()
    state = SessionState(history=[{"role": "user", "content": "x"}], last_yaml="Y")
    await backend.save("s1", state)
    loaded = await backend.load("s1")
    assert loaded is not None
    assert loaded.history == [{"role": "user", "content": "x"}]
    assert loaded.last_yaml == "Y"


@pytest.mark.asyncio
async def test_inmemory_delete() -> None:
    backend = InMemoryBackend()
    await backend.save("s1", SessionState())
    await backend.delete("s1")
    assert await backend.load("s1") is None


@pytest.mark.asyncio
async def test_inmemory_lru_eviction() -> None:
    backend = InMemoryBackend(max_sessions=2)
    await backend.save("a", SessionState())
    await backend.save("b", SessionState())
    await backend.save("c", SessionState())  # 'a' should be evicted
    assert await backend.load("a") is None
    assert await backend.load("b") is not None
    assert await backend.load("c") is not None


@pytest.mark.asyncio
async def test_inmemory_ttl_expiry() -> None:
    backend = InMemoryBackend()
    state = SessionState()
    await backend.save("old", state)
    # 手动把 updated_at 改到远过去（模拟长时间不活跃）
    backend._sessions["old"].updated_at = time.time() - 99999
    # 下一次 load 触发 GC → 过期会话被清除
    assert await backend.load("old") is None


@pytest.mark.asyncio
async def test_inmemory_size() -> None:
    backend = InMemoryBackend()
    await backend.save("a", SessionState())
    await backend.save("b", SessionState())
    assert await backend.size() == 2


# ---------- MemoryStore facade ----------


@pytest.mark.asyncio
async def test_store_get_creates_new_session() -> None:
    store = MemoryStore(backend=InMemoryBackend())
    s = await store.get("new_session")
    assert s.history == []
    assert s.last_yaml is None


@pytest.mark.asyncio
async def test_store_append_message_and_window() -> None:
    store = MemoryStore(backend=InMemoryBackend())
    for i in range(15):
        await store.append_message("s1", "user", f"msg{i}")
    s = await store.get("s1")
    assert len(s.history) == 12  # 窗口限制
    assert s.history[0]["content"] == "msg3"  # 前 3 条被截掉


@pytest.mark.asyncio
async def test_store_set_last_yaml() -> None:
    store = MemoryStore(backend=InMemoryBackend())
    await store.set_last_yaml("s1", "Boss:\n  Type: WITHER\n")
    s = await store.get("s1")
    assert s.last_yaml == "Boss:\n  Type: WITHER\n"


@pytest.mark.asyncio
async def test_store_snapshot_empty_session() -> None:
    store = MemoryStore(backend=InMemoryBackend())
    snap = await store.snapshot("nonexistent")
    assert snap == {"history": [], "last_yaml": ""}


@pytest.mark.asyncio
async def test_store_snapshot_trims_to_user_start() -> None:
    store = MemoryStore(backend=InMemoryBackend())
    await store.append_message("s1", "assistant", "orphan")
    await store.append_message("s1", "user", "q1")
    await store.append_message("s1", "assistant", "a1")
    snap = await store.snapshot("s1", max_turns=3)
    # 第一条 assistant 被去掉（确保 history 以 user 开头）
    assert snap["history"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_store_clear() -> None:
    store = MemoryStore(backend=InMemoryBackend())
    await store.append_message("s1", "user", "hi")
    await store.clear("s1")
    snap = await store.snapshot("s1")
    assert snap == {"history": [], "last_yaml": ""}


# ---------- RedisBackend 降级 ----------


@pytest.mark.asyncio
async def test_redis_backend_fallback_on_connection_failure() -> None:
    """Redis 连不上时自动降级到 InMemoryBackend，不抛异常。"""
    backend = RedisBackend(redis_url="redis://localhost:1/0")  # 不可达端口
    # 第一次操作触发连接 → 失败 → fallback
    await backend.save("s1", SessionState(history=[{"role": "user", "content": "x"}]))
    loaded = await backend.load("s1")
    assert loaded is not None
    assert loaded.history == [{"role": "user", "content": "x"}]


# ---------- MemoryStore 原子路径 ----------


class _FakeAtomicBackend:
    """模拟带原子方法的后端，记录调用。"""

    def __init__(self, atomic_ok: bool = True) -> None:
        self._atomic_ok = atomic_ok
        self.atomic_append_calls: list[tuple] = []
        self.atomic_yaml_calls: list[tuple] = []
        self._inner = InMemoryBackend()

    async def load(self, session_id: str) -> SessionState | None:
        return await self._inner.load(session_id)

    async def save(self, session_id: str, state: SessionState) -> None:
        await self._inner.save(session_id, state)

    async def delete(self, session_id: str) -> None:
        await self._inner.delete(session_id)

    async def size(self) -> int:
        return await self._inner.size()

    async def append_message_atomic(
        self, session_id: str, role: str, content: str
    ) -> bool:
        self.atomic_append_calls.append((session_id, role, content))
        return self._atomic_ok

    async def set_last_yaml_atomic(self, session_id: str, yaml_text: str) -> bool:
        self.atomic_yaml_calls.append((session_id, yaml_text))
        return self._atomic_ok


@pytest.mark.asyncio
async def test_memorystore_append_prefers_atomic() -> None:
    """MemoryStore 优先调用 backend 的 append_message_atomic。"""
    backend = _FakeAtomicBackend(atomic_ok=True)
    store = MemoryStore(backend=backend)
    await store.append_message("s1", "user", "hello")
    assert len(backend.atomic_append_calls) == 1
    assert backend.atomic_append_calls[0] == ("s1", "user", "hello")


@pytest.mark.asyncio
async def test_memorystore_append_fallback_when_atomic_fails() -> None:
    """atomic 返回 False 时走 load+save 路径。"""
    backend = _FakeAtomicBackend(atomic_ok=False)
    store = MemoryStore(backend=backend)
    await store.append_message("s1", "user", "hello")
    # atomic 被调用但返回 False
    assert len(backend.atomic_append_calls) == 1
    # fallback 路径写入了数据
    state = await backend.load("s1")
    assert state is not None
    assert state.history[-1] == {"role": "user", "content": "hello"}


@pytest.mark.asyncio
async def test_memorystore_set_yaml_prefers_atomic() -> None:
    """MemoryStore 优先调用 backend 的 set_last_yaml_atomic。"""
    backend = _FakeAtomicBackend(atomic_ok=True)
    store = MemoryStore(backend=backend)
    await store.set_last_yaml("s1", "Boss:\n  Type: ZOMBIE\n")
    assert len(backend.atomic_yaml_calls) == 1
    assert backend.atomic_yaml_calls[0] == ("s1", "Boss:\n  Type: ZOMBIE\n")
