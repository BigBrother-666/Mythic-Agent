"""短期会话记忆——内存级，按 session_id 隔离。"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

from app.core.config import get_settings


@dataclass
class SessionState:
    """单个会话的状态。"""

    history: list[dict[str, str]] = field(default_factory=list)
    last_yaml: str | None = None
    updated_at: float = field(default_factory=time.time)


class MemoryStore:
    """LRU 风格的内存会话存储；超过 TTL 自动过期。"""

    def __init__(self, max_sessions: int = 1024) -> None:
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._max = max_sessions

    def _gc(self) -> None:
        """淘汰过期会话。"""
        ttl = get_settings().session_ttl_seconds
        now = time.time()
        for sid in list(self._sessions.keys()):
            if now - self._sessions[sid].updated_at > ttl:
                del self._sessions[sid]
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)

    def get(self, session_id: str) -> SessionState:
        """取/创建会话。"""
        self._gc()
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState()
        else:
            self._sessions.move_to_end(session_id)
        return self._sessions[session_id]

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """追加一条消息到历史。"""
        s = self.get(session_id)
        s.history.append({"role": role, "content": content})
        s.history = s.history[-12:]  # 限制窗口防膨胀
        s.updated_at = time.time()

    def set_last_yaml(self, session_id: str, yaml_text: str) -> None:
        """记录最近一次生成的 YAML。"""
        s = self.get(session_id)
        s.last_yaml = yaml_text
        s.updated_at = time.time()

    def snapshot(self, session_id: str, max_turns: int = 3) -> dict[str, object]:
        """读取会话快照，供 graph 注入到状态。

        返回最近 max_turns 轮 (user/assistant 配对) 历史 + last_yaml。
        history 截到 user 消息打头的边界，避免发半截 assistant 给 LLM。
        """
        s = self.get(session_id)
        # 取最后 2*max_turns 条；如果第一条是 assistant，再去掉一条让起点是 user
        recent = list(s.history[-2 * max_turns :])
        if recent and recent[0]["role"] == "assistant":
            recent = recent[1:]
        return {"history": recent, "last_yaml": s.last_yaml or ""}


_store: MemoryStore | None = None


def get_memory() -> MemoryStore:
    """全局 memory 单例。"""
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store
