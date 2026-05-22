"""LangFuse v2 可观测性集成——按 session 追踪 agent 全链路。

启用条件：LANGFUSE_ENABLED=true + 配置 public/secret key。
关闭时所有函数返回 None / no-op，零开销。

使用 Langfuse v2 原生 SDK（trace/span/generation），
不依赖 langfuse.callback.CallbackHandler（与 langchain 1.3+ 不兼容）。
"""
from __future__ import annotations

import os
from typing import Any

from app.core.config import get_settings
from app.core.logging import logger

_langfuse_client: Any | None = None
_initialized = False


def _get_client() -> Any | None:
    """惰性初始化全局 Langfuse 客户端。"""
    global _langfuse_client, _initialized
    if _initialized:
        return _langfuse_client
    _initialized = True

    settings = get_settings()
    if not settings.langfuse_enabled:
        return None
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning("LANGFUSE_ENABLED=true but keys not set; disabled")
        return None

    try:
        from langfuse import Langfuse

        # httpx 不支持 SOCKS 代理，连接本地 LangFuse 时需绕过
        saved = os.environ.pop("ALL_PROXY", None)
        saved_lower = os.environ.pop("all_proxy", None)
        try:
            _langfuse_client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        finally:
            if saved is not None:
                os.environ["ALL_PROXY"] = saved
            if saved_lower is not None:
                os.environ["all_proxy"] = saved_lower

        logger.info("LangFuse client initialized (host={})", settings.langfuse_host)
    except Exception as e:  # noqa: BLE001
        logger.warning("LangFuse init failed; tracing disabled: {}", e)
    return _langfuse_client


def create_trace(
    session_id: str | None = None,
    name: str = "chat",
    input_text: str = "",
) -> Any | None:
    """为一次请求创建 trace；未启用时返回 None。"""
    client = _get_client()
    if client is None:
        return None
    try:
        return client.trace(
            name=name,
            session_id=session_id,
            input=input_text[:500],
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to create LangFuse trace: {}", e)
        return None


def flush_langfuse() -> None:
    """Shutdown 时刷新所有待发送的 trace。"""
    client = _get_client()
    if client is not None:
        try:
            client.flush()
        except Exception:  # noqa: BLE001
            pass
