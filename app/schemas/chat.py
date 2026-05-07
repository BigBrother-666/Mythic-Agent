"""聊天 / 流式接口 schema。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求体。"""

    message: str = Field(..., min_length=1, max_length=8000, description="用户输入")
    session_id: str | None = Field(default=None, description="会话 ID（可选；不传则匿名）")
    stream: bool = Field(default=True, description="是否使用 SSE 流式输出")


class ChatChunk(BaseModel):
    """SSE 单条消息的 payload。"""

    type: Literal["plan", "retrieval", "token", "yaml", "validation", "done", "error"]
    content: str = ""
    meta: dict[str, str] | None = None


class ChatResponse(BaseModel):
    """非流式聊天响应。"""

    reply: str
    yaml: str | None = None
    sources: list[str] = Field(default_factory=list)
