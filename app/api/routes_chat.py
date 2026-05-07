"""POST /chat —— SSE 流式 + 非流式两种模式。"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest
from app.schemas.common import APIResponse
from app.services.chat_service import chat_once, chat_stream

router = APIRouter()


@router.post("/chat")
async def chat_endpoint(req: ChatRequest, request: Request):  # type: ignore[no-untyped-def]
    """聊天主入口。"""
    if req.stream:
        async def event_gen():
            async for line in chat_stream(req.message, session_id=req.session_id):
                if await request.is_disconnected():
                    return
                yield line

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    result = await chat_once(req.message, session_id=req.session_id)
    return APIResponse.ok(result)
