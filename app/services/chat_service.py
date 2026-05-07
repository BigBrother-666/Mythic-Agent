"""聊天服务：把 LangGraph 事件流转换为 SSE chunks。"""
from __future__ import annotations

from typing import Any, AsyncIterator

import orjson

from app.agent.graph import stream_events
from app.agent.memory import get_memory
from app.core.logging import logger
from app.schemas.chat import ChatChunk


def _encode(chunk: ChatChunk) -> str:
    """把 ChatChunk 编码成 SSE data line。"""
    payload = orjson.dumps(chunk.model_dump()).decode()
    return f"data: {payload}\n\n"


async def chat_stream(message: str, session_id: str | None = None) -> AsyncIterator[str]:
    """聊天 SSE 主迭代器。"""
    memory = get_memory()
    if session_id:
        memory.append_message(session_id, "user", message)

    final_yaml = ""

    try:
        async for event in stream_events(message):
            for node_name, node_state in event.items():
                if not isinstance(node_state, dict):
                    continue
                if node_name == "planner":
                    yield _encode(
                        ChatChunk(
                            type="plan",
                            content=node_state.get("intent", "chat") or "chat",
                            meta={
                                "queries": ", ".join(node_state.get("search_queries", []) or []),
                                "name": node_state.get("name", "") or "",
                            },
                        )
                    )
                elif node_name == "retriever":
                    docs = node_state.get("retrieved", []) or []
                    yield _encode(
                        ChatChunk(
                            type="retrieval",
                            content=f"retrieved {len(docs)} chunks",
                            meta={"sources": ", ".join(d["source"] for d in docs[:5])},
                        )
                    )
                elif node_name == "generator":
                    text = node_state.get("raw_output", "") or ""
                    yaml_text = node_state.get("yaml_text", "") or ""
                    if yaml_text:
                        final_yaml = yaml_text
                    yield _encode(ChatChunk(type="token", content=text))
                    if yaml_text:
                        yield _encode(ChatChunk(type="yaml", content=yaml_text))
                elif node_name == "validator":
                    errors = node_state.get("validation_errors", []) or []
                    warnings = node_state.get("validation_warnings", []) or []
                    yaml_text = node_state.get("yaml_text", "") or ""
                    if yaml_text:
                        final_yaml = yaml_text
                    yield _encode(
                        ChatChunk(
                            type="validation",
                            content="ok" if not errors else f"{len(errors)} errors",
                            meta={
                                "errors": " | ".join(errors[:5]),
                                "warnings": " | ".join(warnings[:5]),
                            },
                        )
                    )
                elif node_name == "fixer":
                    yaml_text = node_state.get("yaml_text", "") or ""
                    if yaml_text:
                        final_yaml = yaml_text
                        yield _encode(ChatChunk(type="yaml", content=yaml_text))

    except Exception as e:  # noqa: BLE001
        logger.exception("chat_stream failed")
        yield _encode(ChatChunk(type="error", content=str(e)))
        return

    if session_id and final_yaml:
        memory.set_last_yaml(session_id, final_yaml)
    yield _encode(ChatChunk(type="done", content="finished"))


async def chat_once(message: str, session_id: str | None = None) -> dict[str, Any]:
    """非流式聊天；用于测试或不支持 SSE 的客户端。"""
    from app.agent.graph import run_agent

    result = await run_agent(message)
    if session_id:
        memory = get_memory()
        memory.append_message(session_id, "user", message)
        if result.get("yaml_text"):
            memory.set_last_yaml(session_id, result["yaml_text"])
    return {
        "reply": result.get("explanation") or result.get("final_message") or "",
        "yaml": result.get("yaml_text") or None,
        "sources": [r["source"] for r in result.get("retrieved", []) or []][:5],
    }
