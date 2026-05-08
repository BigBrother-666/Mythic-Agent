"""聊天服务：把 LangGraph 事件流转换为 SSE chunks。

事件分两路：
- 节点级 `updates`：plan / retrieval / validation / yaml 等结构化事件
- LLM 级 `messages`：generator / fixer 节点的逐 token 文本流（type=token）

前端只要顺序消费 SSE 即可，不必关心两路的来源。
"""
from __future__ import annotations

from typing import Any, AsyncIterator

import orjson

from app.agent.graph import stream_events
from app.agent.memory import get_memory
from app.core.logging import logger
from app.schemas.chat import ChatChunk

# 哪些节点的 token 流值得透传给用户。
# planner 输出 JSON、validator/retriever 不调 LLM，都不必透传。
_STREAM_NODES = {"generator", "fixer"}


def _encode(chunk: ChatChunk) -> str:
    """把 ChatChunk 编码成 SSE data line。"""
    payload = orjson.dumps(chunk.model_dump()).decode()
    return f"data: {payload}\n\n"


def _node_from_metadata(metadata: dict[str, Any] | None) -> str:
    """从 LangGraph messages mode 的 metadata 中拿到节点名。"""
    if not metadata:
        return ""
    return str(metadata.get("langgraph_node") or "")


def _chunk_text(message_chunk: Any) -> str:
    """把 AIMessageChunk 的 content 提成纯文本。

    `content` 可能是 str，也可能是 [{type:'text', text:...}, ...]（部分 provider 走结构化）。
    """
    content = getattr(message_chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("text") or part.get("content")
                if isinstance(t, str):
                    out.append(t)
            elif isinstance(part, str):
                out.append(part)
        return "".join(out)
    return ""


async def chat_stream(message: str, session_id: str | None = None) -> AsyncIterator[str]:
    """聊天 SSE 主迭代器。"""
    memory = get_memory()
    # 关键：先取 snapshot 给 graph 用，**再**把当前 user message 写入历史，
    # 否则 history 会包含本轮的 user 消息（重复）。
    history: list[dict[str, str]] = []
    last_yaml = ""
    if session_id:
        snap = memory.snapshot(session_id, max_turns=3)
        history = list(snap.get("history") or [])  # type: ignore[arg-type]
        last_yaml = str(snap.get("last_yaml") or "")
        memory.append_message(session_id, "user", message)

    final_yaml = ""
    final_explanation = ""

    try:
        async for mode, payload in stream_events(
            message, history=history, last_yaml=last_yaml
        ):
            # ---- LLM token 流 ----
            if mode == "messages":
                # payload 形如 (AIMessageChunk, metadata_dict)
                if not isinstance(payload, tuple) or len(payload) != 2:
                    continue
                message_chunk, metadata = payload
                node = _node_from_metadata(metadata if isinstance(metadata, dict) else None)
                if node not in _STREAM_NODES:
                    continue
                token = _chunk_text(message_chunk)
                if not token:
                    continue
                yield _encode(ChatChunk(type="token", content=token, meta={"node": node}))
                continue

            # ---- 节点级 updates ----
            if mode != "updates" or not isinstance(payload, dict):
                continue
            for node_name, node_state in payload.items():
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
                                "followup": "yes" if node_state.get("is_followup") else "no",
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
                    # token 流已经实时透出；这里只补发一次 yaml 提取结果与 explanation
                    yaml_text = node_state.get("yaml_text", "") or ""
                    if yaml_text:
                        final_yaml = yaml_text
                        yield _encode(ChatChunk(type="yaml", content=yaml_text))
                    expl = node_state.get("explanation") or node_state.get("final_message") or ""
                    if expl:
                        final_explanation = expl
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

    # 把 assistant 的产物写回 history，让下一轮能看到
    if session_id:
        if final_yaml:
            memory.set_last_yaml(session_id, final_yaml)
        # 写一条 assistant 消息：优先存 explanation；YAML 单独走 last_yaml，不重复塞 history。
        if final_explanation or final_yaml:
            assistant_text = final_explanation or "(已生成 YAML)"
            memory.append_message(session_id, "assistant", assistant_text)
    yield _encode(ChatChunk(type="done", content="finished"))


async def chat_once(message: str, session_id: str | None = None) -> dict[str, Any]:
    """非流式聊天；用于测试或不支持 SSE 的客户端。"""
    from app.agent.graph import run_agent

    memory = get_memory()
    history: list[dict[str, str]] = []
    last_yaml = ""
    if session_id:
        snap = memory.snapshot(session_id, max_turns=3)
        history = list(snap.get("history") or [])  # type: ignore[arg-type]
        last_yaml = str(snap.get("last_yaml") or "")
        memory.append_message(session_id, "user", message)

    result = await run_agent(message, history=history, last_yaml=last_yaml)

    if session_id:
        if result.get("yaml_text"):
            memory.set_last_yaml(session_id, result["yaml_text"])
        reply = result.get("explanation") or result.get("final_message") or ""
        if reply or result.get("yaml_text"):
            memory.append_message(session_id, "assistant", reply or "(已生成 YAML)")
    return {
        "reply": result.get("explanation") or result.get("final_message") or "",
        "yaml": result.get("yaml_text") or None,
        "sources": [r["source"] for r in result.get("retrieved", []) or []][:5],
    }
