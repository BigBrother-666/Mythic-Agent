"""LangGraph: Planner → Retriever → Generator → Validator → (Fix loop) → Done。

事件流通过 `stream_events` 同时暴露两种模式：
- `updates`: 每个节点完成后的状态变更（plan / retrieved / yaml / validation 等结构化事件）
- `messages`: 节点内 LLM 调用的逐 token 流（首 token 时间显著降低）

调用方根据 `(mode, payload)` 二元组分别处理即可。
"""
from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.agent.prompts import (
    GENERATOR_PROMPT,
    PLANNER_PROMPT,
    SYSTEM_PROMPT,
    VALIDATOR_FIX_PROMPT,
    get_generator_prompt,
    render_context_block,
)
from app.agent.tools import format_yaml, validate_yaml
from app.core.config import get_settings
from app.core.logging import logger

# ---------- State ----------

Intent = Literal["mob", "item", "skill", "chat"]


class AgentState(TypedDict, total=False):
    """LangGraph 共享状态。"""

    user_input: str
    intent: Intent
    name: str
    notes: str
    search_queries: list[str]
    retrieved: list[dict[str, Any]]
    hyde_doc: str
    raw_output: str
    yaml_text: str
    explanation: str
    suggestions: list[str]
    validation_errors: list[str]
    validation_warnings: list[str]
    iterations: int
    final_message: str
    # ---- 多轮记忆字段（由 service 层注入） ----
    history: list[dict[str, str]]  # 最近若干轮 [{role, content}]
    last_yaml: str  # 上一次生成的 YAML（作为修改基线）
    is_followup: bool  # planner 判定本次是否为「修改上一份 YAML」


# ---------- LLM 工厂 ----------

_llm: Any | None = None


def get_llm() -> Any:
    """获取全局 LLM 客户端；惰性 import 避免在仅做 chunker / validator 测试时拖入 langchain_openai。"""
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI

        s = get_settings()
        _llm = ChatOpenAI(
            model=s.llm_model,
            api_key=s.openai_api_key or "EMPTY",
            base_url=s.openai_base_url,
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
            streaming=True,
        )
    return _llm


def reset_llm() -> None:
    """重置 LLM 客户端（测试用）。"""
    global _llm
    _llm = None


def _wiki_search_dict_lazy(query: str, top_k: int = 6, category: str | None = None) -> Any:
    """惰性引用 wiki_search_dict，避免在 unit 测试时拖入 RAG 依赖。"""
    from app.agent.tools.wiki_search import wiki_search_dict as _impl

    return _impl(query, top_k=top_k, category=category)


# 给 monkeypatch / 测试一个稳定的注入点。
wiki_search_dict = _wiki_search_dict_lazy


# ---------- 节点实现 ----------


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)
_YAML_BLOCK_RE = re.compile(r"```ya?ml\s*\n([\s\S]*?)```", re.MULTILINE)


def _extract_json(text: str) -> dict[str, Any]:
    """容错地从 LLM 输出中抽取 JSON 对象。"""
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # 容错：去掉行注释再试一次
        cleaned = re.sub(r"//.*", "", m.group(0))
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}


def _extract_yaml(text: str) -> str:
    """从 markdown 中抽取第一个 yaml 代码块。"""
    m = _YAML_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    return ""


def _split_explanation_suggestions(text: str) -> tuple[str, list[str]]:
    """从 generator 输出中抽取 '## 解释' 与 '## 建议'。"""
    explanation = ""
    suggestions: list[str] = []
    parts = re.split(r"^##\s+", text, flags=re.MULTILINE)
    for p in parts:
        head, _, body = p.partition("\n")
        head = head.strip().lower()
        body = body.strip()
        if head.startswith("解释") or head.startswith("explanation"):
            explanation = body
        elif head.startswith("建议") or head.startswith("suggestion"):
            suggestions = [
                line.lstrip("-* ").strip() for line in body.splitlines() if line.strip()
            ]
    return explanation, suggestions


# intent → 默认顶层 key 名
_DEFAULT_NAME_BY_INTENT = {
    "mob": "MyCustomMob",
    "item": "MyCustomItem",
    "skill": "MyCustomSkill",
    "chat": "",
}


async def planner_node(state: AgentState) -> AgentState:
    """Planner：让 LLM 解析用户意图并拟定检索计划。"""
    llm = get_llm()
    context_block = render_context_block(
        state.get("history") or [], state.get("last_yaml") or ""
    )
    prompt = PLANNER_PROMPT.format(
        user_input=state["user_input"], context_block=context_block
    )
    resp = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
    parsed = _extract_json(resp.content if isinstance(resp.content, str) else str(resp.content))
    raw_intent = parsed.get("intent")
    intent: Intent = raw_intent if raw_intent in {"mob", "item", "skill", "chat"} else "chat"
    is_followup = bool(parsed.get("is_followup")) and bool((state.get("last_yaml") or "").strip())
    default_name = _DEFAULT_NAME_BY_INTENT.get(intent, "")
    return {
        **state,
        "intent": intent,
        "name": parsed.get("name") or default_name,
        "notes": parsed.get("notes") or "",
        "search_queries": parsed.get("search_queries") or [],
        "is_followup": is_followup,
    }


async def retriever_node(state: AgentState) -> AgentState:
    """对每个 search query 取 top-k 片段，去重合并。

    intent=item 时**额外**对每条 query 在 wiki=crucible 范围里再检索一次，
    确保物品触发器 (~onUse / Recipes / Furniture) 这类只在 crucible 出现的内容能命中。
    """
    from app.rag.pipeline import get_last_hyde_doc

    settings = get_settings()
    per_query_k = settings.rag_top_k
    queries = state.get("search_queries") or [state["user_input"]]
    intent = state.get("intent", "mob")
    seen_keys: set[str] = set()
    merged: list[dict[str, Any]] = []

    async def _absorb(hits: list[dict[str, Any]]) -> None:
        for h in hits:
            key = f"{h['source']}::{h['title']}::{h['text'][:64]}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(h)

    for q in queries[:4]:  # 最多 4 条避免爆 context
        try:
            await _absorb(await wiki_search_dict(q, top_k=per_query_k))
        except Exception as e:  # noqa: BLE001
            logger.warning("wiki_search failed for {}: {}", q, e)
            continue
        if intent == "item":
            try:
                await _absorb(await wiki_search_dict(q, top_k=per_query_k // 2, category=None))
                await _absorb(
                    await wiki_search_dict(q + " crucible", top_k=per_query_k // 2)
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("crucible wiki_search failed for {}: {}", q, e)

    return {**state, "retrieved": merged[:14], "hyde_doc": get_last_hyde_doc()}


def _format_context(hits: list[dict[str, Any]], limit: int = 6000) -> str:
    """把 RAG 命中拼成 context；按 token 预算粗略截断。"""
    parts: list[str] = []
    used = 0
    for i, h in enumerate(hits, 1):
        snippet = f"[{i}] {h['source']} :: {h['title']}\n{h['text']}"
        if used + len(snippet) > limit:
            break
        parts.append(snippet)
        used += len(snippet)
    return "\n\n".join(parts) or "(no context)"


async def generator_node(state: AgentState) -> AgentState:
    """生成 YAML + 解释 + 建议；按 intent 选不同模板。"""
    llm = get_llm()
    context_block = render_context_block(
        state.get("history") or [], state.get("last_yaml") or ""
    )
    if state.get("intent") == "chat":
        # 闲聊分支：直接用 SYSTEM_PROMPT 回答，不强制 YAML 模板，但仍把历史 + last_yaml 给 LLM 看。
        chat_user = (
            (context_block + "\n用户输入：\n" + state["user_input"])
            if context_block
            else state["user_input"]
        )
        resp = await llm.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=chat_user)]
        )
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        return {**state, "raw_output": text, "yaml_text": "", "explanation": text, "suggestions": [], "final_message": text}

    intent = state.get("intent", "mob")
    name = state.get("name") or _DEFAULT_NAME_BY_INTENT.get(intent, "MyCustom")
    template = get_generator_prompt(intent)
    context = _format_context(state.get("retrieved") or [])
    prompt = template.format(
        user_input=state["user_input"],
        intent=intent,
        name=name,
        notes=state.get("notes", ""),
        context=context,
        context_block=context_block,
        is_followup=str(bool(state.get("is_followup"))).lower(),
    )
    resp = await llm.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    yaml_text = _extract_yaml(text)
    explanation, suggestions = _split_explanation_suggestions(text)
    return {
        **state,
        "raw_output": text,
        "yaml_text": yaml_text,
        "explanation": explanation,
        "suggestions": suggestions,
    }


async def validator_node(state: AgentState) -> AgentState:
    """运行 validate_yaml；记录错误与警告。"""
    yaml_text = state.get("yaml_text") or ""
    if not yaml_text.strip():
        return {**state, "validation_errors": [], "validation_warnings": []}
    intent = state.get("intent", "mob")
    kind = intent if intent in {"mob", "item", "skill"} else "auto"
    res = validate_yaml(yaml_text, kind=kind)
    if res["valid"]:
        try:
            yaml_text = format_yaml(yaml_text)
        except Exception as e:  # noqa: BLE001
            logger.debug("Format failed: {}", e)
    return {
        **state,
        "yaml_text": yaml_text,
        "validation_errors": res["errors"],
        "validation_warnings": res["warnings"],
    }


async def fixer_node(state: AgentState) -> AgentState:
    """让 LLM 根据校验错误重新输出 YAML。"""
    llm = get_llm()
    iterations = state.get("iterations", 0) + 1
    prompt = VALIDATOR_FIX_PROMPT.format(
        errors="\n".join(f"- {e}" for e in state.get("validation_errors", [])),
        yaml_text=state.get("yaml_text", ""),
    )
    resp = await llm.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    new_yaml = _extract_yaml(text) or state.get("yaml_text", "")
    return {**state, "yaml_text": new_yaml, "iterations": iterations}


def _route_after_validation(state: AgentState) -> str:
    """决定是否进 fixer 或结束。"""
    if state.get("intent") == "chat":
        return "done"
    if not state.get("validation_errors"):
        return "done"
    if state.get("iterations", 0) >= 2:
        return "done"
    return "fix"


def _route_after_planner(state: AgentState) -> str:
    """根据 intent 决定是否进 retriever。"""
    if state.get("intent") == "chat" and not state.get("search_queries"):
        return "generator"
    return "retriever"


# ---------- 图构建 ----------

_compiled = None


def build_graph():
    """构建并编译 LangGraph。"""
    global _compiled
    if _compiled is not None:
        return _compiled
    g: StateGraph = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("retriever", retriever_node)
    g.add_node("generator", generator_node)
    g.add_node("validator", validator_node)
    g.add_node("fixer", fixer_node)

    g.set_entry_point("planner")
    g.add_conditional_edges("planner", _route_after_planner, {
        "retriever": "retriever",
        "generator": "generator",
    })
    g.add_edge("retriever", "generator")
    g.add_edge("generator", "validator")
    g.add_conditional_edges("validator", _route_after_validation, {
        "fix": "fixer",
        "done": END,
    })
    g.add_edge("fixer", "validator")
    _compiled = g.compile()
    return _compiled


async def run_agent(
    user_input: str,
    *,
    history: list[dict[str, str]] | None = None,
    last_yaml: str = "",
) -> AgentState:
    """同步式调用入口（非流式）。

    history / last_yaml 由 service 层从 MemoryStore.snapshot 取出后注入；
    planner 据此判定 is_followup，generator 在 followup 时基于 last_yaml 增量改动。
    """
    graph = build_graph()
    state: AgentState = {
        "user_input": user_input,
        "iterations": 0,
        "history": history or [],
        "last_yaml": last_yaml or "",
    }
    result = await graph.ainvoke(state)
    return result


async def stream_events(
    user_input: str,
    *,
    history: list[dict[str, str]] | None = None,
    last_yaml: str = "",
) -> AsyncIterator[tuple[str, Any]]:
    """流式事件迭代器；同时透传节点更新与 LLM token。

    yield 出 `(mode, payload)`：
    - `("updates", {<node_name>: <state_delta>})` —— 节点完成事件
    - `("messages", (AIMessageChunk, metadata))` —— LLM 单个 token chunk；
      调用方可读 `metadata["langgraph_node"]` 决定是否透传给用户
      （例如只透传 `generator` / `fixer`，过滤 `planner` 内部 JSON）
    """
    graph = build_graph()
    state: AgentState = {
        "user_input": user_input,
        "iterations": 0,
        "history": history or [],
        "last_yaml": last_yaml or "",
    }
    async for mode, payload in graph.astream(
        state, stream_mode=["updates", "messages"],
    ):
        yield mode, payload
