"""所有工具集中导出。注意：example_retriever 与 wiki_search 依赖 RAG / 向量库，
这些依赖在测试 yaml_validator / config_formatter 等纯 YAML 工具时不应是必需的，
所以这里使用惰性导入：只有真正访问到对应符号时才解析它们。
"""
from __future__ import annotations

import importlib
from typing import Any

from app.agent.tools.config_formatter import config_formatter_tool, format_yaml
from app.agent.tools.yaml_validator import validate_yaml, yaml_validator_tool

_LAZY_ATTRS = {
    "wiki_search": ("app.agent.tools.wiki_search", "wiki_search"),
    "wiki_search_dict": ("app.agent.tools.wiki_search", "wiki_search_dict"),
    "example_retriever": ("app.agent.tools.example_retriever", "example_retriever"),
}


def __getattr__(name: str) -> Any:
    """惰性属性解析；避免在没有 langchain_community 时也强制 import。"""
    if name in _LAZY_ATTRS:
        module_name, attr = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    if name == "ALL_TOOLS":
        return [
            __getattr__("wiki_search"),
            yaml_validator_tool,
            __getattr__("example_retriever"),
            config_formatter_tool,
        ]
    raise AttributeError(name)


__all__ = [
    "ALL_TOOLS",
    "wiki_search",
    "wiki_search_dict",
    "yaml_validator_tool",
    "validate_yaml",
    "example_retriever",
    "config_formatter_tool",
    "format_yaml",
]
