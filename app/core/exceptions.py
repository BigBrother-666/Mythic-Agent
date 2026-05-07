"""自定义异常。"""
from __future__ import annotations


class MythicAgentError(Exception):
    """所有应用异常的基类。"""


class ConfigError(MythicAgentError):
    """配置错误。"""


class IngestError(MythicAgentError):
    """文档入库失败。"""


class RetrievalError(MythicAgentError):
    """检索失败。"""


class ValidationError(MythicAgentError):
    """YAML 校验失败；message 中应包含具体行号信息。"""


class AgentError(MythicAgentError):
    """Agent 执行错误。"""


class RateLimitError(MythicAgentError):
    """超出限流。"""
