"""应用全局配置——基于 pydantic-settings，从 .env 读取。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置；使用环境变量或 .env 注入。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- LLM ----------
    openai_api_key: str = Field(default="")
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    llm_model: str = Field(default="gpt-4o-mini")
    llm_temperature: float = Field(default=0.2)
    llm_max_tokens: int = Field(default=2048)

    # ---------- Embedding ----------
    embed_model: str = Field(default="BAAI/bge-small-zh-v1.5")
    embed_dim: int = Field(default=512)
    embed_device: str = Field(default="cpu")

    # ---------- Reranker (可选) ----------
    # ENABLE_RERANK=true 时，对融合后的候选做 cross-encoder 重排序
    # bge-reranker-base ≈ 280MB，多语言版用 bge-reranker-v2-m3（≈ 600MB）
    rerank_model: str = Field(default="BAAI/bge-reranker-base")
    rerank_device: str = Field(default="cpu")
    rerank_pool_factor: int = Field(default=4)  # 送入 rerank 的候选数 = top_k * 此倍数
    rerank_max_length: int = Field(default=512)
    rerank_batch_size: int = Field(default=16)

    # ---------- Milvus ----------
    milvus_host: str = Field(default="milvus")
    milvus_port: int = Field(default=19530)
    milvus_collection: str = Field(default="mythicmobs_docs")

    # ---------- Redis ----------
    redis_url: str = Field(default="redis://redis:6379/0")

    # ---------- App ----------
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    app_log_level: str = Field(default="INFO")
    wiki_root: Path = Field(default=Path("/wiki"))
    max_prompt_chars: int = Field(default=8000)
    rate_limit_per_minute: int = Field(default=30)
    enable_rerank: bool = Field(default=False)
    rag_top_k: int = Field(default=8)
    session_ttl_seconds: int = Field(default=3600)

    @property
    def milvus_uri(self) -> str:
        """Milvus 连接 URI；MilvusClient 接收的格式。"""
        return f"http://{self.milvus_host}:{self.milvus_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取单例配置对象。"""
    return Settings()
