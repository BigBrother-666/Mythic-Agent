# MythicMobs Agent

基于 LLM + RAG 的 MythicMobs YAML 配置生成助手。把官方 Wiki 切片入向量数据库；前端输入自然语言 → Agent 调用 RAG 检索 → 生成合法 YAML → 自动校验 → 流式返回。

## 架构

```
浏览器 (HTMX SSE)
    ↓
FastAPI (uvloop, async)
    ↓
LangGraph Agent: planner → retriever → generator → validator → (fix loop)
    ↓
Tools: wiki_search · yaml_validator · example_retriever · config_formatter · version_compat
    ↓
RAG: chunker → bge embedding → Milvus + BM25 (RRF 融合)
```

## 一键启动

```bash
cp .env.example .env
# 编辑 .env，配置api key、embedding模型等环境变量
docker compose up -d --build
# 等 milvus healthy（首次约 30s）
docker compose exec fastapi python -m scripts.ingest --drop
```

打开浏览器 http://localhost:8000

## 直接调 API

```bash
# SSE 流式聊天
curl -N -X POST http://localhost:8000/api/chat \
  -H 'content-type: application/json' \
  -d '{"message":"帮我设计一个 Boss，三阶段，冰冻技能","session_id":"demo"}'

# 直接生成 mob
curl -X POST http://localhost:8000/api/generate/mob \
  -H 'content-type: application/json' \
  -d '{"description":"会召唤小怪并自爆的 Boss","name":"VolatileSummoner"}'

# RAG 检索
curl -X POST http://localhost:8000/api/rag/search \
  -H 'content-type: application/json' \
  -d '{"query":"projectile freeze","top_k":5}'

# 校验 YAML
curl -X POST http://localhost:8000/api/validate \
  -H 'content-type: application/json' \
  -d '{"yaml_text":"Demo:\n  Type: ZOMBIE\n  Health: 100"}'
```

## 本地开发

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# 启动 milvus + redis（推荐用 docker-compose 起依赖）
docker compose up -d milvus redis etcd minio

cp .env.example .env
sed -i 's/MILVUS_HOST=milvus/MILVUS_HOST=localhost/' .env
sed -i 's|REDIS_URL=redis://redis:6379/0|REDIS_URL=redis://localhost:6379/0|' .env

# 向量数据库
WIKI_ROOT=./MythicMobs.wiki python -m scripts.ingest --drop

# 启动 FastAPI
uvicorn app.main:app --reload --loop uvloop
```

## 测试

```bash
pytest tests/unit -q
pytest tests/ -q       # 含集成测试（mock 掉外部依赖）
ruff check .
mypy app
```

## 配置项（.env）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| OPENAI_API_KEY | (必填) | LLM API key |
| OPENAI_BASE_URL | https://api.openai.com/v1 | 兼容端点 (DeepSeek/Moonshot/Ollama 也行) |
| LLM_MODEL | gpt-4o-mini | 模型名 |
| EMBED_MODEL | BAAI/bge-small-zh-v1.5 | 默认轻量；可换 `BAAI/bge-m3` (1024 维) |
| EMBED_DIM | 512 | 与 EMBED_MODEL 必须匹配，bge-m3 → 1024 |
| EMBED_DEVICE | cpu | `cuda` 可显著加速入库 |
| MILVUS_HOST | milvus | docker 内服务名 |
| RAG_TOP_K | 8 | 单次检索返回 chunk 数 |
| ENABLE_RERANK | false | 预留 bge-reranker hook |
| RATE_LIMIT_PER_MINUTE | 30 | per-IP 限流 |
