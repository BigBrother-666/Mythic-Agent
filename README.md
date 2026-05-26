# Mythic Agent

基于 LLM + RAG 的 MythicMobs YAML 配置生成助手。把官方 Wiki 切片入向量数据库；前端输入自然语言 → Agent 调用 RAG 检索 → 生成合法 YAML → 自动校验 → 逐 token 流式返回。

## 架构

```
浏览器 (HTMX SSE，逐 token 渲染，首 token 计时)
    ↓
FastAPI (uvloop, async)
    ↓
LangGraph Agent: planner → retriever → generator → validator → (fix loop)
    ↓ astream(stream_mode=["updates","messages"])  ← 节点更新 + LLM token 双轨
Tools: wiki_search、yaml_validator、example_retriever、config_formatter
    ↓
RAG: chunker → bge embedding → Milvus + BM25 (RRF 融合) → [可选 HyDE / Rerank]
```

## 本地部署

```bash
# 在wiki目录下克隆mythicmobs和mythiccrucible wiki，用于构建向量数据库
git clone https://git.lumine.io/mythiccraft/MythicMobs.wiki.git
git clone https://git.lumine.io/mythiccraft/mythiccrucible.wiki.git

# 可选：把你已有的 YAML 配置作为样例放到 examples/{mobs,items,skills}

cp .env.example .env
# 编辑 .env，配置api key、embedding模型等环境变量

# CPU 部署
docker compose up -d --build

# GPU 部署
# 默认 CUDA 12.9 wheel；
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build

# 如果启动了 langfuse
docker compose --profile langfuse up -d

# 等 milvus healthy
docker compose exec fastapi python -m scripts.ingest --drop --examples /examples
```

打开浏览器 http://localhost:8000

GPU override 会自动把 `EMBED_DEVICE` / `RERANK_DEVICE` 设为 `cuda`，无需手动改 `.env`。验证 GPU 是否生效：

```bash
docker compose exec fastapi python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

入库会同时覆盖三类来源，每个 chunk 带 `metadata.wiki` 标签：

| 来源                            | metadata.wiki | 用途                                          |
| ------------------------------- | ------------- | --------------------------------------------- |
| `wiki/MythicMobs.wiki/`         | `mythicmobs`  | mob / 技能 / 触发器 / 条件等核心 wiki         |
| `wiki/mythiccrucible.wiki/`     | `crucible`    | 物品的额外机制 / 触发器 / 家具等              |
| `examples/{mobs,items,skills}/` | `local`       | 你的本地 YAML 样例（按顶层 key 切，整段保留） |

retriever 可按 `wiki=mythicmobs|crucible|local` 过滤；缺省不过滤。

## 本地开发

```bash
python3.12 -m venv .venv && source .venv/bin/activate

# CPU 版 torch
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
# 或 GPU 版（CUDA 12.9）
pip install --extra-index-url https://download.pytorch.org/whl/cu129 -r requirements.txt
# 用 GPU 跑 embed/rerank 时改 .env：
EMBED_DEVICE=cuda
RERANK_DEVICE=cuda

# 在wiki目录下克隆mythicmobs和mythiccrucible wiki，用于构建向量数据库
git clone https://git.lumine.io/mythiccraft/MythicMobs.wiki.git
git clone https://git.lumine.io/mythiccraft/mythiccrucible.wiki.git

# 启动 milvus + redis（推荐用 docker-compose 起依赖）
docker compose up -d milvus redis etcd minio
# 如果使用了 langfuse
docker compose --profile langfuse up -d etcd minio milvus redis langfuse langfuse-db
# 单独启动 langfuse
docker compose --profile langfuse up -d

cp .env.example .env
sed -i 's/MILVUS_HOST=milvus/MILVUS_HOST=localhost/' .env
sed -i 's|REDIS_URL=redis://redis:6379/0|REDIS_URL=redis://localhost:6379/0|' .env

# 向量数据库（同时切 wiki + examples）
WIKI_ROOT=./wiki python -m scripts.ingest --drop --examples ./examples

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

## 评估 (Eval)

`eval/queries.yaml` 里有 133 条人工标注题目，覆盖 mob / item / metaskill / mechanic /
targeter / condition / trigger 7 类，中英混合。`eval/run_eval.py` 跑出 `Recall@K (K=1,3,5,8)` + `MRR`，
并把 per-query 命中明细写到 `eval/results/{timestamp}_{label}.json`。

```bash
# 按 .env 当前设置跑一次
docker compose exec fastapi python -m eval.run_eval
# 或 不使用容器 本地跑
python -m eval.run_eval

# rerank开关对比
docker compose exec fastapi python -m eval.run_eval --toggle-rerank

# 可单独覆盖
docker compose exec fastapi python -m eval.run_eval --rerank on --top-k 8
```

评估结果（N=133）：

Embed Model: BAAI/bge-small-zh-v1.5
Rerank Model: BAAI/bge-reranker-v2-m3

| metric   | rerank_off | rerank_on | Delta   |
| -------- | ---------- | --------- | ------- |
| recall@1 | 0.241      | 0.383     | +14.3pp |
| recall@3 | 0.474      | 0.617     | +14.3pp |
| recall@5 | 0.579      | 0.669     | +9.0pp  |
| recall@8 | 0.662      | 0.722     | +6.0pp  |
| mrr      | 0.376      | 0.505     | +12.9pp |

---

Embed Model: Qwen/Qwen3-Embedding-0.6B
Rerank Model: Qwen/Qwen3-Reranker-0.6B

| metric   | rerank_off | rerank_on | Delta  |
| -------- | ---------- | --------- | ------ |
| recall@1 | 0.391      | 0.444     | +5.3pp |
| recall@3 | 0.654      | 0.699     | +4.5pp |
| recall@5 | 0.767      | 0.797     | +3.0pp |
| recall@8 | 0.872      | 0.880     | +0.8pp |
| mrr      | 0.542      | 0.588     | +4.5pp |

开启HyDE: 

| metric   | rerank_off | rerank_on | Delta  |
| -------- | ---------- | --------- | ------ |
| recall@1 | 0.459      | 0.474     | +1.5pp |
| recall@3 | 0.684      | 0.699     | +1.5pp |
| recall@5 | 0.805      | 0.805     | +0.0pp |
| recall@8 | 0.887      | 0.872     | -1.5pp |
| mrr      | 0.600      | 0.605     | +0.6pp |


新增评估题目时，每条 case 至少给 1 条 `expected_sources`（substring 匹配，容忍路径前缀差异），
按 wiki 来源标注 `expected_wiki ∈ {mythicmobs, crucible, any}`。

## 流式协议

`POST /api/chat` 在 `stream:true` 时返回 SSE，`data:` 后是 JSON：

```ts
type ChatChunk = {
  type: "plan" | "retrieval" | "token" | "yaml" | "validation" | "error" | "done";
  content: string;
  meta?: Record<string, string>;  // token 事件带 meta.node ∈ {"generator","fixer"}
}
```

事件顺序示例：

```
plan       → intent=mob, queries=...                     // planner 结束
retrieval  → retrieved 6 chunks, sources=...             // retriever 结束
token      → "## YAML\n"           meta.node=generator   // ↓ generator LLM 逐 token
token      → "```yaml\nDeepSea..." meta.node=generator
token      → "  Type: GUARDIAN..." meta.node=generator
yaml       → 已格式化的完整 YAML（generator 节点结束后补发）
validation → ok / 2 errors（含 warnings/errors meta）
token      → ...                   meta.node=fixer       // 仅当校验失败、进入修复循环
yaml       → 修复后的 YAML
done
```

前端只需顺序消费 SSE：`token` 增量追加显示（`meta.node` 切换时清缓冲），`yaml` 覆盖右侧高亮区，状态栏显示首 token 时间与总耗时。

## API调用

Agent 的 planner 节点会自动根据用户描述判定 4 类意图：`mob` / `item` / `skill` / `chat`，
并选用对应的 generator prompt（mob 模板 / item 模板含 Crucible 触发器与 Recipes / skill
独立 metaskill 模板）。也可以走 `/api/generate/{kind}` 显式指定。

```bash
# SSE 流式聊天（intent 自动判定）
curl -N -X POST http://localhost:8000/api/chat \
  -H 'content-type: application/json' \
  -d '{"message":"帮我设计一个 Boss，三阶段，冰冻技能","session_id":"demo"}'

curl -N -X POST http://localhost:8000/api/chat \
  -H 'content-type: application/json' \
  -d '{"message":"做一把右键释放冲击波的剑","session_id":"demo"}'

curl -N -X POST http://localhost:8000/api/chat \
  -H 'content-type: application/json' \
  -d '{"message":"写一个范围冰冻 metaskill，带粒子和音效","session_id":"demo"}'

# 显式指定 kind
curl -X POST http://localhost:8000/api/generate/mob \
  -H 'content-type: application/json' \
  -d '{"description":"会召唤小怪并自爆的 Boss","name":"VolatileSummoner"}'

curl -X POST http://localhost:8000/api/generate/item \
  -H 'content-type: application/json' \
  -d '{"description":"右键召唤一个临时陷阱的法杖","name":"TrapStaff"}'

curl -X POST http://localhost:8000/api/generate/skill \
  -H 'content-type: application/json' \
  -d '{"description":"对周围 5 格内冰冻 3 秒并造成伤害","name":"FrostNova"}'

# RAG 检索
curl -X POST http://localhost:8000/api/rag/search \
  -H 'content-type: application/json' \
  -d '{"query":"projectile freeze","top_k":5}'

# 校验 YAML
curl -X POST http://localhost:8000/api/validate \
  -H 'content-type: application/json' \
  -d '{"yaml_text":"Demo:\n  Type: ZOMBIE\n  Health: 100"}'
```

## 配置项（.env）

| 变量                  | 默认                      | 说明                                                              |
| --------------------- | ------------------------- | ----------------------------------------------------------------- |
| OPENAI_API_KEY        | (必填)                    | LLM API key                                                       |
| OPENAI_BASE_URL       | https://api.openai.com/v1 | OpenAI Chat API 兼容端点                                          |
| LLM_MODEL             | gpt-4o-mini               | 模型名                                                            |
| LLM_TEMPERATURE       | 0.2                       | LLM 采样温度；越低越确定性                                        |
| LLM_MAX_TOKENS        | 2048                      | LLM 单次生成最大 token 数                                         |
| EMBED_MODEL           | BAAI/bge-small-zh-v1.5    | 默认轻量；可换 `BAAI/bge-m3` (1024 维)                            |
| EMBED_DIM             | 512                       | 与 EMBED_MODEL 必须匹配，bge-m3 → 1024                            |
| EMBED_DEVICE          | cpu                       | `cuda` 可显著加速入库                                             |
| EMBED_BATCH_SIZE      | 16                        | Embedding 推理批量大小                                            |
| ENABLE_RERANK         | false                     | 启用 cross-encoder 二阶段重排序                                   |
| RERANK_MODEL          | BAAI/bge-reranker-base    | rerank 模型；多语言可换 `BAAI/bge-reranker-v2-m3`                 |
| RERANK_DEVICE         | cpu                       | `cuda` 显著加速                                                   |
| RERANK_POOL_FACTOR    | 4                         | 送入 rerank 的候选数 = top_k × 此值；越大越准但越慢               |
| RERANK_MAX_LENGTH     | 512                       | Rerank 模型最大输入 token 长度                                    |
| RERANK_BATCH_SIZE     | 16                        | Rerank 推理批量大小                                               |
| MILVUS_HOST           | milvus                    | docker 内服务名                                                   |
| MILVUS_PORT           | 19530                     | Milvus gRPC 端口                                                  |
| MILVUS_COLLECTION     | mythicmobs_docs           | Milvus collection 名称                                            |
| REDIS_URL             | redis://redis:6379/0      | Redis 连接 URL                                                    |
| APP_HOST              | 0.0.0.0                   | FastAPI 监听地址                                                  |
| APP_PORT              | 8000                      | FastAPI 监听端口                                                  |
| APP_LOG_LEVEL         | INFO                      | 日志级别（DEBUG / INFO / WARNING / ERROR）                        |
| WIKI_ROOT             | /wiki                     | wiki 父目录（容器内）；下含 MythicMobs.wiki / mythiccrucible.wiki |
| MAX_PROMPT_CHARS      | 8000                      | 送入 LLM 的 RAG 上下文最大字符数                                  |
| RATE_LIMIT_PER_MINUTE | 30                        | per-IP 限流                                                       |
| RAG_TOP_K             | 8                         | 单次检索返回 chunk 数                                             |
| RAG_MAX_PER_SOURCE    | 2                         | 同一 source 最多保留的 chunk 数；防止大文档霸占结果               |
| ENABLE_HYDE           | false                     | 启用 HyDE（向量检索前用 LLM 生成假设性文档增强召回）             |
| HYDE_MODEL            | (空=复用 LLM_MODEL)      | HyDE 使用的模型；可用小模型降低延迟                               |
| HYDE_MAX_TOKENS       | 256                       | HyDE 生成假设性文档的最大 token 数                                |
| SESSION_TTL_SECONDS   | 3600                      | 会话过期时间（秒）；过期后清除对话历史                            |
| MEMORY_BACKEND        | redis                     | 短期记忆存放位置，可填写 memory 或 redis                          |
| MEMORY_REDIS_PREFIX   | mma:sess:                 | Redis key 的前缀                                                  |
| LANGFUSE_ENABLED      | false                     | 启用 LangFuse 全链路追踪                                         |
| LANGFUSE_PUBLIC_KEY   | (空)                      | LangFuse 项目 public key                                         |
| LANGFUSE_SECRET_KEY   | (空)                      | LangFuse 项目 secret key                                         |
| LANGFUSE_HOST         | http://langfuse:3000      | LangFuse 服务地址                                                 |
