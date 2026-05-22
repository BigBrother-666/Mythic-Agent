"""FastAPI app 工厂；包含 lifespan、限流、模板挂载。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api import ALL_ROUTERS
from app.core.cache import close_cache, get_cache
from app.core.config import get_settings
from app.core.logging import logger, setup_logging
from app.rag.pipeline import warmup as rag_warmup
from app.schemas.common import APIResponse

_FRONTEND_DIR = Path(__file__).parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """应用启动与关闭钩子。"""
    setup_logging()
    settings = get_settings()
    logger.info("Starting MythicMobs Agent (model={})", settings.llm_model)
    await get_cache()
    try:
        await rag_warmup()
    except Exception as e:  # noqa: BLE001
        logger.warning("RAG warmup failed (BM25 will lazy-build later): {}", e)
    yield
    await close_cache()
    from app.core.tracing import flush_langfuse

    flush_langfuse()
    logger.info("Stopped")


def _rate_limit_key(request: Request) -> str:
    """简单 IP 维度限流 key。"""
    forwarded = request.headers.get("x-forwarded-for")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "anon")
    return f"rl:{ip}:{request.url.path}"


def create_app() -> FastAPI:
    """app 工厂。"""
    settings = get_settings()
    app = FastAPI(
        title="MythicMobs Agent",
        description="基于 LLM + RAG 的 MythicMobs YAML 配置生成助手",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def length_and_rate_limit(request: Request, call_next):  # type: ignore[no-untyped-def]
        """请求长度限制 + Redis 滑动窗口限流。"""
        if request.method == "POST":
            cl = request.headers.get("content-length")
            if cl and int(cl) > settings.max_prompt_chars * 4:
                return JSONResponse(
                    status_code=413,
                    content=APIResponse.fail("请求体过大").model_dump(),
                )
            cache = await get_cache()
            key = _rate_limit_key(request)
            count = await cache.incr_window(key, window_seconds=60)
            if count > settings.rate_limit_per_minute:
                return JSONResponse(
                    status_code=429,
                    content=APIResponse.fail(f"超出限流 ({settings.rate_limit_per_minute}/min)").model_dump(),
                )
        return await call_next(request)

    @app.exception_handler(HTTPException)
    async def http_exc_handler(request: Request, exc: HTTPException):  # type: ignore[no-untyped-def]
        return JSONResponse(
            status_code=exc.status_code,
            content=APIResponse.fail(str(exc.detail)).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception):  # type: ignore[no-untyped-def]
        logger.exception("Unhandled error at {}", request.url.path)
        return JSONResponse(
            status_code=500,
            content=APIResponse.fail(f"内部错误: {exc}").model_dump(),
        )

    for r in ALL_ROUTERS:
        app.include_router(r, prefix="/api")

    # ---------- 静态前端 ----------
    static_dir = _FRONTEND_DIR / "static"
    templates_dir = _FRONTEND_DIR / "templates"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    if templates_dir.exists():
        templates = Jinja2Templates(directory=str(templates_dir))

        @app.get("/", include_in_schema=False)
        async def index(request: Request):  # type: ignore[no-untyped-def]
            return templates.TemplateResponse(request, "index.html")

    return app


app = create_app()
