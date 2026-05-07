"""api 子包。"""
from app.api.routes_chat import router as chat_router
from app.api.routes_generate import router as generate_router
from app.api.routes_health import router as health_router
from app.api.routes_rag import router as rag_router
from app.api.routes_validate import router as validate_router

ALL_ROUTERS = [health_router, chat_router, generate_router, validate_router, rag_router]

__all__ = ["ALL_ROUTERS"]
