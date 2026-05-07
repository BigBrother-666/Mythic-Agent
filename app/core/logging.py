"""结构化日志初始化。"""
from __future__ import annotations

import logging
import sys

from loguru import logger

from app.core.config import get_settings


class _InterceptHandler(logging.Handler):
    """把标准 logging 的输出转给 loguru，统一日志格式。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging() -> None:
    """配置全局日志输出。"""
    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.app_log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            "| <level>{level: <8}</level> "
            "| <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>"
        ),
        backtrace=False,
        diagnose=False,
    )
    logging.basicConfig(handlers=[_InterceptHandler()], level=settings.app_log_level, force=True)
    for noisy in ("uvicorn.error", "uvicorn.access", "httpx", "pymilvus"):
        logging.getLogger(noisy).handlers = [_InterceptHandler()]
        logging.getLogger(noisy).propagate = False


__all__ = ["logger", "setup_logging"]
