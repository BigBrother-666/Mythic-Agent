"""所有 API 共用的响应结构。"""
from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """统一 API 响应结构。"""

    success: bool = Field(default=True, description="请求是否成功")
    message: str = Field(default="ok", description="附加说明")
    data: T | None = Field(default=None, description="实际返回数据")

    @classmethod
    def ok(cls, data: T | None = None, message: str = "ok") -> "APIResponse[T]":
        """构造成功响应。"""
        return cls(success=True, message=message, data=data)

    @classmethod
    def fail(cls, message: str, data: Any = None) -> "APIResponse[Any]":
        """构造失败响应。"""
        return cls(success=False, message=message, data=data)
