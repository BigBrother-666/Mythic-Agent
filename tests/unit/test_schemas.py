"""schemas / APIResponse 测试。"""
from __future__ import annotations

from app.schemas.common import APIResponse
from app.schemas.generate import GenerateRequest


def test_api_response_ok() -> None:
    r = APIResponse.ok({"k": 1}, message="great")
    assert r.success is True
    assert r.message == "great"
    assert r.data == {"k": 1}


def test_api_response_fail() -> None:
    r = APIResponse.fail("oops")
    assert r.success is False
    assert r.message == "oops"


def test_generate_request_validation() -> None:
    req = GenerateRequest(description="A boss with three phases")
    assert req.description.startswith("A boss")
    assert req.constraints == []
