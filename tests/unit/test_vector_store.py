"""vector_store 中按字节安全截断的单元测试。

Milvus VARCHAR 的 max_length 是字节数，不是字符数；中文 1 字符 = 3 字节，
我们要确保 _truncate_bytes 不会按字符切（导致中文 chunk 实际膨胀 3 倍超限）
也不会切坏多字节字符产生乱码。
"""
from __future__ import annotations

from app.rag.vector_store import MilvusStore


def test_truncate_bytes_handles_chinese() -> None:
    """3000 个汉字 = 9000 字节，按 8000 字节限制应能正确截短不报错。"""
    text = "深" * 3000
    out = MilvusStore._truncate_bytes(text, 8000)
    assert len(out.encode("utf-8")) <= 8000
    # 不能把汉字切坏
    assert all(ord(c) > 0x4E00 for c in out)


def test_truncate_bytes_no_truncation_when_fits() -> None:
    text = "Hello 世界"
    out = MilvusStore._truncate_bytes(text, 1000)
    assert out == text


def test_truncate_bytes_drops_partial_multibyte() -> None:
    """字节边界恰好落在多字节字符中间时，errors='ignore' 应去掉残缺尾。"""
    text = "ab" + "中"  # ab + (E4 B8 AD) 共 5 字节
    out = MilvusStore._truncate_bytes(text, 3)  # ab + 1 字节 → 残缺
    assert out == "ab"


def test_truncate_bytes_empty() -> None:
    assert MilvusStore._truncate_bytes("", 100) == ""
