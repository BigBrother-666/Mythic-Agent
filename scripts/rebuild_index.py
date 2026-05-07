"""删除 Milvus collection 并重建索引。"""
from __future__ import annotations

import asyncio

from app.core.logging import setup_logging
from app.rag.vector_store import get_milvus_store


async def main() -> None:
    """drop & recreate empty collection。"""
    setup_logging()
    store = get_milvus_store()
    store.drop()
    store.create_collection(drop_existing=True)


if __name__ == "__main__":
    asyncio.run(main())
