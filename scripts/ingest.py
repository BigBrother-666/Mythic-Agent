"""一次性把整个 wiki 入库到 Milvus。

用法（在容器内或本地）：
    python -m scripts.ingest --drop          # 删表重建
    python -m scripts.ingest                 # 增量（如果 collection 已存在则在其后追加）
"""
from __future__ import annotations

import asyncio

from app.rag.pipeline import main_cli

if __name__ == "__main__":
    asyncio.run(main_cli())
