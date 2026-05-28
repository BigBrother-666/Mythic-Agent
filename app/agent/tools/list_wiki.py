"""List tools —— 列出 wiki 中的 mechanics/targeters/triggers 文件名。"""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from app.core.config import get_settings


def _list_md_files(*dirs: Path) -> list[str]:
    """收集多个目录下的 .md 文件名（不含扩展名，不递归子目录），去重排序。"""
    names: set[str] = set()
    for d in dirs:
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() == ".md":
                names.add(f.stem)
    return sorted(names, key=str.lower)


@tool("list_mechanics")
def list_mechanics() -> str:
    """列出所有可用的 MythicMobs + Crucible Mechanics 名称。
    使用场景：想知道有哪些 mechanic 可用，或确认某个 mechanic 名称是否存在。
    返回值为换行分隔的 mechanic 名称列表（来自 wiki 文件名）。
    """
    s = get_settings()
    dirs = [
        s.wiki_root / "MythicMobs.wiki" / "Skills" / "Mechanics",
        s.wiki_root / "mythiccrucible.wiki" / "Skills" / "Mechanics",
    ]
    names = _list_md_files(*dirs)
    if not names:
        return "(no mechanics found)"
    return "\n".join(names)


@tool("list_targeters")
def list_targeters() -> str:
    """列出所有可用的 MythicMobs + Crucible Targeters 名称。
    使用场景：想知道有哪些 targeter 可用，或确认某个 targeter 名称是否存在。
    返回值为换行分隔的 targeter 名称列表（来自 wiki 文件名）。
    """
    s = get_settings()
    dirs = [
        s.wiki_root / "MythicMobs.wiki" / "Skills" / "Targeters",
        s.wiki_root / "mythiccrucible.wiki" / "Skills" / "Targeters",
    ]
    names = _list_md_files(*dirs)
    if not names:
        return "(no targeters found)"
    return "\n".join(names)


@tool("list_triggers")
def list_triggers() -> str:
    """列出所有可用的 MythicMobs + Crucible Triggers 名称。
    使用场景：想知道有哪些 trigger 可用，或确认某个 trigger 名称是否存在。
    返回值为换行分隔的 trigger 名称列表（来自 wiki 文件名）。
    """
    s = get_settings()
    dirs = [
        s.wiki_root / "MythicMobs.wiki" / "Skills" / "Triggers",
        s.wiki_root / "mythiccrucible.wiki" / "Skills" / "Triggers",
    ]
    names = _list_md_files(*dirs)
    if not names:
        return "(no triggers found)"
    return "\n".join(names)


@tool("list_conditions")
def list_conditions() -> str:
    """列出所有可用的 MythicMobs + Crucible Conditions 名称。
    使用场景：想知道有哪些 condition 可用，或确认某个 condition 名称是否存在。
    返回值为换行分隔的 condition 名称列表（来自 wiki 文件名）。
    """
    s = get_settings()
    dirs = [
        s.wiki_root / "MythicMobs.wiki" / "Skills" / "Conditions",
        s.wiki_root / "mythiccrucible.wiki" / "Skills" / "Conditions",
    ]
    names = _list_md_files(*dirs)
    if not names:
        return "(no conditions found)"
    return "\n".join(names)
