"""结构化 Markdown 切分器；按 heading 与 YAML 代码块切，保留父级 heading 路径。

绝不在围栏代码块内切分；尽量保留 YAML 块完整性，便于 LLM 在生成时可以参考完整示例。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ----- 正则 -----
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_FENCE_RE = re.compile(r"^(```|~~~)\s*([a-zA-Z0-9_+-]*)\s*$")
_TAG_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")

# ----- 顶级目录到 category 的映射 -----
_CATEGORY_MAP = {
    "skills": "mechanics",          # 小写历史目录
    "Skills": "mechanics",          # 大写主目录；下面会根据子目录细化
    "Mobs": "mobs",
    "Items": "items",
    "drops": "drops",
    "examples": "examples",
    "Guides": "guides",
    "config": "config",
    "templates": "templates",
    "Enum": "enum",
    "API": "api",
    "Troubleshooting": "troubleshooting",
}

_SKILLS_SUB_MAP = {
    "Mechanics": "mechanics",
    "Targeters": "targeters",
    "Conditions": "conditions",
    "Triggers": "triggers",
    "Placeholders": "placeholders",
    "Tags": "tags",
}


@dataclass
class Chunk:
    """单个 chunk；包含文本与 metadata。"""

    text: str
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def has_yaml(self) -> bool:
        return bool(self.metadata.get("has_yaml"))


def _infer_category(source: str) -> str:
    """从源路径推断 category。"""
    parts = Path(source).parts
    if not parts:
        return "misc"
    top = parts[0]
    if top == "Skills" and len(parts) >= 2:
        sub = parts[1]
        return _SKILLS_SUB_MAP.get(sub, "mechanics")
    return _CATEGORY_MAP.get(top, top.lower())


def _extract_tags(headings: list[str], source: str) -> list[str]:
    """从 heading 路径与文件名提取 tags。"""
    bag: set[str] = set()
    for h in headings:
        for token in _TAG_TOKEN_RE.findall(h):
            if len(token) > 2:
                bag.add(token.lower())
    stem = Path(source).stem
    for token in _TAG_TOKEN_RE.findall(stem):
        if len(token) > 2:
            bag.add(token.lower())
    return sorted(bag)


def _split_by_blocks(text: str) -> list[tuple[str, str]]:
    """把文本按代码围栏切成 (kind, body) 段，kind in {"text","code:<lang>"}。
    保证围栏块完整不被切。"""
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    in_fence = False
    fence_marker = ""
    fence_lang = ""
    for line in text.splitlines(keepends=False):
        m = _FENCE_RE.match(line)
        if m and (not in_fence or m.group(1) == fence_marker):
            if not in_fence:
                if buf:
                    out.append(("text", "\n".join(buf)))
                    buf = []
                in_fence = True
                fence_marker = m.group(1)
                fence_lang = (m.group(2) or "").lower()
                buf.append(line)
            else:
                buf.append(line)
                out.append((f"code:{fence_lang}", "\n".join(buf)))
                buf = []
                in_fence = False
                fence_marker = ""
                fence_lang = ""
        else:
            buf.append(line)
    if buf:
        out.append(("code" if in_fence else "text", "\n".join(buf)))
    return out


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """按段落软切超长文本块。"""
    if len(text) <= max_chars:
        return [text]
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paragraphs:
        if not p.strip():
            continue
        if cur_len + len(p) + 2 > max_chars and cur:
            chunks.append("\n\n".join(cur))
            cur = [p]
            cur_len = len(p)
        else:
            cur.append(p)
            cur_len += len(p) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def chunk_markdown(
    text: str,
    source: str,
    *,
    max_chunk_chars: int = 1500,
    min_chunk_chars: int = 80,
) -> list[Chunk]:
    """结构化切分一个 markdown 文档。

    步骤：
    1. 按行扫描，遇到 heading 切节，遇到围栏块整体吸收。
    2. 每个节内部如超过 max_chunk_chars 再按段落软切；YAML 块单独成 chunk，永不被切碎。
    3. 每个 chunk 注入 metadata。
    """
    category = _infer_category(source)
    file_title = Path(source).stem.replace("-", " ").replace("_", " ")

    sections: list[tuple[list[str], str]] = []  # (heading_path, body_text)
    heading_stack: list[str] = []
    body_lines: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        body = "\n".join(body_lines).strip("\n")
        if body.strip():
            sections.append((list(heading_stack), body))

    for line in text.splitlines():
        if not in_fence:
            f = _FENCE_RE.match(line)
            if f:
                in_fence = True
                fence_marker = f.group(1)
                body_lines.append(line)
                continue
            h = _HEADING_RE.match(line)
            if h:
                flush()
                body_lines = []
                level = len(h.group(1))
                heading_text = h.group(2).strip()
                heading_stack = heading_stack[: level - 1]
                while len(heading_stack) < level - 1:
                    heading_stack.append("")
                heading_stack.append(heading_text)
                continue
            body_lines.append(line)
        else:
            body_lines.append(line)
            if line.strip().startswith(fence_marker):
                in_fence = False
                fence_marker = ""
    flush()

    chunks: list[Chunk] = []
    for headings, body in sections:
        clean_headings = [h for h in headings if h]
        title = clean_headings[0] if clean_headings else file_title
        heading_breadcrumb = " / ".join(clean_headings) if clean_headings else file_title
        prefix = f"# {heading_breadcrumb}\n\n" if clean_headings else ""

        blocks = _split_by_blocks(body)
        # 把 text 与 code 块按相邻聚合：每个 code 块保持完整，text 块如超长则拆分。
        for kind, payload in blocks:
            if not payload.strip():
                continue
            if kind.startswith("code"):
                lang = kind.split(":", 1)[1] if ":" in kind else ""
                has_yaml = lang in {"yml", "yaml"}
                chunk_text = prefix + payload.strip()
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            "category": category,
                            "source": source,
                            "title": title,
                            "headings": clean_headings,
                            "tags": _extract_tags(clean_headings, source),
                            "has_yaml": has_yaml,
                            "kind": "code",
                            "lang": lang,
                        },
                    )
                )
            else:
                for piece in _split_long_text(payload.strip(), max_chunk_chars):
                    if len(piece) < min_chunk_chars and len(piece) < len(payload):
                        continue
                    chunks.append(
                        Chunk(
                            text=prefix + piece,
                            metadata={
                                "category": category,
                                "source": source,
                                "title": title,
                                "headings": clean_headings,
                                "tags": _extract_tags(clean_headings, source),
                                "has_yaml": False,
                                "kind": "text",
                                "lang": "",
                            },
                        )
                    )

    return chunks


def iter_wiki_files(wiki_root: Path) -> list[Path]:
    """遍历 wiki 目录下所有 markdown 文件；过滤掉 uploads / API / .git / .gitlab。"""
    skip = {"uploads", ".git", ".gitlab", "API", "Wiki-Editors-Resources"}
    files: list[Path] = []
    for p in wiki_root.rglob("*.md"):
        rel = p.relative_to(wiki_root)
        if any(part in skip for part in rel.parts):
            continue
        files.append(p)
    return sorted(files)
