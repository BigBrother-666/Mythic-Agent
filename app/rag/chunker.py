"""结构化 Markdown 切分器；按 heading 与 YAML 代码块切，保留父级 heading 路径。

绝不在围栏代码块内切分；尽量保留 YAML 块完整性，便于 LLM 在生成时可以参考完整示例。

source 路径约定（统一使用相对于 ingest root 的相对路径）：
- `MythicMobs.wiki/Skills/Mechanics/projectile.md`     —— mythicmobs wiki
- `mythiccrucible.wiki/Skills/Mechanics/xxx.md`        —— crucible wiki
- `examples/items/foo.yml`                              —— 本地示例 yml
metadata.wiki ∈ {"mythicmobs","crucible","local"}，retriever 可按此过滤。
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


def _infer_wiki(source: str) -> str:
    """从相对路径推断属于哪个 wiki 源。"""
    parts = Path(source).parts
    if not parts:
        return "local"
    head = parts[0].lower()
    if head.startswith("mythicmobs"):
        return "mythicmobs"
    if head.startswith("mythiccrucible"):
        return "crucible"
    if head.startswith("examples"):
        return "local"
    # 兼容旧的 ingest 直接传 wiki 子目录的情况
    return "mythicmobs"


def _strip_wiki_prefix(source: str) -> tuple[str, str]:
    """剥掉 `MythicMobs.wiki/` 这种前缀，返回 (wiki_label, rest_path)。"""
    parts = Path(source).parts
    if not parts:
        return "local", source
    head = parts[0].lower()
    rest = str(Path(*parts[1:])) if len(parts) > 1 else ""
    if head.startswith("mythicmobs"):
        return "mythicmobs", rest
    if head.startswith("mythiccrucible"):
        return "crucible", rest
    if head.startswith("examples"):
        return "local", str(Path(*parts))  # examples 路径完整保留
    return "mythicmobs", source


@dataclass
class Chunk:
    """单个 chunk；包含文本与 metadata。"""

    text: str
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def has_yaml(self) -> bool:
        return bool(self.metadata.get("has_yaml"))


def _infer_category(source: str) -> str:
    """从源路径推断 category。

    支持的形态：
    - `MythicMobs.wiki/Skills/Mechanics/...` → mechanics
    - `mythiccrucible.wiki/Skills/Mechanics/...` → mechanics
    - `examples/items/foo.yml` → examples
    - 直接传 wiki 子目录（无 wiki prefix）也兼容旧调用
    """
    parts = Path(source).parts
    if not parts:
        return "misc"
    # 跳过 wiki 前缀
    head_lower = parts[0].lower()
    if head_lower.startswith("mythicmobs") or head_lower.startswith("mythiccrucible"):
        parts = parts[1:]
        if not parts:
            return "misc"
    top = parts[0]
    if top == "Skills" and len(parts) >= 2:
        sub = parts[1]
        return _SKILLS_SUB_MAP.get(sub, "mechanics")
    if len(parts) == 1:
        # wiki 根目录下的散文档（changelog / 概览 / FAQ 之类）
        return "guides"
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
    wiki_label = _infer_wiki(source)
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
                            "wiki": wiki_label,
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
                                "wiki": wiki_label,
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


def iter_yaml_files(root: Path) -> list[Path]:
    """遍历目录下所有 .yml / .yaml 文件。"""
    files: list[Path] = []
    for ext in ("*.yml", "*.yaml"):
        files.extend(root.rglob(ext))
    return sorted(files)


def chunk_yaml_file(text: str, source: str, *, max_chunk_chars: int = 4000) -> list[Chunk]:
    """把一个独立的 YAML 文件切成 chunks。

    策略：按顶层 key（无缩进的 `XxxName:`）切；每个顶层对象一个 chunk，超长再按段落软切。
    metadata.has_yaml=True，category 推断自路径（examples/items → examples），
    wiki 推断为 local。

    这样 example_retriever 检索时能直接拿到完整可粘贴的样例。
    """
    category = _infer_category(source)
    wiki_label = _infer_wiki(source)
    file_title = Path(source).stem

    # 按「行首非空白字符开始的一行（顶层 key）」切。注释行 / 空行附给前一段。
    blocks: list[tuple[str, list[str]]] = []  # (key, lines)
    cur_key: str | None = None
    cur_lines: list[str] = []
    for line in text.splitlines():
        if line and not line[0].isspace() and ":" in line and not line.lstrip().startswith("#"):
            if cur_key is not None:
                blocks.append((cur_key, cur_lines))
            cur_key = line.split(":", 1)[0].strip()
            cur_lines = [line]
        else:
            cur_lines.append(line)
    if cur_key is not None:
        blocks.append((cur_key, cur_lines))

    chunks: list[Chunk] = []
    for key, lines in blocks:
        body = "\n".join(lines).strip("\n")
        if not body.strip():
            continue
        wrapped_pieces: list[str]
        if len(body) <= max_chunk_chars:
            wrapped_pieces = [body]
        else:
            wrapped_pieces = _split_long_text(body, max_chunk_chars)
        for piece in wrapped_pieces:
            chunk_text = f"# {file_title} :: {key}\n\n```yaml\n{piece}\n```"
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "category": category,
                        "source": source,
                        "title": key,
                        "headings": [file_title, key],
                        "tags": _extract_tags([key], source),
                        "has_yaml": True,
                        "kind": "code",
                        "lang": "yaml",
                        "wiki": wiki_label,
                    },
                )
            )
    return chunks
