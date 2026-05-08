"""chunker 单元测试。"""
from __future__ import annotations

from app.rag.chunker import chunk_markdown, chunk_yaml_file, iter_wiki_files

SAMPLE = """# Projectile Skill

## Description
The Projectile skill fires a meta-projectile.

It supports many options.

## Attributes
| Attribute | Default |
|-----------|---------|
| Velocity  | 5       |

## Examples
A complete example:

```yaml
ShadowBolt:
  Skills:
  - projectile{onTick=ParticleEffect;v=8;mr=30} @target
```

And another inline:

```yml
FrostBolt:
  Type: SKELETON
  Skills:
  - projectile @target
```

End of file.
"""


def test_chunker_extracts_yaml_intact() -> None:
    chunks = chunk_markdown(SAMPLE, source="Skills/Mechanics/projectile.md")
    assert len(chunks) >= 4
    yaml_chunks = [c for c in chunks if c.has_yaml]
    assert len(yaml_chunks) == 2
    # YAML 块必须保留完整结构
    joined = "\n".join(c.text for c in yaml_chunks)
    assert "ShadowBolt" in joined
    assert "FrostBolt" in joined
    assert "projectile{onTick=ParticleEffect;v=8;mr=30}" in joined


def test_chunker_metadata_category() -> None:
    chunks = chunk_markdown(SAMPLE, source="Skills/Mechanics/projectile.md")
    for c in chunks:
        assert c.metadata["category"] == "mechanics"
        assert c.metadata["source"] == "Skills/Mechanics/projectile.md"
        assert isinstance(c.metadata["tags"], list)
        # 标题或文件名 token 至少有一个
        assert any(t in {"projectile", "skills", "mechanics"} for t in c.metadata["tags"])


def test_chunker_handles_no_headings() -> None:
    text = "Just a plain paragraph with no heading.\n\nAnother one."
    chunks = chunk_markdown(text, source="Misc/plain.md")
    assert len(chunks) >= 1
    assert chunks[0].text  # 非空


def test_chunker_no_split_inside_fence() -> None:
    text = (
        "## Section\n"
        "```yaml\n"
        "Mob:\n"
        "  ## fake heading inside fence\n"
        "  Type: ZOMBIE\n"
        "```\n"
    )
    chunks = chunk_markdown(text, source="Mobs/x.md")
    yaml_chunks = [c for c in chunks if c.has_yaml]
    assert len(yaml_chunks) == 1
    assert "## fake heading inside fence" in yaml_chunks[0].text


def test_iter_wiki_files_skips_uploads(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "Mobs").mkdir()
    (tmp_path / "Mobs" / "Demon.md").write_text("# Demon\nbody")
    (tmp_path / "uploads").mkdir()
    (tmp_path / "uploads" / "image.md").write_text("ignored")
    files = iter_wiki_files(tmp_path)
    assert len(files) == 1
    assert files[0].name == "Demon.md"


# ---------- 多 wiki / 本地 examples ----------

def test_chunk_markdown_infers_wiki_label_mythicmobs() -> None:
    chunks = chunk_markdown(SAMPLE, source="MythicMobs.wiki/Skills/Mechanics/projectile.md")
    assert chunks
    assert chunks[0].metadata["wiki"] == "mythicmobs"
    assert chunks[0].metadata["category"] == "mechanics"


def test_chunk_markdown_infers_wiki_label_crucible() -> None:
    chunks = chunk_markdown(SAMPLE, source="mythiccrucible.wiki/Skills/Mechanics/foo.md")
    assert chunks
    assert chunks[0].metadata["wiki"] == "crucible"
    assert chunks[0].metadata["category"] == "mechanics"


def test_chunk_yaml_file_splits_by_top_keys() -> None:
    text = """\
DeepSeaBoss:
  Type: GUARDIAN
  Health: 1200
  Skills:
  - skill{s=FrostBreath} @target

DeepSeaBoss_FrostBreath:
  Skills:
  - freeze{ticks=80} @target

# 注释行
DeepSeaThrall:
  Type: DROWNED
  Health: 80
"""
    chunks = chunk_yaml_file(text, source="examples/mobs/deep_sea.yml")
    titles = [c.metadata["title"] for c in chunks]
    assert "DeepSeaBoss" in titles
    assert "DeepSeaBoss_FrostBreath" in titles
    assert "DeepSeaThrall" in titles
    assert all(c.metadata["wiki"] == "local" for c in chunks)
    assert all(c.metadata["category"] == "examples" for c in chunks)
    assert all(c.has_yaml for c in chunks)
    # YAML 内容必须完整保留
    boss = next(c for c in chunks if c.metadata["title"] == "DeepSeaBoss")
    assert "Type: GUARDIAN" in boss.text
    assert "Health: 1200" in boss.text


def test_chunk_yaml_file_empty_returns_empty() -> None:
    assert chunk_yaml_file("", source="examples/x.yml") == []
    assert chunk_yaml_file("# only comment\n", source="examples/x.yml") == []
