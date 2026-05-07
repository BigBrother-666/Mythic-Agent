"""chunker 单元测试。"""
from __future__ import annotations

from app.rag.chunker import chunk_markdown, iter_wiki_files

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
