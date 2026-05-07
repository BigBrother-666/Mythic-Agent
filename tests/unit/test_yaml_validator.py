"""yaml validator + formatter 单元测试。"""
from __future__ import annotations

from app.agent.tools import format_yaml, validate_yaml


VALID_MOB = """
DeepSeaBoss:
  Type: GUARDIAN
  Display: '&b深海守卫'
  Health: 500
  Damage: 12
  Skills:
  - skill{s=FrostNova} ~onTimer:60
"""

VALID_ITEM = """
FrostBlade:
  Material: DIAMOND_SWORD
  Display: '&b冰霜之刃'
  Lore:
  - '&7一把寒冰打造的剑'
  Attributes:
    MainHand:
      Damage: 12
"""

VALID_SKILL = """
NovaBurst:
  Skills:
  - particles{p=cloud;a=20} @self
  - damage{a=5} @entitiesinradius{r=4}
"""

INVALID_SYNTAX = """
DeepSeaBoss:
  Type: GUARDIAN
   Health: 100  # 缩进错误
"""

MISSING_TYPE = """
DeepSeaBoss:
  Display: '名字'
  Health: 200
"""

DEPRECATED_MAXHEALTH = """
OldMob:
  Type: ZOMBIE
  MaxHealth: 100
"""


def test_valid_mob() -> None:
    res = validate_yaml(VALID_MOB)
    assert res["valid"] is True
    assert res["kind"] == "mob"
    assert not res["errors"]


def test_valid_item() -> None:
    res = validate_yaml(VALID_ITEM)
    assert res["valid"] is True
    assert res["kind"] == "item"


def test_valid_skill() -> None:
    res = validate_yaml(VALID_SKILL)
    assert res["valid"] is True
    assert res["kind"] == "skill"


def test_invalid_syntax() -> None:
    res = validate_yaml(INVALID_SYNTAX)
    assert res["valid"] is False
    assert any("解析失败" in e or "parse" in e.lower() for e in res["errors"])


def test_missing_required_type() -> None:
    res = validate_yaml(MISSING_TYPE, kind="mob")
    assert res["valid"] is False
    assert any("Type" in e for e in res["errors"])


def test_deprecated_warning() -> None:
    res = validate_yaml(DEPRECATED_MAXHEALTH, kind="mob")
    assert any("MaxHealth" in w for w in res["warnings"])


def test_format_yaml_indent() -> None:
    formatted = format_yaml(VALID_MOB)
    # 顶层 Type 必须排在 Display 前面
    type_idx = formatted.find("Type:")
    display_idx = formatted.find("Display:")
    assert 0 <= type_idx < display_idx
