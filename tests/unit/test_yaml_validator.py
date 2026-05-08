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


# ---------- 回归测试 ----------

# 真实 LLM 输出形态：mob 主体 + 多个 metaskill 写在同一份 YAML 里。
# 修复前 _detect_kind 仅看「第一个 value」推断为 mob，把 _FrostBreath 等也当 mob 校验，
# 导致 5 条「缺少必填字段 Type」假错。
MOB_WITH_METASKILLS = """
AbyssalLeviathan:
  Type: DROWNED
  Health: 1200
  Skills:
  - skill{s=AbyssalLeviathan_FrostBreath} @target ~onTimer:120
  - skill{s=AbyssalLeviathan_SummonThralls} @self ~onTimer:260

AbyssalThrall:
  Type: DROWNED
  Health: 80

AbyssalLeviathan_FrostBreath:
  Skills:
  - freeze{ticks=80} @target
  - damage{a=10} @target

AbyssalLeviathan_SummonThralls:
  Skills:
  - summon{type=AbyssalThrall;amount=3} @self
"""


def test_mixed_mob_and_metaskills_is_valid() -> None:
    """关键回归：mob + metaskill 混写不应报「缺少 Type」错误。"""
    res = validate_yaml(MOB_WITH_METASKILLS, kind="mob")
    assert res["valid"] is True, f"unexpected errors: {res['errors']}"
    # 已经按各自形态校验，所以 metaskill 不会被当成缺 Type 的 mob
    assert not any("缺少必填字段 Type" in e for e in res["errors"])
    # 主体 kind 报告应仍是 mob（多数投票：2 mob > 2 skill 中 mob 排在前；其实这里 2:2，
    # max 行为视字典顺序而定。我们只断言两类对象都被识别。）
    assert res["kind"] in {"mob", "skill"}


def test_mixed_yaml_in_auto_mode() -> None:
    """auto 模式下也应正常通过。"""
    res = validate_yaml(MOB_WITH_METASKILLS, kind="auto")
    assert res["valid"] is True
    assert not any("缺少必填字段 Type" in e for e in res["errors"])


def test_real_metaskill_with_extra_fields() -> None:
    """metaskill 可以带 Cooldown、TargetConditions 等字段。"""
    text = """
ShitC4:
  Cooldown: 40
  TargetConditions:
  - gliding true
  Skills:
  - sound{sound=minecraft:block.note_block.bit;p=1.5} @self
  - delay 200
  - remove @self
"""
    res = validate_yaml(text, kind="skill")
    assert res["valid"] is True, f"unexpected errors: {res['errors']}"


def test_metaskill_skills_not_a_list_is_error() -> None:
    """skill 的 Skills 字段如果不是 list 就该报错。"""
    text = """
BadSkill:
  Skills: just-a-string
"""
    res = validate_yaml(text, kind="auto")
    assert res["valid"] is False
    assert any("Skills" in e and "列表" in e for e in res["errors"])


def test_unknown_object_only_warns() -> None:
    """既无 Type/Material/Id/Skills 的对象，只该出 warning，不该 error。"""
    text = """
Strange:
  Foo: bar
"""
    res = validate_yaml(text, kind="auto")
    assert res["valid"] is True
    assert any("类型未识别" in w for w in res["warnings"])
