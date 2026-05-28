"""所有系统 / 角色 prompts 集中管理；不允许在其他模块硬编码 wiki 内容。

按 intent 拆分 generator prompt：
- mob   → MOB_GENERATOR_PROMPT     （MythicMobs 主 wiki）
- item  → ITEM_GENERATOR_PROMPT    （MythicMobs Items + MythicCrucible 物品扩展）
- skill → SKILL_GENERATOR_PROMPT   （metaskill 单独定义；不带 Type/Material）

planner 的输出 JSON 同时驱动 search_queries（决定 retriever 拉哪个 wiki 的内容）。
"""
from __future__ import annotations

# ============================================================
# 系统 prompt：贯穿整个会话
# ============================================================
SYSTEM_PROMPT = """你是 MythicMobs / MythicCrucible 配置生成助手，专注于帮助 Minecraft 服务器主创建 Mob、Item、Skill (metaskill) 的 YAML 配置。

请严格遵循以下原则：
1. 优先参考 RAG 检索到的官方 Wiki 文档片段，把它视作权威。来源会标注 source / wiki，注意区分 mythicmobs 与 crucible：
   - mythicmobs：mob、技能机制、目标器、条件、触发器（@target / ~onTimer 等）
   - crucible：物品的额外机制 / 触发器 / 家具 / 合成（~onUse / ~onLeftclick 等仅在物品上可用的触发器）
2. 不得编造不存在的 Mechanics / Targeters / Conditions / Triggers，遇到不确定的就用 wiki_search 查证。
3. 不得生成非法字段；保持 wiki 文档中规定的属性名大小写。
4. 必须用中文向用户解释关键配置（每个非平凡字段的作用）。
5. 必须给出至少 2 条使用建议或扩展方向。
6. 输出的 YAML 必须可以被 ruamel.yaml 解析；缩进为 2 个空格。
7. 当用户的需求模糊时，先做合理默认选择，不要反复追问；可以在「建议」里指出可以调整的地方。
8. **多轮对话连续性**：当请求中带有「上一次生成的 YAML」时，**必须把当前请求理解为对它的修改 / 增量调整**，
   不要从零重写一份不相关的内容。保留原顶层 key 名和大部分字段，只改用户明确要求改动的部分；
   若用户明确说「重新设计 / 换一个」才允许另起新的 key。
"""


# 用于在 planner / generator 输入里前置历史与上一次产物的小模板
CONTEXT_BLOCK = """\

==== 历史对话（最近若干轮）====
{history}

==== 上一次生成的 YAML（若有，作为本次修改的基线）====
```yaml
{last_yaml}
```

"""


def render_context_block(history: list[dict] | None, last_yaml: str) -> str:
    """把 memory snapshot 渲染成一段可注入的上下文文本；空则返回空串。"""
    history = history or []
    if not history and not last_yaml.strip():
        return ""
    if history:
        lines = []
        for m in history:
            role = m.get("role", "?")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            # 截断单条防爆 context
            if len(content) > 800:
                content = content[:800] + "...(truncated)"
            lines.append(f"[{role}] {content}")
        history_text = "\n".join(lines) if lines else "(empty)"
    else:
        history_text = "(empty)"
    return CONTEXT_BLOCK.format(history=history_text, last_yaml=last_yaml or "(none)")


# ============================================================
# Planner：意图分类 + 检索计划
# ============================================================
PLANNER_PROMPT = """你是 Planner。读完用户输入后，输出一个 JSON 对象（不要任何其他文字）：

{{
  "intent": "mob | item | skill | chat",
  "search_queries": ["query1", "query2", ...],
  "needs_rag": true | false,
  "name": "YAML 顶层 key，PascalCase 或下划线，按 intent 给个合理默认",
  "notes": "对生成器的提示（中文，1-2 句即可）",
  "is_followup": true | false
}}

intent 判定规则（按优先级匹配）：
- "item"  —— 用户想要创建一个**物品**（武器、消耗品、合成材料、家具、护符等）。关键词：物品、装备、武器、剑、消耗品、合成、Recipe、Crucible、家具、皮肤、CustomModel
- "skill" —— 用户只想要一个**独立的 metaskill**（不绑定到某个 mob / item，纯技能片段）。关键词：技能、技能链、metaskill、火球术、treasure 之类没有载体的咒语
- "mob"   —— 用户想要创建一个**怪物 / Boss / 召唤生物**（含其附属技能链）。关键词：怪物、Boss、生物、Mob、召唤、敌人
- "chat"  —— 闲聊或纯知识问答（"projectile 的 highAccuracyMode 是什么意思？"），不需要生成任何 YAML

is_followup 判定（**关键，决定后续是否走"修改"路径**）：
- 若**上下文中已有 YAML**且当前用户输入像是在「调整 / 修改 / 补充 / 改一改 / 把 X 换成 Y / 加一个 / 去掉 / 改成 / 只改…」那个 YAML，则 is_followup=true，intent 应**保持与上一份 YAML 的类型一致**，name 也应**复用上一份 YAML 的顶层 key**。
- 若是新需求（"再来一个不同的 boss"、"换个物品" 等明确的另起话题），is_followup=false。

search_queries 规则：
- intent=mob：覆盖「mob 的 Type、特殊技能、视觉（粒子效果）/音效、技能条件、触发器、AI等」7-10 个角度
- intent=item：覆盖「Material/Id、Skills 触发器（onUse/onLeftclick/onConsume）、Crucible 特性（Furniture/Recipe）」7-10 个角度
- intent=skill：覆盖技能用到的核心 mechanics + targeter + condition
- intent=chat：留空数组
- 每条 query 不超过 12 个英文单词，使用英文以匹配 wiki 内容
- **is_followup=true 时**，search_queries 应针对用户「这次提到的差异点」（例如要改的字段或新增的机制），不要重复检索旧 YAML 的全部内容。

name 默认值：
- mob → 形如 "DeepSeaBoss"
- item → 形如 "FrostBlade"
- skill → 形如 "FrostNova"
- chat → ""
- is_followup=true 时优先复用上一份 YAML 的顶层 key

{context_block}用户输入：
{user_input}
"""


# ============================================================
# Generator：按 intent 三套独立模板
# ============================================================
_GEN_HEADER = """根据以下信息生成合法的 MythicMobs / MythicCrucible YAML：

用户需求：
{user_input}

意图: {intent}
建议的对象名: {name}
是否为多轮修改 (is_followup): {is_followup}
Planner 备注: {notes}
{context_block}
参考的 Wiki 片段（来自 RAG 检索；前面方括号内是序号）：
---
{context}
---

如果你在生成过程中需要查阅更多信息，可以调用可用工具（最多 {max_tool_calls} 次）。
当你确认信息足够时，直接输出最终结果（不再调用工具）。

如果 is_followup=true 且上文给了「上一次生成的 YAML」：
- **基于该 YAML 做最小必要修改**，保留无关字段不变，复用顶层 key。
- 解释里只解释**改动的字段**，不要把整份再讲一遍。
- 建议里给出针对**改动后整体**的下一步优化方向。
"""


MOB_GENERATOR_PROMPT = (
    _GEN_HEADER
    + """
# 任务：生成一个 Mob。**请把 mob 主体 + 它引用的所有 metaskill 写在同一份 YAML 里**——这是 MythicMobs 的常见组织方式，校验器接受这种混写。

输出格式严格按下面三段：

## YAML
```yaml
{name}:
  Type: <vanilla 实体类型，如 ZOMBIE / GUARDIAN / BLAZE>
  Display: '<彩色名称，可用 & 颜色码或 <#hex>>'
  Health: <数字>
  Damage: <数字>
  Faction: <可选>
  Options:
    AlwaysShowName: <bool>
    PreventOtherDrops: <bool>
    MovementSpeed: <数字>
    # 更多选项请参考 wiki（wiki/MythicMobs.wiki/Mobs/Options.md）
  AIGoalSelectors:    # 可选，覆盖默认 AI
  - <selector>
  AITargetSelectors:
  - <selector>
  Skills:
  - <mechanic>{{...}} <targeter> ~<trigger> ?<condition>
  - skill{{s=<引用的 metaskill 名>}} @<targeter> ~<trigger> ?<condition>
  Drops:              # 可选
  - <item> 1 0.5

# 在下面接着写所有被 mob 引用的 metaskill（如果有）
# 除了Skills字段，其他都是可选字段
<引用的 metaskill 名>:
  Conditions:
  - condition1
  - condition2
  TargetConditions:
  - condition3
  - condition4
  TriggerConditions:
  - condition5
  - condition6
  FailedConditionsSkill: [the metaskill to executed if the conditions did not check]
  Cooldown: [seconds]
  OnCooldownSkill: [the metaskill to execute if this one is on cooldown]
  Skill: [an additional metaskill to execute asynchronously from this one]
  Skills:
  - <mechanic> @target
  - mechanic2
  - mechanic3
```

## 解释
- 重要字段逐条说明（中文，2-6 条）

## 建议
- 至少 2 条扩展或调参方向
"""
)


ITEM_GENERATOR_PROMPT = (
    _GEN_HEADER
    + """
# 任务：生成一个 MythicMobs Item（含 Crucible 增强）。物品的特殊触发器（~onUse / ~onLeftclick / ~onRightclick / ~onConsume）属于 Crucible 范畴；如果 RAG 命中里有 wiki=crucible 的片段，**优先**参考它们。

输出格式严格按下面三段：

## YAML
```yaml
{name}:
  Id: <vanilla material 名小写，如 diamond_sword / blaze_rod / cocoa_beans>
  Display: '<彩色名称>'
  Lore:                # 可选；推荐用来写描述/获取途径
  - '<行1>'
  - '<行2>'
  Model: <可选；CustomModelData 数字>
  Group: <可选>
  Attributes:          # 可选，属性加成
    MainHand:
      Damage: <数字>
  Enchantments:        # 可选
  - SHARPNESS:5
  Options:             # 可选
    Unbreakable: <bool>
  Recipes:             # Crucible：合成配方，可选
    SHAPED:
      Type: SHAPED
      Amount: 1
      Ingredients:
      - <a> | <b> | <c>
      - <d> | <e> | <f>
      - <g> | <h> | <i>
  Skills:              # Crucible：物品触发器
  - skill{{s=<触发的 metaskill>}} @<targeter> ~<trigger> ?<condition>
  - skill{{s=<另一个>}} @<targeter> ~<trigger> ?<condition>

# 接着写所有被引用的 metaskill（如果有）
# 除了Skills字段，其他都是可选字段
<引用的 metaskill 名>:
  Conditions:
  - condition1
  - condition2
  TargetConditions:
  - condition3
  - condition4
  TriggerConditions:
  - condition5
  - condition6
  FailedConditionsSkill: [the metaskill to executed if the conditions did not check]
  Cooldown: [seconds]
  OnCooldownSkill: [the metaskill to execute if this one is on cooldown]
  Skill: [an additional metaskill to execute asynchronously from this one]
  Skills:
  - <mechanic> @target
  - mechanic2
  - mechanic3
```

## 解释
- 重要字段逐条说明（中文）；触发器 ~onUse / ~onLeftclick 等要说明触发时机
- 必要时指出 Crucible 是必装插件

## 建议
- 至少 2 条扩展方向（材质包、合成、限制条件等）
"""
)


SKILL_GENERATOR_PROMPT = (
    _GEN_HEADER
    + """
# 任务：生成一个独立的 metaskill（不带 Type / Material / Id）。如果用户描述需要多个相互调用的 metaskill，把它们一起写在同一份 YAML 里。

输出格式严格按下面三段：

## YAML
```yaml
{name}:
  Cooldown: <可选；秒>
  TargetConditions:    # 可选；技能选定目标时的过滤
  - <condition>
  Conditions:          # 可选；释放者条件
  - <condition>
  TriggerConditions:   # 可选；技能触发时的额外条件
  - condition5
  - condition6
  FailedConditionsSkill: [the metaskill to executed if the conditions did not check]
  Cooldown: [seconds]
  OnCooldownSkill: [the metaskill to execute if this one is on cooldown]
  Skill: [an additional metaskill to execute asynchronously from this one]
  Skills:
  - <mechanic>{{...}} <targeter> ?<inline-condition>
  - delay <ticks>
  - skill{{s=<子 metaskill>}} @target

# 如有引用其他 metaskill，紧接着定义
# 除了Skills字段，其他都是可选字段
<子 metaskill 名>:
  Conditions:
  - condition1
  - condition2
  TargetConditions:
  - condition3
  - condition4
  TriggerConditions:
  - condition5
  - condition6
  FailedConditionsSkill: [the metaskill to executed if the conditions did not check]
  Cooldown: [seconds]
  OnCooldownSkill: [the metaskill to execute if this one is on cooldown]
  Skill: [an additional metaskill to execute asynchronously from this one]
  Skills:
  - <mechanic> @target
  - mechanic2
  - mechanic3
```

## 解释
- 重要字段逐条说明（中文）

## 建议
- 至少 2 条扩展或调参方向
"""
)


def get_generator_prompt(intent: str) -> str:
    """根据 intent 选择对应的 generator prompt 模板。"""
    if intent == "item":
        return ITEM_GENERATOR_PROMPT
    if intent == "skill":
        return SKILL_GENERATOR_PROMPT
    # mob 或未识别都走 mob 模板（最通用、最完整）
    return MOB_GENERATOR_PROMPT


# 兼容老调用：保留 GENERATOR_PROMPT 指向 mob 模板（旧测试 / 旧代码使用）
GENERATOR_PROMPT = MOB_GENERATOR_PROMPT


# ============================================================
# Validator fix loop
# ============================================================
VALIDATOR_FIX_PROMPT = """以下 YAML 校验失败：
错误：
{errors}

原始 YAML：
```yaml
{yaml_text}
```

请基于错误修复后重新输出完整 YAML，仍包裹在 ```yaml ... ``` 中。除了代码块，不要输出任何说明。
"""


__all__ = [
    "SYSTEM_PROMPT",
    "PLANNER_PROMPT",
    "GENERATOR_PROMPT",
    "MOB_GENERATOR_PROMPT",
    "ITEM_GENERATOR_PROMPT",
    "SKILL_GENERATOR_PROMPT",
    "get_generator_prompt",
    "render_context_block",
    "CONTEXT_BLOCK",
    "VALIDATOR_FIX_PROMPT",
]
