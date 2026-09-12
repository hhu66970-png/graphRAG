"""问题条件相关性选择与「直接结论」（第三部分）。

两个职责，都是**展示组织**层面的判断，不改动任何检索、抽取或核验算法：

1. `classify_units`：把「通过两层核验的证据」再按**问题明说的条件**分档。
   第一层核验只说明结论忠实于证据，第二层（`applicability.py`）只要求
   「至少存在一条与问题条件匹配的证据」。仅此还不够：实测「附子孕妇能用吗？」
   里，只要有一条孕妇证据成立，同批被接受的「阴虚阳亢者忌用」也会一起进入
   主要回答，并按限制程度排序**排在孕妇「慎用」之前**——用户先看到的是一句
   与所问人群无关的「忌用」。本模块把这类单元分出去。

   - DIRECT(A)：证据明确涉及问题指定的人群／生制品／用法（同义表内确认）。
   - UNCONDITIONAL(B)：证据没有限定其他条件，对所有适用对象成立。
   - MISMATCH(C)：证据明确指向别的条件（阴虚阳亢者、儿童、哺乳期…）。

2. `decide`：按**已通过适用性核验的证据**与 `RestrictionSeverity` 生成
   「用户可见的第一句」。措辞由模板给出，禁止模型改写——把「慎用」写成
   「禁用」是这一层最危险的失败模式。

设计约束：
- 相关性只依据**问题明说了什么**与**单元声明了什么**，不做相似度、不猜同义、
  不做实体消歧（沿用 `applicability.py` 的小型可审查术语表，表外一律不算匹配）。
- 相关性**不参与**冲突检测的取舍：不同人群之间程度不同不算本题冲突，
  同一人群下的程度冲突仍然照常降级（见 `service.py` 的调用顺序）。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..domain.high_risk import (
    SEVERITY_DISPLAY,
    ConditionalEvidenceUnit,
    RestrictionSeverity,
    RiskCategory,
)
from .applicability import (
    AskedConditions,
    _group_index,
    is_enumeration_question,
    is_generic_object,
    same_population,
    same_route_family,
)
from .parsing import _POPULATION_CUES

#: 人群限定语的**形态**标记。教材常把人群写成对象本身：
#: 「阴虚阳亢者忌用」「孕妇慎用」。这些说法不在同义表内，但显然是人群限定，
#: 不能被当成「无条件通用」——否则换个问法就能把别的病证的禁忌当作答案。
#:
#: 只收这些标记，不做任何语义推断：命中的是「这句话限定了某类人」这一**形态**，
#: 至于限定的是哪类人，由同义表决定是否与问题一致；表外的按「其他人群」处理。
_POPULATION_MARKERS: tuple[str, ...] = ("者", "人群", "患者", "病人")


class Relevance(str, Enum):
    """一条证据相对当前问题条件的位置。"""

    DIRECT = "direct"              # A：与问题指定的条件一致
    UNCONDITIONAL = "unconditional"  # B：未限定其他条件
    MISMATCH = "mismatch"          # C：明确指向别的条件


#: 展示优先级：精确匹配在前，通用其次，其他最后。**高于**限制程度排序。
_RELEVANCE_ORDER: dict[Relevance, int] = {
    Relevance.DIRECT: 0,
    Relevance.UNCONDITIONAL: 1,
    Relevance.MISMATCH: 2,
}

RELATIVE_LABEL: dict[Relevance, str] = {
    Relevance.DIRECT: "与本题指定条件一致",
    Relevance.UNCONDITIONAL: "未限定其他条件",
    Relevance.MISMATCH: "与本题指定条件不同",
}


def ordered(relevance: Relevance) -> int:
    return _RELEVANCE_ORDER[relevance]


def _declared_population(unit: ConditionalEvidenceUnit) -> str | None:
    """单元**声明**的人群。字段为空时回落到对象名与其他限制语。"""
    if unit.population:
        return unit.population
    for text in (unit.object_name or "", *unit.other_restrictions):
        if text and _group_index(text) is not None:
            return text
    return None


def _population_limited_elsewhere(unit: ConditionalEvidenceUnit) -> str | None:
    """单元是否把适用对象限定在**别的某类人**身上（表外的写法）。

    返回那段原文，供说明使用；没有则 None。只做形态判定，不解释语义。
    """
    for text in (unit.object_name or "", *unit.other_restrictions):
        if text and any(marker in text for marker in _POPULATION_MARKERS):
            return text
    return None


def classify(question: str, unit: ConditionalEvidenceUnit) -> Relevance:
    """这条证据相对该问题指定的条件属于哪一档。

    Args:
        question: 子问题原文。
        unit: 已通过核验的候选证据单元。

    Returns:
        `Relevance`。问题没有明说条件时一律 `DIRECT`（本模块不做无中生有的筛选）。

    Side effects: 无。
    """
    asked = AskedConditions.from_question(question)
    asked_people = asked_populations(question)
    if not asked_people and not asked.preparation and not asked.route:
        # 问题没有明说任何条件：没有可筛选的维度，一律 DIRECT，
        # 本模块不做无中生有的筛选。
        return Relevance.DIRECT
    #: 是否确认了「这条证据正是对本题指定条件说的」。A 与 B 的区别全在这个标记上：
    #: 命中了问题指定的人群是 A（精确匹配），没有限定人群才是 B（无条件通用）。
    confirmed = False

    # ---- 人群：本模块要解决的主要情形 -----------------------------------
    if asked_people:
        declared = _declared_population(unit)
        if declared is not None:
            # 问题里可能同时问了几类人，命中**任意一类**即算匹配。
            if any(same_population(person, declared) for person in asked_people):
                confirmed = True
            else:
                return Relevance.MISMATCH
        elif (
            unit.risk_category is RiskCategory.CONTRAINDICATION
            and (unit.object_name or "").strip()
            and not is_generic_object(unit.object_name)
        ):
            # 使用禁忌的宾语就是「对谁禁忌」。宾语非空、又不属于问题指定的人群
            # （阴虚阳亢者、产妇、表外写法……），说明这条讲的是别人。
            # 泛称宾语除外：反向枚举问法会读到对象是「中药」的行，那不是人群。
            return Relevance.MISMATCH
        elif _population_limited_elsewhere(unit) is not None:
            # 限定了别的某类人，且写在对象名之外的字段里。
            return Relevance.MISMATCH

    # ---- 生制品：生品与制品的数值不可互相套用 ---------------------------
    if asked.preparation:
        if unit.preparation:
            if asked.preparation not in unit.preparation:
                return Relevance.MISMATCH
            confirmed = True
        elif unit.dosage is not None and unit.dosage.preparation:
            if asked.preparation not in unit.dosage.preparation:
                return Relevance.MISMATCH
            confirmed = True

    # ---- 用法：只在单元确实区分了用法时才作为筛选依据 -------------------
    if asked.route and unit.route:
        if not same_route_family(asked.route, unit.route):
            return Relevance.MISMATCH
        confirmed = True

    return Relevance.DIRECT if confirmed else Relevance.UNCONDITIONAL


def classify_units(
    question: str, units: list[ConditionalEvidenceUnit]
) -> dict[str, Relevance]:
    """整批单元的分档结果：`{unit_id: Relevance}`。"""
    return {unit.unit_id: classify(question, unit) for unit in units}


# --------------------------------------------------------------- 直接结论


class Decision(str, Enum):
    """用户可见第一句的类别。取值与限制程度一一对应，不混用。"""

    PROHIBITED = "prohibited"  # 禁用
    AVOID = "avoid"            # 忌用／不宜
    CAUTION = "caution"        # 慎用
    CONFLICT = "conflict"      # 同一条件下程度不一致
    UNKNOWN = "unknown"        # 没有与问题对象和条件匹配的证据
    NONE = "not_applicable"    # 该问法不由本模块生成首句（如纯用量问题）


#: 单一程度 → 首句模板。`{population}` / `{subject}` 由证据与问题填入。
#:
#: 措辞刻意区分三档，**不得互相替代**：
#: 把「慎用」说成「禁用」或绝对的「不能使用」都会造成过度限制，
#: 反过来把「禁用」说成「慎用」则是放行。
_DECISION_TEMPLATES: dict[RestrictionSeverity, str] = {
    RestrictionSeverity.PROHIBITED: "不能使用。教材明确记载：{population}禁用{subject}。",
    RestrictionSeverity.AVOID: "不宜使用。教材记载：{population}忌用{subject}。",
    RestrictionSeverity.CAUTION: "不能直接判断为可以使用。教材记载：{population}应慎用{subject}。",
    RestrictionSeverity.UNSPECIFIED: (
        "暂时不能给出确定结论：教材未标注禁用／忌用／慎用的程度，"
        "不得自行选择其中一档。"
    ),
}

#: 有证据、但证据落在**别的条件**上，或干脆没有适用证据时的首句。
#: 禁止把「没有查到禁忌」写成「可以使用」。
_UNKNOWN_TEMPLATE = "根据当前教材证据，无法判断{population}能否使用{subject}。"

#: 同一条件下程度不一致时的首句。**不挑选更严的一档**。
_CONFLICT_TEMPLATE = (
    "暂时不能给出单一结论：当前教材证据在{labels}之间存在差异。"
)


@dataclass(frozen=True)
class DecisionResult:
    """首句判定结果。`text` 为空表示本问法不由本模块生成首句。"""

    decision: Decision
    text: str = ""
    severities: tuple[RestrictionSeverity, ...] = ()


def _severity_of(unit: ConditionalEvidenceUnit) -> RestrictionSeverity:
    return unit.severity


def asked_populations(question: str) -> list[str]:
    """问题里**明说**的全部人群，按同义表去重，保持出现顺序。

    为什么不能用 `extract_population`：它只返回第一个命中。
    「麻黄孕妇和儿童能用吗？」里两个人群都问了，若只取「孕妇」，
    儿童那一条证据就会被判成「其他人群」而被踢出主回答——那不是错配，
    是问题确实问了两类人。
    """
    normalized = "".join((question or "").split())
    seen: dict[int, str] = {}
    for cue in _POPULATION_CUES:
        if cue not in normalized:
            continue
        group = _group_index(cue)
        if group is None:
            continue
        seen.setdefault(group, cue)
    return list(seen.values())


def _display_subject(
    units: list[ConditionalEvidenceUnit], pool: list[str]
) -> str:
    """首句里点名的药。**只在名称唯一时点名**，否则用「该药」。

    点名取自证据单元的主语字段（不是模型文本）；单元里没有时回落到调用方给的
    池子（同一答案里其他单元的主语）。两者都不唯一时不猜——写「该药」比
    写错一个药名安全。
    """
    for source in (units, pool):
        names = {
            name
            for name in (
                unit.subject_name if isinstance(unit, ConditionalEvidenceUnit) else unit
                for unit in source
            )
            if name and name not in _GENERIC_SUBJECTS
        }
        if len(names) == 1:
            return next(iter(names))
    return "该药"


#: 泛称主语。它们不是「用户问的那味药」，首句里点名会把问题讲歪。
_GENERIC_SUBJECTS: frozenset[str] = frozenset({"中药", "中药材", "药材", "药"})


def _wider_severities(
    applicable: list[ConditionalEvidenceUnit],
    read_units: list[ConditionalEvidenceUnit],
    asked: list[str],
) -> set[RestrictionSeverity]:
    """同一主语、同一人群下**读到的全部**限制程度（含未进入本次上下文的）。

    首句只用注入集合时，被 token 预算裁掉的那一处会让回答**看起来**是一条
    没有分歧的结论——等于由裁剪替用户选了一档。实测：川乌孕妇，第 179 页的
    「慎用」被裁掉，留「禁用」，首句成了「不能使用」，与同一答案里
    「未成立的部分：…存在不一致的限制程度」自相矛盾。本函数只补齐程度集合，
    不改任何单元的内容，也不做取舍。
    """
    subjects = {unit.subject_id or unit.subject_name for unit in applicable}
    severities = {
        unit.severity
        for unit in applicable
        if unit.severity is not RestrictionSeverity.UNSPECIFIED
    }
    for unit in read_units:
        if unit.severity is RestrictionSeverity.UNSPECIFIED:
            continue
        if (unit.subject_id or unit.subject_name) not in subjects:
            continue
        declared = _declared_population(unit) or ""
        if not any(same_population(person, declared) for person in asked):
            continue
        severities.add(unit.severity)
    return severities


def decide(
    question: str,
    units: list[ConditionalEvidenceUnit],
    *,
    relevance: dict[str, Relevance] | None = None,
    subject_pool: list[str] | None = None,
    read_units: list[ConditionalEvidenceUnit] | None = None,
) -> DecisionResult:
    """按已通过适用性核验的证据生成首句。

    只使用 **DIRECT（A）** 档：不同人群之间的程度差异不是本题的结论，
    更不能因为别的病证写「忌用」就把本题的首句拔高成「不宜使用」。

    Args:
        question: 子问题原文（用于取问题明说的人群）。
        units: 通过核验的证据单元。
        relevance: `classify_units` 的结果。None 表示不做相关性筛选（全部视作 A）。
        subject_pool: 备用主语名（同一答案里其他单元的主语），仅在 `units` 无主语时使用。
        read_units: 本题读到的**全部**可引用单元（含因 token 预算未进入上下文的）。
            给出时，同一主语、同一人群下只要读到两种限制程度，首句即按冲突
            表述——被裁掉的那一处同样算分歧，不能因为没进上下文就当作不存在。

    Returns:
        `DecisionResult`。`Decision.NONE` 表示该问法不由本模块生成首句，
        由结论行直接承担（见下）。

    Side effects: 无。

    NONE 的四种情形，都是「一句话概括会失真」的情形，宁可不说：
    1. 问题没有指定人群（「用量是多少」没有「能否使用」这一层）；
    2. 问题同时指定了多个人群：各人群程度可能不同，合成一句就会张冠李戴；
    3. 反向枚举问法（「孕妇禁用的中药有哪些？」）：问的是一个集合，
       点名其中一味药会把部分清单说成结论；
    4. 候选证据的主语不唯一：同上，点名哪个都不对。
    """
    asked = asked_populations(question)
    if not asked:
        # 「用量是多少」这类问法没有「能否使用」的首句，直接由结论行承担。
        return DecisionResult(Decision.NONE)
    if len(asked) > 1 or is_enumeration_question(question):
        return DecisionResult(Decision.NONE)

    applicable = [
        unit
        for unit in units
        if (relevance is None or relevance.get(unit.unit_id) is Relevance.DIRECT)
        and unit.risk_category
        in (RiskCategory.CONTRAINDICATION, RiskCategory.INCOMPATIBILITY)
    ]
    subject = _display_subject(applicable, list(subject_pool or []))
    if subject == "该药" and len(applicable) > 1:
        # 多条证据、主语又点不出来：无法用一句话概括。
        return DecisionResult(Decision.NONE)

    severities = {
        _severity_of(unit)
        for unit in applicable
        if _severity_of(unit) is not RestrictionSeverity.UNSPECIFIED
    }
    if severities and read_units is not None:
        severities = _wider_severities(applicable, read_units, asked)
    if not severities:
        return DecisionResult(
            Decision.UNKNOWN,
            _UNKNOWN_TEMPLATE.format(population=asked[0], subject=subject),
        )
    if len(severities) > 1:
        # 同一人群下程度不一致：并列说明，不挑一档。
        labels = "、".join(
            f"「{SEVERITY_DISPLAY[item]}」"
            for item in sorted(severities, key=_severity_order)
        )
        return DecisionResult(
            Decision.CONFLICT,
            _CONFLICT_TEMPLATE.format(labels=labels),
            tuple(sorted(severities, key=_severity_order)),
        )

    severity = next(iter(severities))
    return DecisionResult(
        Decision(
            {
                RestrictionSeverity.PROHIBITED: Decision.PROHIBITED,
                RestrictionSeverity.AVOID: Decision.AVOID,
                RestrictionSeverity.CAUTION: Decision.CAUTION,
                RestrictionSeverity.UNSPECIFIED: Decision.UNKNOWN,
            }[severity]
        ),
        _DECISION_TEMPLATES[severity].format(
            population=asked[0], subject=subject
        ),
        (severity,),
    )


#: 冲突说明里程度的列出顺序：重 → 轻，与 `severity_conflicts` 的说明一致。
_SEVERITY_RANK: dict[RestrictionSeverity, int] = {
    RestrictionSeverity.PROHIBITED: 0,
    RestrictionSeverity.AVOID: 1,
    RestrictionSeverity.CAUTION: 2,
    RestrictionSeverity.UNSPECIFIED: 3,
}


def _severity_order(severity: RestrictionSeverity) -> int:
    return _SEVERITY_RANK[severity]


def sort_units(
    units: list[ConditionalEvidenceUnit],
    relevance: dict[str, Relevance],
) -> list[ConditionalEvidenceUnit]:
    """先按相关档位，再按限制程度从重到轻稳定排序。

    相关性**优先于**限制程度：不这样的话，一条「阴虚阳亢者忌用」会永远
    排在孕妇「慎用」前面，用户读到的第一张卡就是与本题人群无关的一条。
    """
    from .applicability import _SEVERITY_ORDER

    return sorted(
        units,
        key=lambda unit: (
            ordered(relevance.get(unit.unit_id, Relevance.DIRECT)),
            _SEVERITY_ORDER[unit.severity],
        ),
    )
