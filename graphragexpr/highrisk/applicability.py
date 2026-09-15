"""问题—证据适用性核验（第二层核验）。

第一层核验（`verify.py`）回答的是「生成的结论**是否符合所取证据**」。
本模块回答的是「这些证据**是否能回答用户所问的对象和条件**」。

两层缺一不可。只做第一层时，程序可以忠实复述「孕妇慎用」，却拿它回答
「哺乳期能否使用」，最后仍然显示「已核验」——复述忠实并不等于回答了问题。
实测：「附子哺乳期能用吗？」「附子小儿能用吗？」在没有本模块时都返回
`grounded`，证据却是孕妇慎用；「布洛芬孕妇能吃吗？」返回 `grounded` 并列出
徐长卿、千金子、厚朴的孕妇禁忌。

三条判定，全部**只依据「问题明说了什么」与「单元声明了什么」**：

1. `assess_applicability`：问题指定的人群／生制品／用法，是否有证据单元支持。
2. `is_enumeration_question`：当前问法是否**确实允许**按人群反向枚举读取。
3. `severity_conflicts`：同一对象同一条件下出现不一致的限制程度时如实并列，
   不替用户取舍。

**不重做实体消歧**：主语是否已被完整识别由服务层结合链接器结果判定后传入。
本模块不做相似度、不做模糊匹配、不猜同义实体；术语映射只有下面这一张
小而可审查的表，表外一律判为「不匹配」或「未知」，宁可不下结论。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..domain.high_risk import (
    SEVERITY_DISPLAY,
    ConditionalEvidenceUnit,
    RestrictionSeverity,
)
from .parsing import extract_population, extract_preparation, extract_route

#: 反向枚举问法。只有这类问法在**没有识别出对象**时才允许按人群反向读取。
#:
#: 「孕妇禁用的中药有哪些？」问的是一个集合，按人群反查是正确的读法；
#: 「布洛芬孕妇能吃吗？」问的是某个具体对象能否使用，对象没识别出来时必须
#: 弃权，绝不能把别的药材的结论端上来当答案。
_ENUMERATION_TERMS: tuple[str, ...] = (
    "有哪些", "有哪几种", "哪几种", "哪几味", "哪一味", "哪些", "什么药",
    "有啥药", "有什么药", "都有什么", "都有哪些", "列举", "举例", "名单",
)


def is_enumeration_question(question: str) -> bool:
    """问法是否在要求「列出满足某条件的一组对象」。

    只按问法本身判断，不看是否有主语：有主语时走正向读取，这个判定不参与；
    没有主语时它决定是否允许反向枚举读取。
    """
    normalized = "".join((question or "").split())
    return any(term in normalized for term in _ENUMERATION_TERMS)


#: 泛称词。它们在本题材里指的是**一整类**东西，不是某个具体对象。
#:
#: 必须显式列出，因为本体里确实可能把它们建成了实体——实测「中药」是
#: 一个 HERB（`HERB:2170a739bcb95637`）。链接器按文本匹配命中它之后，
#: 「孕妇禁用的中药有哪些？」会被当成「中药这味药」的正向问题，
#: 读空并弃权——真正需要反向枚举读取的问法反而答不出来。
_GENERIC_OBJECT_TERMS: frozenset[str] = frozenset({
    "中药", "中药材", "药材", "草药", "药物", "药品", "中成药",
    "方剂", "饮片", "中药饮片", "药",
})


def is_generic_object(name: str) -> bool:
    """这个名称是否只是泛称，不能算「用户指定的对象」。

    只做**全等**匹配，不做子串：「川乌」「制川乌」都不会因为是泛称被误伤。
    """
    return "".join((name or "").split()) in _GENERIC_OBJECT_TERMS


#: 人群的**明确同义表达**。只收临床与教材里可视为同一人群的说法；
#: 表外不猜测——「表里没有」等于「不知道是否同一人群」，按不匹配处理，
#: 而不是按相似度凑一个匹配出来。
#:
#: 刻意不收的：「产妇」与「孕妇」不是同一人群（产后 vs 妊娠期），
#: 「新生儿」与「儿童」是包含关系而非同义，均各自成组或不入表。
_POPULATION_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("孕妇", "妊娠", "妊娠期"),
    ("哺乳期", "哺乳", "乳母"),
    ("小儿", "儿童"),
    ("婴幼儿", "新生儿", "婴儿"),
    ("老人", "年老", "老年人", "高龄"),
)


#: 用法归属的**家族**。用家族而不是字符串包含判断，是因为「煎服」并不包含
#: 「内服」二字，但煎服就是内服的一种：若按子串判，问「附子内服多少克」
#: 会与「煎服」证据判成不匹配，把一个**本来正确**的答案降级。
#:
#: 只分内服／外用两个家族：教材里区分用法主要是为了分开这两个途径，
#: 家族内更细的差异（煎服 vs 研末服）由剂量单元自身的条件表达，
#: 不在这里做二次判断。
_ROUTE_FAMILIES: tuple[tuple[str, ...], ...] = (
    (
        "内服", "煎服", "水煎服", "研末服", "冲服", "吞服", "口服", "灌服",
        "含服", "泡服", "开水泡服", "入丸散", "烊化", "先煎", "后下", "另煎",
        "包煎", "调服", "送服",
    ),
    ("外用", "煎汤外洗", "敷患处", "调敷", "外洗", "外敷"),
)


def _route_family(text: str) -> int | None:
    """Return the route family index of `text`, or None when unknown."""
    for index, family in enumerate(_ROUTE_FAMILIES):
        if any(term in text for term in family):
            return index
    return None


def same_route_family(asked: str, declared: str) -> bool:
    """问题指定的用法与证据声明的用法是否属于同一途径。

    任一侧不在表内即判不匹配（例如证据写了一个表外的新用法，此时不能
    假装它满足问题指定的途径）。
    """
    asked_family = _route_family(asked)
    declared_family = _route_family(declared)
    return asked_family is not None and asked_family == declared_family


def _group_index(text: str) -> int | None:
    """Return the synonym-group index of `text`, or None when not in the table."""
    for index, group in enumerate(_POPULATION_SYNONYM_GROUPS):
        if any(term in text for term in group):
            return index
    return None


def same_population(asked: str | None, declared: str | None) -> bool:
    """问题指定的人群与单元声明的人群是否可视为同一人群。

    两侧都允许是短语：单元常写「气虚津亏者及孕妇」，问题常写「孕妇」。
    判定走上面那张同义表；任一侧不在表内即判不匹配（诚实的不确定，
    不是「大概相同」）。
    """
    if not asked or not declared:
        return False
    asked_group = _group_index(asked)
    declared_group = _group_index(declared)
    return asked_group is not None and asked_group == declared_group


@dataclass(frozen=True)
class AskedConditions:
    """问题**明说**的条件。没说的留空，不补默认值。"""

    population: str | None = None
    preparation: str | None = None
    route: str | None = None

    @classmethod
    def from_question(cls, question: str) -> "AskedConditions":
        """从用户问题解析适用性约束（剂型、人群等）。"""
        return cls(
            population=extract_population(question),
            preparation=extract_preparation(question),
            route=extract_route(question),
        )


class Applicability(str, Enum):
    """证据对当前子问题的适用性判定。"""

    APPLICABLE = "applicable"
    CONDITION_MISMATCH = "condition_mismatch"
    SUBJECT_UNCONFIRMED = "subject_unconfirmed"


@dataclass(frozen=True)
class ApplicabilityAssessment:
    """一次适用性核验的结果。`reason` 会进入用户可见文本。"""

    verdict: Applicability
    reason: str = ""

    @property
    def allows_grounded(self) -> bool:
        """当前约束是否允许给出「已核验」结论。"""
        return self.verdict is Applicability.APPLICABLE


def _matches_population(unit: ConditionalEvidenceUnit, asked: str) -> bool:
    """判断证据适用人群是否覆盖问题中的人群。"""
    return same_population(asked, unit.population)


def assess_applicability(
    question: str,
    units: list[ConditionalEvidenceUnit],
    *,
    subject_unconfirmed_reason: str | None = None,
) -> ApplicabilityAssessment:
    """这些证据能否回答该问题。

    Args:
        question: 该子问题的原文。
        units: 本子问题**通过第一层核验**的证据单元。传空列表表示没有可放行的
            结论，此时不需要适用性判定（调用方按既有逻辑判 source_only/abstain）。
        subject_unconfirmed_reason: 主语未确认的说明。非 None 时直接判
            `SUBJECT_UNCONFIRMED`，不再看条件——对象都不确定，条件无从谈起。

    Returns:
        `ApplicabilityAssessment`。只有 `APPLICABLE` 才允许标为 grounded。

    Side effects: 无。
    """
    if subject_unconfirmed_reason:
        return ApplicabilityAssessment(
            Applicability.SUBJECT_UNCONFIRMED, subject_unconfirmed_reason
        )
    if not units:
        return ApplicabilityAssessment(Applicability.APPLICABLE)

    asked = AskedConditions.from_question(question)

    # 人群：问题指定了人群时，**必须**有证据单元对该人群作出说明。
    # 这是「哺乳期问题拿孕妇证据作答」的直接拦截点。
    if asked.population:
        if not any(_matches_population(unit, asked.population) for unit in units):
            declared = sorted({unit.population for unit in units if unit.population})
            detail = "、".join(declared) if declared else "未声明任何人群"
            return ApplicabilityAssessment(
                Applicability.CONDITION_MISMATCH,
                f"所取证据未涉及问题指定的人群「{asked.population}」"
                f"（证据涉及：{detail}），不能用来回答该人群的用法",
            )

    # 生制品：生品与制品的剂量不可互相套用，问题指定时必须匹配。
    if asked.preparation:
        if not any(
            unit.preparation and asked.preparation in unit.preparation
            for unit in units
        ):
            return ApplicabilityAssessment(
                Applicability.CONDITION_MISMATCH,
                f"所取证据未涉及问题指定的炮制形态「{asked.preparation}」",
            )

    # 用法：只在**证据本身区分了用法**时才作为必需条件。若全部单元都未声明
    # 用法，不能因为「问题提了用法」就判不适用——那会凭空制造拒答。
    if asked.route:
        declared_routes = [unit.route for unit in units if unit.route]
        if declared_routes and not any(
            same_route_family(asked.route, route) for route in declared_routes
        ):
            return ApplicabilityAssessment(
                Applicability.CONDITION_MISMATCH,
                f"所取证据未涉及问题指定的用法「{asked.route}」"
                f"（证据涉及：{'、'.join(sorted(set(declared_routes)))}）",
            )

    return ApplicabilityAssessment(Applicability.APPLICABLE)


#: 并行展示时的程度排序：重的在前。只影响展示顺序，不改变任何单元的内容，
#: 更不把不同程度合并成一个。
_SEVERITY_ORDER: dict[RestrictionSeverity, int] = {
    RestrictionSeverity.PROHIBITED: 0,
    RestrictionSeverity.AVOID: 1,
    RestrictionSeverity.CAUTION: 2,
    RestrictionSeverity.UNSPECIFIED: 3,
}


def sort_by_severity(
    units: list[ConditionalEvidenceUnit],
) -> list[ConditionalEvidenceUnit]:
    """按限制程度从重到轻稳定排序，使「禁用」与「慎用」在列表里一眼可分。

    稳定排序：同程度的单元保持原有相对顺序，不引入新的不确定性。
    """
    return sorted(units, key=lambda unit: _SEVERITY_ORDER[unit.severity])


#: 中成药剂型字。问题里的药材名后面若**连着**这些字样，说明问题指的很可能是一个
#: 完整的成药对象，而不是这味药材本身。
#:
#: 为什么不能只靠本体里的实体名比对：`龙胆泻肝丸` 在本体里没有对应实体，
#: 「存在更长实体名」这条检查对它无从触发，于是「龙胆泻肝丸的用量是多少？」
#: 会拿「龙胆：煎服，3～6g」当作答案并标记为已核验——正是要拦的对象错配。
#: 这里只做**形态**判定：名称与剂型字之间不跨标点，最多隔 `_MAX_INFIX` 个汉字。
#:
#: 刻意不收「汤」「酒」「膏」「饮」：它们同时是煎煮、浸泡等**用法**用词，
#: 「附子煮汤」「附子泡酒」说的是用法而不是另一个对象，收进来会凭空拒答。
#: 也刻意不收「片」：「附子饮片」「附子切片」指的还是这味药材本身。
_DOSAGE_FORM_SUFFIXES: tuple[str, ...] = (
    "口服液", "软胶囊", "胶囊", "颗粒", "冲剂", "滴丸", "糖浆", "丸剂",
    "丸", "散",
)

#: 名称与剂型字之间允许隔开的汉字数上限（「龙胆泻肝丸」＝「龙胆」+「泻肝」+「丸」）。
_MAX_INFIX = 4


def longer_object_span(question: str, name: str) -> str | None:
    """问题里的名称后面是否还连着一个成药名，返回那个更长的片段。

    例如 `("龙胆泻肝丸的用量是多少", "龙胆")` → `"龙胆泻肝丸"`。

    这是「链接只命中了名称子串」的**形态**证据：命中的名称直接续着成药剂型字，
    说明用户说的是另一个更长的对象。本函数不做消歧、不猜那个对象是什么、
    也不去替换主语，只把「名称不完整」这一事实报出来。

    Returns:
        更长的片段；没有则 None。
    """
    if not question or not name:
        return None
    start = 0
    while True:
        index = question.find(name, start)
        if index < 0:
            return None
        start = index + 1
        tail_end = index + len(name)
        for extra in range(_MAX_INFIX + 1):
            suffix = next(
                (
                    item
                    for item in _DOSAGE_FORM_SUFFIXES
                    if question.startswith(item, tail_end + extra)
                ),
                None,
            )
            if suffix:
                return question[index : tail_end + extra] + suffix
            # 中间只能是汉字；遇到标点、空格、数字就说明名称在这里已经结束
            if tail_end + extra >= len(question) or not _is_han(question[tail_end + extra]):
                break
    return None


def _is_han(char: str) -> bool:
    """判断字符是否为汉字。"""
    return "一" <= char <= "鿿"


def _object_key(unit: ConditionalEvidenceUnit) -> str:
    """同一「对象」的比较键。

    对象名若是人群类的说法（教材里常写成「孕妇慎用」「妊娠禁用」这样的短语），
    一律归到人群维度上——**即使它有 object_id 也不用 id**。实测川乌的两处来源
    分别抽成了 `CONTRAINDICATION:…（孕妇慎用）` 与 `INDICATION:…（妊娠）`，
    两个 id 不同、实体类型也不同，按 id 分组时它们永远不会被放在一起比，
    「慎用」与「禁用」的分歧就不会被标出来——正是要拦的「错误放行」。
    """
    name = unit.object_name or ""
    group = _group_index(name)
    if group is not None:
        return f"population#{group}"
    return unit.object_id or name


def severity_conflicts(units: list[ConditionalEvidenceUnit]) -> dict[str, str]:
    """同一对象、同一人群下出现不一致的限制程度时，返回冲突说明。

    任务书要求「条件冲突的单元并列保留，不取平均、不求并集」；这里再加一条：
    **也不替用户按更严的一档表述**。两处来源程度不同而未经人工裁定时，
    把它们并列摆出来并标明冲突，而不是自行挑一档当作已经核验的结论。

    Returns:
        `{unit_id: 冲突说明}`，只包含涉事单元。空字典表示无冲突。
    """
    # 分组键必须**带上主语**：反向枚举问法一次返回多味药，千金子（禁用）与
    # 徐长卿（慎用）是两个不同对象各自的结论，不是互相冲突。少了主语这一维，
    # 一次正常的枚举会被误判成「同一对象下程度不一致」。
    grouped: dict[
        tuple[str, str, int | None], dict[RestrictionSeverity, list[str]]
    ] = {}
    for unit in units:
        if unit.severity is RestrictionSeverity.UNSPECIFIED:
            continue
        key = (
            unit.subject_id or unit.subject_name or "",
            _object_key(unit),
            _group_index(unit.population or ""),
        )
        grouped.setdefault(key, {}).setdefault(unit.severity, []).append(unit.unit_id)

    hit: dict[str, str] = {}
    for severities in grouped.values():
        if len(severities) < 2:
            continue
        labels = "、".join(
            SEVERITY_DISPLAY[severity]
            for severity in sorted(severities, key=_SEVERITY_ORDER.__getitem__)
        )
        note = (
            f"同一对象下存在不一致的限制程度（{labels}），两处来源并列保留，"
            "未做取舍，不得按其中一档作为已经核验的结论；系统暂不作单一结论"
        )
        for unit_ids in severities.values():
            for unit_id in unit_ids:
                hit[unit_id] = note
    return hit
