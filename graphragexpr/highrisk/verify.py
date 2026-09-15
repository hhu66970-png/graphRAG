"""候选结论的程序核验（任务书 §6）。

核验的立场
----------
「引用不存在」「数字单位错配」「明确对象错配」「关键条件遗漏」属于**硬性失败**。
模型再次判断「支持」、向量高相似度或任何加权总分，都**不能**推翻硬性失败
（任务书 §6）。因此本模块只输出结构化判定，不接受任何「置信度」输入。

核验按 **单元整体** 进行：数值与它的用法／生制品／人群／程度是一个不可拆的整体，
所以先定位「支持该结论数值的那个单元」，再要求结论把该单元的条件全部带上。
这样「数值都出现在原文但绑定错了」和「保留了剂量却漏掉限制」都会被抓到。

纯函数：不查库、不调模型、不写数据。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..domain.high_risk import (
    ConditionalEvidenceUnit,
    PolarityVerdict,
    RestrictionSeverity,
    RiskCategory,
    SEVERITY_DISPLAY,
)
from .claims import CandidateClaim, ClaimKind
from .parsing import parse_dosage_units

# ---- 硬性失败码。任何一条出现即判定该候选结论不予采纳 ----------------------
FAIL_UNKNOWN_EVIDENCE = "unknown_evidence_reference"
FAIL_NO_CITATION = "missing_citation"
FAIL_SUBJECT_MISMATCH = "subject_mismatch"
FAIL_OBJECT_MISMATCH = "object_mismatch"
FAIL_VALUE_MISMATCH = "dosage_value_mismatch"
FAIL_UNIT_MISMATCH = "dosage_unit_mismatch"
FAIL_BINDING_MISMATCH = "dosage_binding_mismatch"
FAIL_MISSING_CONDITION = "missing_required_condition"
FAIL_FABRICATED_CONDITION = "fabricated_condition"
FAIL_SEVERITY_ALTERED = "severity_altered"
FAIL_POLARITY_ALTERED = "polarity_altered"
FAIL_UNSAFE_CLAIM = "unsafe_negative_claim"
FAIL_UNSUPPORTED_POPULATION = "unsupported_population"
FAIL_UNRESOLVED_CONFLICT = "unresolved_conflict"
FAIL_EVIDENCE_NOT_LOCATABLE = "evidence_not_locatable"

#: 出现这些表述意味着结论在宣称「安全／无禁忌」。
#: 只有在证据单元**明确否定禁忌关系**时才允许这么说——「没查到」不等于「可以」。
_SAFETY_PHRASES: tuple[str, ...] = (
    "无配伍禁忌", "没有配伍禁忌", "不存在配伍禁忌",
    "无禁忌", "没有禁忌", "不存在禁忌", "无使用禁忌",
    "可以安全", "可安全使用", "安全使用", "没有问题", "可以服用",
)

#: 「安全／可同用」类表述的**带间隔**写法。必须用正则而不是字面子串：
#: 最危险的一句恰恰是「麻黄可以与藜芦同用」——中间夹了对象名，字面子串
#: 「可以同用」匹配不到，等于把这句话直接放过了。
#:
#: 反向的假阳性同样必须避免：「不可与藜芦同用」「不能与藜芦同用」是**正确的
#: 禁忌陈述**，正是答案里该出现的那句话；因此用否定后视把它们排除掉，
#: 否则核验会把该说的话判成不安全表述。
_SAFETY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<![不未])可以(?:与[^，。；、]{1,20})?同用"),
    re.compile(r"(?<![不未没])可(?:与[^，。；、]{1,20})?同用"),
    re.compile(r"(?<![不未没])能(?:与[^，。；、]{1,20})?同用"),
    re.compile(r"(?<![不未])可以(?:与[^，。；、]{1,20})?合用"),
    re.compile(r"(?<![不未没])可(?:与[^，。；、]{1,20})?合用"),
    re.compile(r"(?<![不未])可以(?:与[^，。；、]{1,20})?配伍"),
    re.compile(r"(?<![不未没])可(?:与[^，。；、]{1,20})?配伍"),
)


def _safety_phrase_in(statement: str) -> str | None:
    """Return the safety claim found in `statement`, if any.

    先查字面短语，再查带间隔的正则；返回命中的原文片段，供说明里引用。
    """
    for phrase in _SAFETY_PHRASES:
        if phrase in statement:
            return phrase
    for pattern in _SAFETY_PATTERNS:
        match = pattern.search(statement)
        if match:
            return match.group(0)
    return None

#: 人群词。结论不得把教材的一般说明推成对特定人群的适用结论。
_POPULATION_WORDS: tuple[str, ...] = (
    "婴幼儿", "新生儿", "小儿", "儿童", "孕妇", "妊娠", "产妇", "哺乳期",
    "老人", "年老", "体弱", "久病", "肝肾功能不全", "肾功能不全", "肝功能不全",
)

#: 频次词。教材未写「每日／每次」时结论不得补入（任务书 §5.2）。
_FREQUENCY_WORDS: tuple[str, ...] = ("每日", "每天", "一日", "每次", "日服", "分次服")


@dataclass
class ClaimVerdict:
    """一条候选结论的核验结果。"""

    claim_id: str
    passed: bool
    hard_failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    supporting_unit_ids: list[str] = field(default_factory=list)
    matched_dosage_raw: str | None = None


@dataclass
class VerificationOutcome:
    """一次核验的全部结果。"""

    verdicts: list[ClaimVerdict] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)

    @property
    def accepted(self) -> list[ClaimVerdict]:
        """核验通过的结论列表。"""
        return [verdict for verdict in self.verdicts if verdict.passed]

    @property
    def rejected(self) -> list[ClaimVerdict]:
        """核验失败的结论列表。"""
        return [verdict for verdict in self.verdicts if not verdict.passed]

    def has_any_pass(self) -> bool:
        """是否至少有一条结论核验通过。"""
        return bool(self.accepted)


def _unit_numbers(unit: ConditionalEvidenceUnit) -> set[float]:
    """抽出证据单元中的剂量数字。"""
    if unit.dosage is None or not unit.dosage.has_numeric_value():
        return set()
    return {float(unit.dosage.value_min), float(unit.dosage.value_max)}


def _claimed_numbers(claim: CandidateClaim) -> list[float]:
    """All numbers the claim puts forward, from both structured fields and text."""
    numbers = [float(value) for value in claim.asserted_numbers]
    for binding in parse_dosage_units(claim.statement):
        if binding.value_min is not None:
            numbers.append(float(binding.value_min))
        if binding.value_max is not None:
            numbers.append(float(binding.value_max))
    return numbers


def _supporting_units(
    claim: CandidateClaim, units: list[ConditionalEvidenceUnit]
) -> list[ConditionalEvidenceUnit]:
    """Units that actually back the quantities this claim puts forward.

    剂量结论：只保留「数值集合覆盖结论全部数值」的单元——这就是
    「数值对但绑定错」的拦截点。
    禁忌／配伍结论：剂量数值不是它的主张内容，全部被引用的单元都是支持单元。
    按类别区分很重要：否则一句禁忌原文里顺带出现的剂量数字会把
    禁忌结论误判成「数值无据」。
    """
    if claim.kind is not ClaimKind.DOSAGE:
        return list(units)
    numbers = set(_claimed_numbers(claim))
    if not numbers:
        return list(units)
    return [unit for unit in units if numbers.issubset(_unit_numbers(unit))]


def _check_conditions(
    claim: CandidateClaim, supporting: list[ConditionalEvidenceUnit]
) -> tuple[list[str], list[str]]:
    """Compare asserted conditions against the conditions of the supporting units.

    Returns:
        (硬性失败, 说明)。核心规则：

        - 支持单元有的条件，结论漏掉 → `missing_required_condition`；
        - 支持单元没有的条件，结论却写上 → `fabricated_condition`。
    """
    failures: list[str] = []
    notes: list[str] = []
    if not supporting:
        return [FAIL_BINDING_MISMATCH], ["没有任何证据单元覆盖结论中的数值"]

    for attribute, asserted, label in (
        ("route", claim.asserted_route, "用法"),
        ("preparation", claim.asserted_preparation, "生制品"),
        ("population", claim.asserted_population, "人群"),
        ("frequency", claim.asserted_frequency, "频次"),
    ):
        unit_values = {
            getattr(unit, attribute) for unit in supporting if getattr(unit, attribute)
        }
        if unit_values and asserted not in unit_values:
            failures.append(FAIL_MISSING_CONDITION)
            notes.append(
                f"结论漏掉或错配了{label}：证据单元为 {'、'.join(sorted(unit_values))}，"
                f"结论为 {asserted or '未给出'}"
            )
        elif not unit_values and asserted:
            failures.append(FAIL_FABRICATED_CONDITION)
            notes.append(f"结论补充了教材未说明的{label}：{asserted}")

    severities = {
        unit.severity for unit in supporting if unit.severity is not RestrictionSeverity.UNSPECIFIED
    }
    if severities:
        if claim.asserted_severity not in severities:
            failures.append(FAIL_SEVERITY_ALTERED)
            expected = "、".join(SEVERITY_DISPLAY[item] for item in sorted(severities, key=str))
            actual = (
                SEVERITY_DISPLAY[claim.asserted_severity]
                if claim.asserted_severity
                else "未给出"
            )
            notes.append(f"限制程度被改变或省略：证据为「{expected}」，结论为「{actual}」")
    elif claim.asserted_severity is not None:
        failures.append(FAIL_FABRICATED_CONDITION)
        notes.append(
            f"结论补充了原文未标注的限制程度：{SEVERITY_DISPLAY[claim.asserted_severity]}"
        )

    return failures, notes


def _check_text_level_binding(
    claim: CandidateClaim, supporting: list[ConditionalEvidenceUnit]
) -> tuple[list[str], list[str]]:
    """Check the dosage bindings that appear *inside the statement text*.

    这一层专门拦截「数值都出现在原文，但把 A 的数值配给了 B 的用法」：
    对结论文本里解析出的每一组剂量，都要求存在一个支持单元与它**整体**吻合
    （数值 + 单位 + 用法 + 生制品），而不是只要求数字出现过。
    """
    if claim.kind is not ClaimKind.DOSAGE:
        return [], []
    failures: list[str] = []
    notes: list[str] = []
    statement_bindings = parse_dosage_units(claim.statement)
    dosage_units = [
        unit for unit in supporting if unit.risk_category is RiskCategory.DOSAGE
    ]
    for binding in statement_bindings:
        if not binding.has_numeric_value():
            continue
        matched = False
        for unit in dosage_units:
            if unit.dosage is None:
                continue
            if binding.value_min != unit.dosage.value_min:
                continue
            if binding.value_max != unit.dosage.value_max:
                continue
            if (
                binding.unit
                and unit.dosage.unit
                and binding.unit != unit.dosage.unit
            ):
                continue
            if binding.route and unit.route and binding.route != unit.route:
                continue
            if (
                binding.preparation
                and unit.preparation
                and binding.preparation != unit.preparation
            ):
                continue
            matched = True
            break
        if not matched:
            failures.append(FAIL_BINDING_MISMATCH)
            notes.append(
                f"结论文本中的剂量「{binding.raw}」无法与任何被引用证据单元整体吻合"
                "（数值、单位、用法、生制品必须同时对上）"
            )
    return failures, notes


def _check_statement_matches_assertions(
    claim: CandidateClaim, _supporting: list[ConditionalEvidenceUnit] | None = None
) -> tuple[list[str], list[str]]:
    """Require the claim's own text to agree with its structured fields.

    为什么需要这一层：模型可以在结构化字段里填一份、在结论文本里写另一份。
    「字段说 route=煎服、文本写成生品 3～9g」这种情况，字段本身与证据并不冲突，
    只比对字段就会漏过；而它恰恰说明模型在自行加工细节。任一方向不一致都驳回：

    - 文本出现条件、字段为空 → 结论实际上补充了教材未说明的条件；
    - 字段有值、文本里没有   → 结论漏掉了自己声明的关键条件。

    `asserted_severity` 不参与文本比对：结论文本里的标签本身就含「忌」字
    （「使用禁忌」），按文本反推程度会误判，程度改由 `_check_conditions`
    与证据单元的枚举值直接比对。
    """
    if claim.kind is not ClaimKind.DOSAGE:
        return [], []
    failures: list[str] = []
    notes: list[str] = []
    for binding in parse_dosage_units(claim.statement):
        if not binding.has_numeric_value():
            continue
        for label, in_text, asserted in (
            ("单位", binding.unit, claim.asserted_unit),
            ("生制品", binding.preparation, claim.asserted_preparation),
            ("用法", binding.route, claim.asserted_route),
            ("频次", binding.frequency, claim.asserted_frequency),
            ("人群", binding.population, claim.asserted_population),
        ):
            if in_text == asserted:
                continue
            if in_text and not asserted:
                failures.append(FAIL_FABRICATED_CONDITION)
                notes.append(
                    f"结论文本出现{label}「{in_text}」，但结构化字段未声明，"
                    "视为自行补充教材未说明的条件"
                )
            elif asserted and not in_text:
                failures.append(FAIL_MISSING_CONDITION)
                notes.append(
                    f"结论声明了{label}「{asserted}」，但结论文本中没有出现，"
                    "结论与自身的字段不一致"
                )
            else:
                failures.append(FAIL_BINDING_MISMATCH)
                notes.append(
                    f"结论文本的{label}「{in_text}」与结构化字段「{asserted}」不一致"
                )
    return failures, notes


def _check_unsafe_claims(
    claim: CandidateClaim, supporting: list[ConditionalEvidenceUnit]
) -> tuple[list[str], list[str]]:
    """Block safety conclusions that no unit actually supports."""
    failures: list[str] = []
    notes: list[str] = []
    statement = claim.statement or ""
    hit = _safety_phrase_in(statement)
    if hit:
        negated = [
            unit
            for unit in supporting
            if unit.polarity_verdict is PolarityVerdict.RELATION_DOES_NOT_HOLD
        ]
        if not negated:
            failures.append(FAIL_UNSAFE_CLAIM)
            notes.append(
                f"结论使用了「{hit}」这类安全性表述，但没有任何证据单元明确否定该禁忌；"
                "未检索到禁忌不等于可以同用"
            )
    return failures, notes


def _check_population_scope(
    claim: CandidateClaim, supporting: list[ConditionalEvidenceUnit]
) -> tuple[list[str], list[str]]:
    """Block population/frequency wording the textbook never stated."""
    failures: list[str] = []
    notes: list[str] = []
    statement = claim.statement or ""
    unit_populations = {unit.population for unit in supporting if unit.population}
    unit_frequencies = {unit.frequency for unit in supporting if unit.frequency}

    for word in _POPULATION_WORDS:
        if word in statement and word != claim.asserted_population:
            if not any(word in population for population in unit_populations):
                failures.append(FAIL_UNSUPPORTED_POPULATION)
                notes.append(
                    f"结论提到人群「{word}」，但所选证据未对该人群作出说明，"
                    "不得把一般说明推成特定人群的适用结论"
                )
    for word in _FREQUENCY_WORDS:
        if word in statement and word != claim.asserted_frequency:
            if not any(word in frequency for frequency in unit_frequencies):
                failures.append(FAIL_FABRICATED_CONDITION)
                notes.append(
                    f"结论提到频次「{word}」，但教材原文未说明，不得自动补入默认频次"
                )
    return failures, notes


def _check_unresolved(
    claim: CandidateClaim, supporting: list[ConditionalEvidenceUnit]
) -> tuple[list[str], list[str]]:
    """Refuse a single definitive conclusion while conflicts remain open."""
    failures: list[str] = []
    notes: list[str] = []
    blocking: list[str] = []
    for unit in supporting:
        if not unit.is_clearly_parsed:
            blocking.append(unit.undetermined_reason or unit.unit_id)
        elif unit.polarity_ambiguous:
            blocking.append(unit.polarity_basis)
    if blocking and not claim.is_empty_conclusion():
        failures.append(FAIL_UNRESOLVED_CONFLICT)
        notes.append(
            "存在未解决冲突或无法确定的证据，不能生成单一确定结论："
            + "；".join(dict.fromkeys(blocking))
        )
    return failures, notes


def verify_claims(
    claims: list[CandidateClaim],
    units: list[ConditionalEvidenceUnit],
    *,
    allowed_unit_ids: list[str] | None = None,
) -> VerificationOutcome:
    """Verify candidate claims against the final injected evidence units.

    Args:
        claims: 模型返回的候选结论。
        units: 本次**最终注入**的证据单元。
        allowed_unit_ids: 允许被引用的单元标识集合。默认等于 `units` 的标识。
            传入更小的集合可以表达「这些证据已被预算裁剪或核验剔除」。

    Returns:
        VerificationOutcome。`accepted` 中的结论才允许进入最终自然语言。

    Side effects: 无。
    """
    units_by_id = {unit.unit_id: unit for unit in units}
    allowed = set(allowed_unit_ids if allowed_unit_ids is not None else units_by_id)

    outcome = VerificationOutcome()
    for claim in claims:
        verdict = ClaimVerdict(claim_id=claim.claim_id, passed=True)

        if claim.is_empty_conclusion():
            # 「没有证据」的声明不携带事实主张，因此不做条件与数值比对；
            # 但仍**要求引用存在**——否则「未检索到」可以被包装成一条不引用
            # 任何证据的结论，从核验旁边绕过去。这条注释原先只对应一句说明，
            # 判定仍是 passed=True，注释与行为不一致；这里按注释执行。
            if not claim.cited_unit_ids:
                verdict.hard_failures.append(FAIL_NO_CITATION)
                verdict.notes.append(
                    "空结论声明未附任何证据引用；「未检索到」不能绕过核验"
                )
                verdict.passed = False
            outcome.verdicts.append(verdict)
            continue

        if not claim.cited_unit_ids:
            verdict.hard_failures.append(FAIL_NO_CITATION)
            verdict.notes.append("候选结论没有引用任何证据单元")
            verdict.passed = False
            outcome.verdicts.append(verdict)
            continue

        unknown = [
            unit_id
            for unit_id in claim.cited_unit_ids
            if unit_id not in allowed or unit_id not in units_by_id
        ]
        if unknown:
            verdict.hard_failures.append(FAIL_UNKNOWN_EVIDENCE)
            verdict.notes.append(
                "引用了不存在或不属于本次最终注入证据的标识：" + "、".join(unknown)
            )
            verdict.passed = False
            outcome.verdicts.append(verdict)
            continue

        cited = [units_by_id[unit_id] for unit_id in claim.cited_unit_ids]

        for unit in cited:
            if not unit.is_citable():
                verdict.hard_failures.append(FAIL_EVIDENCE_NOT_LOCATABLE)
                verdict.notes.append(
                    f"证据单元 {unit.unit_id} 的原文无法定位，不能作为引用依据"
                )
            if claim.subject_name and unit.subject_name and claim.subject_name != unit.subject_name:
                verdict.hard_failures.append(FAIL_SUBJECT_MISMATCH)
                verdict.notes.append(
                    f"结论主语「{claim.subject_name}」与证据主体「{unit.subject_name}」不一致"
                )
            if (
                claim.object_name
                and unit.object_name
                and claim.object_name != unit.object_name
            ):
                verdict.hard_failures.append(FAIL_OBJECT_MISMATCH)
                verdict.notes.append(
                    f"结论对象「{claim.object_name}」与证据对象「{unit.object_name}」不一致"
                )
            if (
                unit.polarity_verdict is PolarityVerdict.RELATION_DOES_NOT_HOLD
                and claim.asserted_polarity is PolarityVerdict.RELATION_HOLDS
            ):
                verdict.hard_failures.append(FAIL_POLARITY_ALTERED)
                verdict.notes.append(
                    f"证据单元 {unit.unit_id} 否定该禁忌关系，结论却按关系成立表述"
                )

        supporting = _supporting_units(claim, cited)
        verdict.supporting_unit_ids = [unit.unit_id for unit in supporting]

        if claim.kind is ClaimKind.DOSAGE:
            claimed = [float(value) for value in claim.asserted_numbers]
            if claimed and not supporting:
                verdict.hard_failures.append(FAIL_VALUE_MISMATCH)
                verdict.notes.append(
                    "结论给出的数值没有任何证据单元支持："
                    + "、".join(str(value) for value in claimed)
                )
            if claim.asserted_unit:
                unit_units = {
                    unit.dosage.unit
                    for unit in supporting
                    if unit.dosage and unit.dosage.unit
                }
                if unit_units and claim.asserted_unit not in unit_units:
                    verdict.hard_failures.append(FAIL_UNIT_MISMATCH)
                    verdict.notes.append(
                        f"结论单位「{claim.asserted_unit}」与证据单位不一致："
                        + "、".join(sorted(unit_units))
                    )

        for check in (
            _check_conditions,
            _check_statement_matches_assertions,
            _check_text_level_binding,
            _check_unsafe_claims,
            _check_population_scope,
            _check_unresolved,
        ):
            failures, notes = check(claim, supporting)
            verdict.hard_failures.extend(failures)
            verdict.notes.extend(notes)

        verdict.hard_failures = list(dict.fromkeys(verdict.hard_failures))
        verdict.notes = list(dict.fromkeys(verdict.notes))
        verdict.passed = not verdict.hard_failures
        if verdict.passed and supporting:
            first_dosage = next(
                (unit.dosage for unit in supporting if unit.dosage), None
            )
            verdict.matched_dosage_raw = first_dosage.raw if first_dosage else None
        outcome.verdicts.append(verdict)

    outcome.rejections = [
        {
            "claim_id": verdict.claim_id,
            "hard_failures": verdict.hard_failures,
            "notes": verdict.notes,
        }
        for verdict in outcome.rejected
    ]
    return outcome
