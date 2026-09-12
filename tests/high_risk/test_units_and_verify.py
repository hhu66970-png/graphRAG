"""条件单元组装与候选结论核验的单元级测试（任务书 §5 / §6）。

验收测试只看响应形态；这里直接对单元字段与核验判定断言，覆盖三类容易被「看起来对」
掩盖的问题：

- 组装侧：一数值一单元、单元标识稳定（结论会引用它）、存量解析冲突不静默取舍、
  纯原文来源不得冒充通过断言；
- 核验侧：引用不存在、漏掉／凭空补上条件、程度被改写、无据的安全性表述，
  以及未解决冲突不得生成单一确定结论——这些都是硬性失败，模型自评与相似度不能推翻。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphragexpr.domain.high_risk import (
    ConditionalEvidenceUnit,
    PolarityVerdict,
    RestrictionSeverity,
    RiskCategory,
    SourceKind,
)
from graphragexpr.highrisk.claims import CandidateClaim, ClaimKind
from graphragexpr.highrisk.units import (
    build_units_from_assertions,
    build_units_from_chunks,
)
from graphragexpr.highrisk.verify import (
    FAIL_FABRICATED_CONDITION,
    FAIL_MISSING_CONDITION,
    FAIL_NO_CITATION,
    FAIL_SEVERITY_ALTERED,
    FAIL_UNKNOWN_EVIDENCE,
    FAIL_UNRESOLVED_CONFLICT,
    FAIL_UNSAFE_CLAIM,
    verify_claims,
)

from .conftest import assertion_row, dosage_row


def dosage_units(text: str, assertion_id: str = "a-dose") -> list[ConditionalEvidenceUnit]:
    return build_units_from_assertions([dosage_row(assertion_id, text)])


def restriction_units(
    predicate: str,
    quote: str,
    *,
    object_id: str,
    object_name: str,
    assertion_id: str = "a-restriction",
) -> list[ConditionalEvidenceUnit]:
    return build_units_from_assertions(
        [
            assertion_row(
                assertion_id=assertion_id,
                predicate=predicate,
                polarity="NEGATIVE",
                object_id=object_id,
                object_name=object_name,
                evidence_quote=quote,
                chunk_texts=[quote],
            )
        ]
    )


def grounded_dosage_claim(claim_id: str, unit_id: str, **overrides) -> CandidateClaim:
    """一条与「煎服，3～9g」证据单元逐项吻合的剂量结论（对照组用）。"""
    fields: dict = {
        "claim_id": claim_id,
        "kind": ClaimKind.DOSAGE,
        "statement": "麻黄：煎服，3～9g。",
        "subject_name": "麻黄",
        "asserted_route": "煎服",
        "asserted_numbers": [3.0, 9.0],
        "asserted_unit": "g",
        "cited_unit_ids": [unit_id],
    }
    fields.update(overrides)
    return CandidateClaim(**fields)


def rejections_for(outcome, claim_id: str) -> list[dict]:
    return [item for item in outcome.rejections if item["claim_id"] == claim_id]


# ---------------------------------------------------------------- 组装：一数值一单元
def test_two_doses_become_two_units_with_their_own_conditions():
    """同段两处剂量 → 两个单元，各自带自己的炮制／用法条件，不得互换。"""
    # 同一片段内按生制品分档
    prep_units = build_units_from_assertions([dosage_row("a1", "生品 1.5～3g，制品 3～6g。")])
    assert [unit.dosage.raw for unit in prep_units] == ["1.5～3g", "3～6g"]
    assert [unit.preparation for unit in prep_units] == ["生品", "制品"]
    assert [unit.route for unit in prep_units] == [None, None]

    # 分号隔开的两处用法：强分隔符保证用法与数值留在各自的单元里
    route_units = build_units_from_assertions([dosage_row("a2", "煎服 3～9g；研末服 1～2g。")])
    assert [unit.dosage.raw for unit in route_units] == ["3～9g", "1～2g"]
    assert [unit.route for unit in route_units] == ["煎服", "研末服"]
    assert len({unit.unit_id for unit in prep_units}) == 2


def test_unparsable_dosage_becomes_an_unclear_unit():
    """「适量外用」解析不出数值 → 保留为不明确单元，不凭存量字符串作答。"""
    units = build_units_from_assertions([dosage_row("a3", "适量外用")])

    assert len(units) == 1
    unit = units[0]
    assert unit.risk_category is RiskCategory.DOSAGE
    assert unit.dosage is None
    assert unit.is_clearly_parsed is False
    assert unit.undetermined_reason
    assert "适量外用" in unit.undetermined_reason


def test_unit_ids_are_stable_across_builds():
    """单元标识由来源与内容派生：同一批行重复组装得到同一批标识。"""
    rows = [dosage_row("a1", "生品 1.5～3g，制品 3～6g。")]

    first = [unit.unit_id for unit in build_units_from_assertions(rows)]
    second = [unit.unit_id for unit in build_units_from_assertions(rows)]

    assert first == second
    assert len(set(first)) == 2
    # 内容相同但断言标识不同的行不得撞号：结论要跨请求核对引用
    other_rows = [dosage_row("a9", "生品 1.5～3g，制品 3～6g。")]
    other = [unit.unit_id for unit in build_units_from_assertions(other_rows)]
    assert set(other).isdisjoint(first)


def test_stored_parse_conflict_is_recorded_not_silently_resolved():
    """存量解析值与按原文重解析不一致时并列记录，不静默采用其一。"""
    stored = {"raw": "3～9g", "value_min": 9.0, "value_max": 3.0, "unit": "g"}
    row = assertion_row(
        assertion_id="a4",
        predicate="HAS_DOSAGE",
        object_value=stored,
        evidence_quote="麻黄 3～9g。",
        chunk_texts=["麻黄 3～9g。"],
    )
    unit = build_units_from_assertions([row])[0]

    # 采用按原文重解析的结果，但冲突必须被记下来交给核验与人工
    assert (unit.dosage.value_min, unit.dosage.value_max) == (3.0, 9.0)
    assert any("存量解析" in conflict for conflict in unit.conflicts), unit.conflicts


def test_predicate_outside_the_three_high_risk_ones_is_ignored():
    """范围外谓词被忽略，不静默扩大本任务的处理范围。"""
    rows = [
        assertion_row(assertion_id="out", predicate="TREATS", evidence_quote="麻黄发汗解表。"),
        assertion_row(
            assertion_id="in",
            predicate="CONTRAINDICATED_IN",
            polarity="NEGATIVE",
            object_id="P:孕妇",
            object_name="孕妇",
            evidence_quote="孕妇慎用。",
            chunk_texts=["孕妇慎用。"],
        ),
    ]

    assert [unit.assertion_id for unit in build_units_from_assertions(rows)] == ["in"]


# ---------------------------------------------------------------- 组装：原文来源
def test_text_chunk_unit_never_carries_an_assertion_id():
    """纯原文证据只记文本块与位置标识，不得被升级为通过断言。"""
    units = build_units_from_chunks(
        [{"chunk_id": "c-9", "text": "孕妇慎用。", "page_start": 12}]
    )

    assert len(units) == 1
    unit = units[0]
    assert unit.source_kind is SourceKind.TEXT_CHUNK
    assert unit.assertion_id is None
    assert unit.validation_status is None
    assert unit.chunk_ids == ["c-9"]
    assert unit.page_start == 12
    # 未经断言校验，不能单独支撑确定性结论
    assert unit.is_clearly_parsed is False
    assert unit.undetermined_reason


def test_text_chunk_unit_with_assertion_id_is_rejected():
    """契约层直接拒绝「原文来源 + 事实标识」这一组合（任务书 §4）。"""
    with pytest.raises(ValidationError, match="TEXT_CHUNK"):
        ConditionalEvidenceUnit(
            unit_id="unit:text",
            risk_category=RiskCategory.CONTRAINDICATION,
            predicate="CONTRAINDICATED_IN",
            source_kind=SourceKind.TEXT_CHUNK,
            assertion_id="a-1",
            evidence_quote="孕妇慎用。",
        )


# ---------------------------------------------------------------- 核验：引用
def test_unknown_evidence_reference_is_a_hard_failure():
    """引用了不存在、或不属于本次最终注入证据的标识 → 硬性失败。"""
    units = dosage_units("煎服，3～9g。")
    real_id = units[0].unit_id

    fabricated = grounded_dosage_claim("c-fabricated", "unit:does-not-exist")
    trimmed = grounded_dosage_claim("c-trimmed", real_id)

    # 标识确实存在于 units，但被允许集合裁掉（预算裁剪／已被剔除）后同样不可引用
    outcome = verify_claims([fabricated, trimmed], units, allowed_unit_ids=[])

    for claim_id in ("c-fabricated", "c-trimmed"):
        rejections = rejections_for(outcome, claim_id)
        assert rejections, outcome.rejections
        assert FAIL_UNKNOWN_EVIDENCE in rejections[0]["hard_failures"]

    # 对照：同一标识在默认允许集（= units 的标识）下通过
    baseline = verify_claims([trimmed], units)
    assert [verdict.claim_id for verdict in baseline.accepted] == ["c-trimmed"]


def test_claim_without_citation_is_rejected():
    """没有任何引用的结论不予采纳。"""
    units = dosage_units("煎服，3～9g。")
    claim = grounded_dosage_claim("c1", "unit:not-cited", cited_unit_ids=[])

    outcome = verify_claims([claim], units)

    assert outcome.accepted == []
    assert [verdict.claim_id for verdict in outcome.rejected] == ["c1"]
    assert FAIL_NO_CITATION in outcome.rejected[0].hard_failures


# ---------------------------------------------------------------- 核验：条件
def test_route_mismatch_is_a_hard_failure():
    """数值与证据一致，但用法被说成另一种 → 硬性失败。"""
    units = dosage_units("煎服，3～9g。")
    unit = units[0]
    assert unit.route == "煎服"

    wrong_route = grounded_dosage_claim(
        "c1",
        unit.unit_id,
        statement="麻黄：用量3～9g。",
        asserted_route="研末服",
    )
    outcome = verify_claims([wrong_route], units)

    rejections = rejections_for(outcome, "c1")
    assert rejections, outcome.rejections
    assert FAIL_MISSING_CONDITION in rejections[0]["hard_failures"]

    # 对照：用法与证据一致时通过
    ok = verify_claims([grounded_dosage_claim("c2", unit.unit_id)], units)
    assert [verdict.claim_id for verdict in ok.accepted] == ["c2"]


def test_dropped_condition_is_a_hard_failure():
    """保留了剂量却丢掉生制品条件 → 硬性失败（条件与数值不可拆）。"""
    units = dosage_units("生品 1.5～3g，制品 3～6g。")
    sheng = next(unit for unit in units if unit.dosage.raw == "1.5～3g")

    lossy = grounded_dosage_claim(
        "c1",
        sheng.unit_id,
        statement="麻黄：用量1.5～3g。",
        asserted_route=None,
        asserted_numbers=[1.5, 3.0],
    )
    outcome = verify_claims([lossy], units)

    rejections = rejections_for(outcome, "c1")
    assert rejections, outcome.rejections
    assert FAIL_MISSING_CONDITION in rejections[0]["hard_failures"]


def test_invented_condition_is_a_hard_failure():
    """教材未说明的条件被写进结论 → 硬性失败。"""
    units = dosage_units("3～9g")
    unit = units[0]
    assert unit.route is None and unit.preparation is None

    fabricated = grounded_dosage_claim("c1", unit.unit_id)
    outcome = verify_claims([fabricated], units)

    rejections = rejections_for(outcome, "c1")
    assert rejections, outcome.rejections
    assert FAIL_FABRICATED_CONDITION in rejections[0]["hard_failures"]


def test_severity_cannot_be_altered():
    """证据说慎用，结论说禁用 → 硬性失败；程度与证据一致时通过。"""
    units = restriction_units(
        "CONTRAINDICATED_IN", "孕妇慎用。", object_id="P:孕妇", object_name="孕妇"
    )
    unit = units[0]
    assert unit.severity is RestrictionSeverity.CAUTION

    escalated = CandidateClaim(
        claim_id="c1",
        kind=ClaimKind.CONTRAINDICATION,
        statement="孕妇禁用麻黄。",
        subject_name="麻黄",
        object_name="孕妇",
        asserted_population="孕妇",
        asserted_severity=RestrictionSeverity.PROHIBITED,
        cited_unit_ids=[unit.unit_id],
    )
    outcome = verify_claims([escalated], units)

    rejections = rejections_for(outcome, "c1")
    assert rejections, outcome.rejections
    assert FAIL_SEVERITY_ALTERED in rejections[0]["hard_failures"]

    kept = CandidateClaim(
        claim_id="c2",
        kind=ClaimKind.CONTRAINDICATION,
        statement="孕妇慎用麻黄。",
        subject_name="麻黄",
        object_name="孕妇",
        asserted_population="孕妇",
        asserted_severity=RestrictionSeverity.CAUTION,
        cited_unit_ids=[unit.unit_id],
    )
    kept_outcome = verify_claims([kept], units)
    assert [verdict.claim_id for verdict in kept_outcome.accepted] == ["c2"]


# ---------------------------------------------------------------- 核验：安全性表述
def test_safety_claim_without_negated_evidence_is_rejected():
    """「可以同用」这类表述不得在没有任何单元否定该禁忌时出现。"""
    units = restriction_units(
        "INCOMPATIBLE_WITH", "不宜与藜芦同用。", object_id="H:藜芦", object_name="藜芦"
    )
    unit = units[0]
    assert unit.polarity_verdict is PolarityVerdict.RELATION_HOLDS

    unsafe = CandidateClaim(
        claim_id="c1",
        kind=ClaimKind.INCOMPATIBILITY,
        statement="麻黄与藜芦可以同用。",
        subject_name="麻黄",
        object_name="藜芦",
        asserted_severity=RestrictionSeverity.AVOID,
        cited_unit_ids=[unit.unit_id],
    )
    outcome = verify_claims([unsafe], units)

    rejections = rejections_for(outcome, "c1")
    assert rejections, outcome.rejections
    assert FAIL_UNSAFE_CLAIM in rejections[0]["hard_failures"]

    # 对照：证据明确否定该禁忌关系（「无配伍禁忌」）时，同一类表述不再被判为无据
    negated_units = restriction_units(
        "INCOMPATIBLE_WITH", "无配伍禁忌。", object_id="H:藜芦", object_name="藜芦"
    )
    control = CandidateClaim(
        claim_id="c2",
        kind=ClaimKind.INCOMPATIBILITY,
        statement="麻黄与藜芦无配伍禁忌。",
        subject_name="麻黄",
        object_name="藜芦",
        asserted_severity=negated_units[0].severity,
        cited_unit_ids=[negated_units[0].unit_id],
    )
    control_failures = {
        code
        for item in verify_claims([control], negated_units).rejections
        for code in item["hard_failures"]
    }
    assert FAIL_UNSAFE_CLAIM not in control_failures


# ---------------------------------------------------------------- 核验：未解决冲突
def test_unclear_unit_cannot_support_a_definitive_claim():
    """未明确解析的单元（如倒置范围）只能降级为原文展示，不能支撑确定结论。"""
    units = dosage_units("9～3g")
    unit = units[0]
    assert unit.is_clearly_parsed is False

    claim = grounded_dosage_claim(
        "c1",
        unit.unit_id,
        statement="麻黄：用量9～3g。",
        asserted_route=None,
        asserted_numbers=[9.0, 3.0],
    )
    outcome = verify_claims([claim], units)

    rejections = rejections_for(outcome, "c1")
    assert rejections, outcome.rejections
    assert FAIL_UNRESOLVED_CONFLICT in rejections[0]["hard_failures"]


# ---------------------------------------------------------------- 核验：结果一致性
def test_accepted_rejected_and_rejections_stay_consistent():
    units = dosage_units("煎服，3～9g。")
    good = grounded_dosage_claim("c-good", units[0].unit_id)
    bad = grounded_dosage_claim("c-bad", "unit:not-cited", cited_unit_ids=[])

    outcome = verify_claims([good, bad], units)

    assert [verdict.claim_id for verdict in outcome.accepted] == ["c-good"]
    assert [verdict.claim_id for verdict in outcome.rejected] == ["c-bad"]
    assert [item["claim_id"] for item in outcome.rejections] == [
        verdict.claim_id for verdict in outcome.rejected
    ]
    assert len(outcome.accepted) + len(outcome.rejected) == len(outcome.verdicts) == 2
    assert outcome.has_any_pass() is True
    assert all(verdict.passed for verdict in outcome.accepted)
    assert all(not verdict.passed for verdict in outcome.rejected)
    # 被驳回的结论必须带硬性失败码与说明，二者一致
    assert all(item["hard_failures"] for item in outcome.rejections)
    assert all(item["notes"] for item in outcome.rejections)
