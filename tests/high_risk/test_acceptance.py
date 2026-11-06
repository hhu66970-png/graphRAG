"""高风险链路验收测试：从请求到响应。

每个用例都走完整的 `HighRiskService.answer`：
读取 → 条件单元 → 候选结论 → 程序核验 → 状态 → 渲染。
断言的是**响应形态**（status / abstained / claims / answer_text / rejections），
不是内部函数返回值，因此接口层接错线也会被发现。

覆盖的验收场景（编号与测试名一一对应）：

 1  单剂量正常成立
 2  同段两种用法 → 两个条件单元，不跨段拼配
 3  生制品分档 → 剂量随制品不同，不得互换
 4  倒置范围 → 不自动交换，降级为仅来源
 5  「未检索到禁忌」不得写成「可以同用」
 6  禁用／忌用／慎用三档不被抹平
 7  配伍禁忌的否定作用域：不宜与藜芦同用 → 关系成立
 8  使用禁忌的否定作用域：孕妇慎用 → 关系成立（不得判为无法确定）
 9  引用不存在的证据标识 → 硬性失败
10  数值都出现在原文但绑定错 → 硬性失败
11  关键条件遗漏 → 硬性失败
12  把一般剂量推成儿童剂量 → 硬性失败
13  复合问题部分成立 → partial_grounded，已支持结论不得被隐藏
14  数值与单位错配 → 硬性失败
15  完全无证据 → abstain，且答案不得出现安全性表述
16  模型自评「支持」不能推翻硬性失败
"""

from __future__ import annotations

import pytest

from graphragexpr.domain.high_risk import AnswerStatus
from graphragexpr.highrisk.claims import CandidateClaim, ClaimKind
from graphragexpr.highrisk.service import HighRiskService
from graphragexpr.highrisk.verify import (
    FAIL_BINDING_MISMATCH,
    FAIL_FABRICATED_CONDITION,
    FAIL_MISSING_CONDITION,
    FAIL_UNIT_MISMATCH,
    FAIL_UNKNOWN_EVIDENCE,
    FAIL_UNSUPPORTED_POPULATION,
)

from .conftest import FakeHighRiskStore, ScriptedLLM, assertion_row, batch, dosage_row

MAHUANG = "H:麻黄"


def service_for(rows, *, llm=None, budget: int = 3000) -> HighRiskService:
    """构造带假存储的高风险服务。"""
    return HighRiskService(
        FakeHighRiskStore(rows),
        run_id="run-test",
        llm=llm,
        unit_token_budget=budget,
    )


def linked_herb(name: str = "麻黄", canonical_id: str = MAHUANG) -> list[dict]:
    """测试用：已链接的中药实体。"""
    return [{"canonical_id": canonical_id, "entity_type": "HERB", "preferred_name": name}]


# ---------------------------------------------------------------- 场景 1
def test_01_plain_dosage_is_grounded():
    rows = [dosage_row("a1", "煎服，3～9g。", quote="麻黄煎服，3～9g。")]
    answer = service_for(rows).answer("麻黄用量是多少？", linked_entities=linked_herb())

    assert answer.status is AnswerStatus.GROUNDED
    assert answer.abstained is False
    assert len(answer.claims) == 1
    assert "3～9g" in answer.claims[0].statement
    assert "煎服" in answer.claims[0].statement
    assert answer.evidence_units[0].severity.value == "UNSPECIFIED"
    assert answer.evidence_units[0].page_label == "第 82 页"


# ---------------------------------------------------------------- 场景 2
def test_02_two_routes_in_one_clause_stay_separate():
    text = "煎服，3～9g；挽救虚脱可用 15～30g。"
    rows = [dosage_row("a1", text, quote=text)]
    answer = service_for(rows).answer("麻黄用量是多少？", linked_entities=linked_herb())

    statements = [claim.statement for claim in answer.claims]
    assert len(statements) == 2, statements
    # 每一段都必须带着自己的前提，不得把两段的数值与条件拼在一起
    assert any("3～9g" in line and "煎服" in line for line in statements)
    assert any("15～30g" in line and "挽救虚脱" in line for line in statements)
    assert not any("3～9g" in line and "15～30g" in line for line in statements)
    # 15～30g 脱离「挽救虚脱可用」就是超量表述，必须原样带出前提
    assert answer.status is AnswerStatus.GROUNDED


# ---------------------------------------------------------------- 场景 3
def test_03_preparation_specific_dosages_are_not_interchangeable():
    text = "生品 1.5～3g，制品 3～6g。"
    rows = [dosage_row("a1", text, quote=text)]
    answer = service_for(rows).answer("麻黄用量是多少？", linked_entities=linked_herb())

    statements = [claim.statement for claim in answer.claims]
    assert len(statements) == 2, statements
    sheng = next(line for line in statements if "1.5～3g" in line)
    zhi = next(line for line in statements if "3～6g" in line)
    assert "生品" in sheng and "制品" not in sheng
    assert "制品" in zhi and "生品" not in zhi


# ---------------------------------------------------------------- 场景 4
def test_04_inverted_range_is_marked_not_swapped():
    rows = [dosage_row("a1", "9～3g", quote="麻黄 9～3g。")]
    answer = service_for(rows).answer("麻黄用量是多少？", linked_entities=linked_herb())

    assert answer.status is AnswerStatus.SOURCE_ONLY
    assert answer.claims == []
    unit = answer.evidence_units[0]
    assert unit.dosage_raw == "9～3g"           # 未交换
    assert unit.is_clearly_parsed is False
    assert any("上界小于下界" in note for note in unit.conflicts)
    assert "9～3g" in answer.answer_text
    assert "9～3g" not in "".join(claim.statement for claim in answer.claims)


# ---------------------------------------------------------------- 场景 6
def test_06_severity_degrees_are_not_collapsed():
    rows = [
        assertion_row(
            assertion_id="a1", predicate="CONTRAINDICATED_IN", polarity="NEGATIVE",
            object_id="P:孕妇", object_name="孕妇",
            evidence_quote="孕妇慎用。", chunk_texts=["孕妇慎用。"],
        ),
        assertion_row(
            assertion_id="a2", predicate="CONTRAINDICATED_IN", polarity="NEGATIVE",
            object_id="P:儿童", object_name="儿童",
            evidence_quote="儿童禁用。", chunk_texts=["儿童禁用。"],
        ),
    ]
    answer = service_for(rows).answer("麻黄孕妇和儿童能用吗？", linked_entities=linked_herb())

    severities = {unit.severity_label for unit in answer.evidence_units}
    assert severities == {"慎用", "禁用"}, severities
    text = "\n".join([answer.answer_text] + [c.statement for c in answer.claims])
    assert "慎用" in text and "禁用" in text
    # 慎用单元不得被升级成禁用
    shen = next(u for u in answer.evidence_units if u.severity_label == "慎用")
    assert shen.severity.value == "CAUTION"


# ---------------------------------------------------------------- 场景 7
def test_07_incompatibility_negation_scope_is_preserved():
    rows = [
        assertion_row(
            assertion_id="a1", predicate="INCOMPATIBLE_WITH", polarity="NEGATIVE",
            object_id="H:藜芦", object_name="藜芦",
            evidence_quote="不宜与藜芦同用。", chunk_texts=["不宜与藜芦同用。"],
        )
    ]
    answer = service_for(rows).answer("麻黄不宜与什么药同用？", linked_entities=linked_herb())

    unit = answer.evidence_units[0]
    assert unit.polarity_verdict.value == "RELATION_HOLDS"
    assert unit.severity.value == "AVOID"
    assert answer.status is AnswerStatus.GROUNDED
    assert "藜芦" in answer.answer_text
    # 绝不能读成「可以和藜芦同用」
    assert "可以同用" not in answer.answer_text
    assert "无禁忌" not in answer.answer_text


# ---------------------------------------------------------------- 场景 8
def test_08_use_restriction_negation_scope_is_preserved():
    rows = [
        assertion_row(
            assertion_id="a1", predicate="CONTRAINDICATED_IN", polarity="NEGATIVE",
            object_id="P:孕妇", object_name="孕妇",
            evidence_quote="孕妇慎用。", chunk_texts=["孕妇慎用。"],
        )
    ]
    answer = service_for(rows).answer("麻黄孕妇能用吗？", linked_entities=linked_herb())

    unit = answer.evidence_units[0]
    assert unit.polarity_verdict.value == "RELATION_HOLDS", unit.polarity_basis
    assert unit.polarity_ambiguous is False
    assert answer.status is AnswerStatus.GROUNDED


# ---------------------------------------------------------------- 场景 5 / 15
def test_05_no_evidence_never_becomes_safe():
    answer = service_for([]).answer("麻黄孕妇能用吗？", linked_entities=linked_herb())

    assert answer.status is AnswerStatus.ABSTAIN
    assert answer.abstained is True
    # 只检查**肯定式**的安全性表述。像「没有证据不等于该用法是安全的」这种
    # 明确否认句必须允许出现——那正是应当说的话。
    for phrase in (
        "可以同用", "可同用", "无禁忌", "没有禁忌", "不存在禁忌",
        "可以服用", "没有问题", "安全使用", "可安全使用", "可以配伍",
    ):
        assert phrase not in answer.answer_text, phrase
    assert "不等于" in answer.answer_text          # 明确否认「没查到＝安全」


def test_15_abstain_on_unresolvable_subject():
    answer = service_for([]).answer("这个药怎么用？", linked_entities=[])

    assert answer.status is AnswerStatus.ABSTAIN
    assert answer.evidence_units == []
    assert "没有识别到药材或限定对象" in answer.answer_text
    assert answer.sub_questions[0].status is AnswerStatus.ABSTAIN


# ---------------------------------------------------------------- 场景 9 / 16
def test_09_fabricated_evidence_reference_is_a_hard_failure():
    text = "煎服，3～9g。"
    rows = [dosage_row("a1", text, quote=text)]
    fabricated = CandidateClaim(
        claim_id="c1", kind=ClaimKind.DOSAGE, statement="麻黄煎服 3～9g。",
        subject_name="麻黄", asserted_route="煎服",
        asserted_numbers=[3.0, 9.0], asserted_unit="g",
        cited_unit_ids=["unit:does-not-exist"],
    )
    # 模型在结论之外还自报「支持」——不得改变判定
    llm = ScriptedLLM([batch(fabricated)])
    answer = service_for(rows, llm=llm).answer(
        "麻黄用量是多少？", linked_entities=linked_herb()
    )

    rejections = [item for item in answer.rejections if item["claim_id"] == "c1"]
    assert rejections, answer.rejections
    assert FAIL_UNKNOWN_EVIDENCE in rejections[0]["hard_failures"]
    # 结构化单元的直接陈述仍然可用，说明驳回的是模型结论而不是整条链路
    assert answer.status is AnswerStatus.GROUNDED


def test_16_model_self_assessment_cannot_overturn_hard_failure():
    text = "煎服，3～9g。"
    rows = [dosage_row("a1", text, quote=text)]
    unit_id = service_for(rows).answer(
        "麻黄用量是多少？", linked_entities=linked_herb()
    ).evidence_units[0].unit_id

    # 数值全部出现在原文，但把「生品」的条件安到了常规剂量上
    misfiled = CandidateClaim(
        claim_id="c-ok-number", kind=ClaimKind.DOSAGE,
        statement="麻黄生品 3～9g。", subject_name="麻黄",
        asserted_preparation="生品", asserted_route="煎服",
        asserted_numbers=[3.0, 9.0], asserted_unit="g",
        cited_unit_ids=[unit_id],
    )
    # 模型声称「证据支持」，并给出与证据不符的条件
    llm = ScriptedLLM([batch(misfiled), batch(misfiled)])
    answer = service_for(rows, llm=llm).answer(
        "麻黄用量是多少？", linked_entities=linked_herb()
    )

    rejections = [item for item in answer.rejections if item["claim_id"] == "c-ok-number"]
    assert rejections, answer.rejections
    # 结构化字段与证据一致（都是煎服），但结论文本里写成「生品」并漏掉了煎服
    assert FAIL_FABRICATED_CONDITION in rejections[0]["hard_failures"]
    assert FAIL_MISSING_CONDITION in rejections[0]["hard_failures"]
    # 被驳回的模型结论不得出现在最终文字里
    assert "生品" not in answer.answer_text
    assert len(llm.calls) == 2, "驳回后应带上原因重试，且重试有次数上限"


# ---------------------------------------------------------------- 场景 10
def test_10_numbers_present_but_bound_wrong():
    text = "生品 1.5～3g，制品 3～6g。"
    rows = [dosage_row("a1", text, quote=text)]
    units = service_for(rows).answer(
        "麻黄用量是多少？", linked_entities=linked_herb()
    ).evidence_units
    zhi = next(unit for unit in units if unit.dosage_raw == "3～6g")

    # 3 和 6 都确实出现在原文里，但被配到了「生品」
    crossed = CandidateClaim(
        claim_id="c-crossed", kind=ClaimKind.DOSAGE,
        statement="麻黄：用量3～6g（炮制：生品）。", subject_name="麻黄",
        asserted_preparation="生品",
        asserted_numbers=[3.0, 6.0], asserted_unit="g",
        cited_unit_ids=[zhi.unit_id],
    )
    llm = ScriptedLLM([batch(crossed)])
    answer = service_for(rows, llm=llm).answer(
        "麻黄用量是多少？", linked_entities=linked_herb()
    )

    rejections = [item for item in answer.rejections if item["claim_id"] == "c-crossed"]
    assert rejections, answer.rejections
    assert FAIL_BINDING_MISMATCH in rejections[0]["hard_failures"] or (
        FAIL_MISSING_CONDITION in rejections[0]["hard_failures"]
    )
    # 被接受的结论里，任何一条都不得把 3～6g 与「生品」配在一起
    for claim in answer.claims:
        assert not ("3～6g" in claim.statement and "生品" in claim.statement), (
            claim.statement
        )


# ---------------------------------------------------------------- 场景 11
def test_11_missing_required_condition_is_rejected():
    text = "生品 1.5～3g，制品 3～6g。"
    rows = [dosage_row("a1", text, quote=text)]
    units = service_for(rows).answer(
        "麻黄用量是多少？", linked_entities=linked_herb()
    ).evidence_units
    sheng = next(unit for unit in units if unit.dosage_raw == "1.5～3g")

    # 保留了剂量却丢掉了生制品条件
    lossy = CandidateClaim(
        claim_id="c-lossy", kind=ClaimKind.DOSAGE,
        statement="麻黄：用量1.5～3g。", subject_name="麻黄",
        asserted_numbers=[1.5, 3.0], asserted_unit="g",
        cited_unit_ids=[sheng.unit_id],
    )
    llm = ScriptedLLM([batch(lossy)])
    answer = service_for(rows, llm=llm).answer(
        "麻黄用量是多少？", linked_entities=linked_herb()
    )

    rejections = [item for item in answer.rejections if item["claim_id"] == "c-lossy"]
    assert rejections, answer.rejections
    assert FAIL_MISSING_CONDITION in rejections[0]["hard_failures"]


# ---------------------------------------------------------------- 场景 12
def test_12_general_dosage_not_pushed_to_children():
    text = "煎服，3～9g。"
    rows = [dosage_row("a1", text, quote=text)]
    unit_id = service_for(rows).answer(
        "麻黄用量是多少？", linked_entities=linked_herb()
    ).evidence_units[0].unit_id

    escalated = CandidateClaim(
        claim_id="c-children", kind=ClaimKind.DOSAGE,
        statement="麻黄：儿童用量3～9g。", subject_name="麻黄",
        asserted_route="煎服", asserted_numbers=[3.0, 9.0], asserted_unit="g",
        cited_unit_ids=[unit_id],
    )
    llm = ScriptedLLM([batch(escalated)])
    answer = service_for(rows, llm=llm).answer(
        "麻黄用量是多少？", linked_entities=linked_herb()
    )

    rejections = [item for item in answer.rejections if item["claim_id"] == "c-children"]
    assert rejections, answer.rejections
    assert FAIL_UNSUPPORTED_POPULATION in rejections[0]["hard_failures"]
    assert "儿童" not in answer.answer_text


# ---------------------------------------------------------------- 场景 13
def test_13_compound_question_partially_grounded():
    rows = [dosage_row("a1", "煎服，3～9g。", quote="麻黄煎服，3～9g。")]
    answer = service_for(rows).answer(
        "麻黄用量是多少？孕妇能用吗？", linked_entities=linked_herb()
    )

    assert answer.status is AnswerStatus.PARTIAL_GROUNDED, [
        (item.question, item.status.value) for item in answer.sub_questions
    ]
    assert answer.abstained is True
    # 顶层 abstained=true 不得导致已成立的结论被隐藏
    assert answer.claims, answer.rejections
    assert "3～9g" in answer.answer_text
    statuses = {item.question: item.status for item in answer.sub_questions}
    assert statuses["麻黄用量是多少"] is AnswerStatus.GROUNDED
    assert statuses["孕妇能用吗"] is AnswerStatus.ABSTAIN


# ---------------------------------------------------------------- 场景 14
def test_14_unit_mismatch_is_rejected():
    text = "煎服，3～9g。"
    rows = [dosage_row("a1", text, quote=text)]
    unit_id = service_for(rows).answer(
        "麻黄用量是多少？", linked_entities=linked_herb()
    ).evidence_units[0].unit_id

    wrong_unit = CandidateClaim(
        claim_id="c-unit", kind=ClaimKind.DOSAGE,
        statement="麻黄：用量3～9mg。", subject_name="麻黄",
        asserted_route="煎服", asserted_numbers=[3.0, 9.0], asserted_unit="mg",
        cited_unit_ids=[unit_id],
    )
    llm = ScriptedLLM([batch(wrong_unit)])
    answer = service_for(rows, llm=llm).answer(
        "麻黄用量是多少？", linked_entities=linked_herb()
    )

    rejections = [item for item in answer.rejections if item["claim_id"] == "c-unit"]
    assert rejections, answer.rejections
    assert FAIL_UNIT_MISMATCH in rejections[0]["hard_failures"]
    assert "mg" not in answer.answer_text


# ---------------------------------------------------------------- 读取限定
def test_read_is_scoped_by_run_and_limit():
    rows = [dosage_row("a1", "煎服，3～9g。", quote="麻黄煎服，3～9g。")]
    store = FakeHighRiskStore(rows)
    HighRiskService(store, run_id="run-test").answer(
        "麻黄用量是多少？", linked_entities=linked_herb()
    )

    assert store.calls, "高风险读取必须走专用入口"
    # 高风险链路会发不止一条只读查询（核验主语是否被完整识别也要查一次），
    # 因此按「带谓词白名单」定位断言读取这条，而不是假定它一定是第 0 条。
    reads = [item for item in store.calls if "predicates" in item]
    assert len(reads) == 1, [item["cypher"][:40] for item in store.calls]
    call = reads[0]
    assert call["run"] == "run-test"
    assert call["limit"] > 0
    assert list(call["predicates"]) == ["HAS_DOSAGE"]
    assert call["predicates"], "必须限定谓词白名单"
    assert call["subject_ids"] == [MAHUANG], call["subject_ids"]
    # 主语完整性核对同样必须限定在 ACTIVE 批次内，不能跨批次查实体。
    assert all(item["run"] == "run-test" for item in store.calls), store.calls


def test_toxic_dose_question_does_not_answer_with_unrelated_contraindication():
    """「使用多少会中毒」不得用过敏体质忌用顶替，并标成已核验。"""
    rows = [
        assertion_row(
            assertion_id="a-allergy",
            predicate="CONTRAINDICATED_IN",
            polarity="NEGATIVE",
            subject_id="H:蜈蚣",
            subject_name="蜈蚣",
            object_id="P:过敏体质",
            object_name="过敏体质忌用",
            evidence_quote="过敏体质者忌用。",
            chunk_texts=["过敏体质者忌用。"],
            chunk_page_starts=[367],
            chunk_page_ends=[368],
        )
    ]
    answer = service_for(rows).answer(
        "蜈蚣使用多少会中毒？",
        linked_entities=linked_herb("蜈蚣", "H:蜈蚣"),
    )

    assert answer.status is AnswerStatus.ABSTAIN
    assert "过敏" not in answer.answer_text
    assert not answer.claims


def test_toxic_dose_question_does_not_treat_regular_dosage_as_threshold():
    """只有常规用量、没有中毒剂量时，不得把煎服克数说成已核验的中毒阈值。"""
    rows = [dosage_row("a1", "煎服，1～3g。", quote="蜈蚣煎服，1～3g。", subject_id="H:蜈蚣", subject_name="蜈蚣")]
    answer = service_for(rows).answer(
        "蜈蚣使用多少会中毒？",
        linked_entities=linked_herb("蜈蚣", "H:蜈蚣"),
    )

    assert answer.status is AnswerStatus.PARTIAL_GROUNDED
    assert "1～3g" in answer.answer_text
    assert "中毒剂量" in "；".join(item.reason for item in answer.sub_questions)
    assert answer.sub_questions[0].status is AnswerStatus.PARTIAL_GROUNDED


def test_shared_incompat_question_does_not_invent_common_partner():
    """两味药「均不能搭配」时，不得把人参反藜芦说成麻黄也反藜芦。"""
    from graphragexpr.highrisk.service import (
        categories_of,
        is_high_risk_question,
    )
    from graphragexpr.domain.high_risk import RiskCategory

    question = "人参和麻黄均不能同时和什么药搭配使用"
    assert is_high_risk_question(question) is True
    assert RiskCategory.INCOMPATIBILITY in categories_of(question)
    assert RiskCategory.DOSAGE not in categories_of(question)

    rows = [
        assertion_row(
            assertion_id="a-rs",
            predicate="INCOMPATIBLE_WITH",
            polarity="NEGATIVE",
            subject_id="H:人参",
            subject_name="人参",
            object_id="H:藜芦",
            object_name="藜芦",
            evidence_quote="不宜与藜芦同用。",
            chunk_texts=["不宜与藜芦同用。"],
            page_start=379,
            page_end=379,
            chunk_page_starts=[379],
            chunk_page_ends=[379],
        )
    ]
    answer = service_for(rows).answer(
        question,
        linked_entities=[
            {"canonical_id": "H:人参", "entity_type": "HERB", "name": "人参"},
            {"canonical_id": "H:麻黄", "entity_type": "HERB", "name": "麻黄"},
        ],
    )

    notes = "；".join(answer.coverage_notes)
    assert "藜芦" in answer.answer_text
    assert "麻黄" in answer.answer_text
    assert "未记载配伍禁忌" in answer.answer_text
    assert "共同" in answer.answer_text
    assert "麻黄" in notes
    assert answer.status is AnswerStatus.PARTIAL_GROUNDED

    store = FakeHighRiskStore(rows)
    HighRiskService(store, run_id="run-test").answer(
        question,
        linked_entities=[
            {"canonical_id": "H:人参", "entity_type": "HERB", "name": "人参"},
            {"canonical_id": "H:麻黄", "entity_type": "HERB", "name": "麻黄"},
        ],
    )
    predicate_reads = [item for item in store.calls if "predicates" in item]
    assert predicate_reads
    assert list(predicate_reads[0]["predicates"]) == ["INCOMPATIBLE_WITH"]
