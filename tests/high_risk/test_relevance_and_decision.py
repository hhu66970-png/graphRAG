"""回答组织：问题条件相关性选择与直接结论。

为什么单独一组：这一组检查的不是「结论有没有证据」（那是
`test_acceptance.py` 与前两层核验的事），而是**同一批已核验证据里，
哪些属于用户问的那件事**，以及**第一句话是不是直接回答了问题**。

拆改前的实际表现（真实库，附子）：「附子孕妇能用吗？」会先输出
「以下结论均通过程序核验…」一行元信息，再按限制程度从重到轻列出
「附子 与 阴虚阳亢者：使用禁忌（忌用）」——用户问的是孕妇，第一眼读到的
却是与孕妇无关的「忌用」。这一组用例把这种组织方式钉死。

环境：全部用 `tests/high_risk/conftest.py` 的内存替身，不连数据库。
"""

from __future__ import annotations

import pytest

from graphragexpr.domain.high_risk import AnswerStatus
from graphragexpr.highrisk.relevance import (
    Decision,
    Relevance,
    classify,
    decide,
)
from graphragexpr.highrisk.render import QUOTE_LABEL
from graphragexpr.highrisk.service import HighRiskService

from .conftest import FakeHighRiskStore, assertion_row

FUZI = "H:附子"


def service_for(rows, *, budget: int = 3000) -> HighRiskService:
    """构造带假存储的高风险服务。"""
    return HighRiskService(
        FakeHighRiskStore(rows), run_id="run-test", unit_token_budget=budget
    )


def linked_fuzi() -> list[dict]:
    """测试用：已链接的附子实体。"""
    return [{"canonical_id": FUZI, "entity_type": "HERB", "preferred_name": "附子"}]


def contra(
    assertion_id: str,
    *,
    object_name: str,
    quote: str,
    page: int,
    object_id: str | None = None,
    subject_name: str = "附子",
    subject_id: str = FUZI,
) -> dict:
    """测试用：禁忌证据行。"""
    return assertion_row(
        assertion_id=assertion_id,
        predicate="CONTRAINDICATED_IN",
        polarity="NEGATIVE",
        subject_id=subject_id,
        subject_name=subject_name,
        object_id=object_id or f"P:{object_name}",
        object_name=object_name,
        evidence_quote=quote,
        chunk_texts=[quote],
        chunk_page_starts=[page],
        chunk_page_ends=[page],
    )


def incompat(assertion_id: str, *, object_name: str, quote: str, page: int) -> dict:
    """测试用：配伍禁忌证据行。"""
    return assertion_row(
        assertion_id=assertion_id,
        predicate="INCOMPATIBLE_WITH",
        polarity="NEGATIVE",
        subject_id=FUZI,
        subject_name="附子",
        object_id=f"H:{object_name}",
        object_name=object_name,
        evidence_quote=quote,
        chunk_texts=[quote],
        chunk_page_starts=[page],
        chunk_page_ends=[page],
    )


#: 与真实库同一形态的一批证据：孕妇慎用、妊娠慎用、另一个人群的忌用。
def fuzi_rows() -> list[dict]:
    """测试用：附子相关高风险行。"""
    return [
        contra("a-p231", object_name="孕妇", quote="孕妇慎用。", page=231),
        contra(
            "a-p67",
            object_name="妊娠",
            quote="妊娠慎用药主要包括桃仁、红花、附子、大黄等。",
            page=67,
        ),
        contra(
            "a-p230",
            object_name="阴虚阳亢者",
            quote="阴虚阳亢者忌用。",
            page=230,
            object_id="P:阴虚阳亢",
        ),
    ]


def answer(question: str, rows: list[dict]):
    """调用服务得到高风险回答。"""
    return service_for(rows).answer(question, linked_entities=linked_fuzi())


def units_of(rows):
    """从回答中取出证据单元。"""
    from graphragexpr.highrisk.units import build_units_from_assertions

    return build_units_from_assertions(rows)


# ------------------------------------------------------- ① 相关性分档本身
def test_direct_population_evidence_is_direct_not_unconditional():
    """命中问题指定人群的证据是 A（精确匹配），不是 B（无条件通用）。

    这条曾经写错：当时把「问题问了人群」一律落成 B，于是「孕妇慎用」被降级
    成「次要补充」，而首句判定只认 A，最终给出「无法判断孕妇能否使用」——
    明明有证据却说不知道。A 与 B 的区别必须由**证据是否命中该人群**决定。
    """
    rows = fuzi_rows()
    by_object = {unit.object_name: unit for unit in units_of(rows)}

    assert classify("附子孕妇能用吗？", by_object["孕妇"]) is Relevance.DIRECT
    assert classify("附子孕妇能用吗？", by_object["妊娠"]) is Relevance.DIRECT
    assert classify("附子孕妇能用吗？", by_object["阴虚阳亢者"]) is Relevance.MISMATCH
    # 问题没提人群时，不做无中生有的筛选
    assert classify("附子的用量是多少？", by_object["阴虚阳亢者"]) is Relevance.DIRECT


def test_unconditional_evidence_is_the_second_tier():
    """没有限定条件（宾语未识别出来）的证据是 B：仍可用，但排在精确匹配之后。"""
    unit = units_of([contra("a-nobody", object_name="", quote="附子有毒，不宜久服。", page=100)])[0]
    unit = unit.model_copy(update={"object_name": None})

    assert classify("附子孕妇能用吗？", unit) is Relevance.UNCONDITIONAL
    # 宾语非空却是别人：即使不在同义表里，也算「其他人群」
    other = units_of([contra("a-case", object_name="湿热证", quote="湿热证忌用。", page=101)])[0]
    assert classify("附子孕妇能用吗？", other) is Relevance.MISMATCH
    # 配伍禁忌的宾语是**另一味药**，不是人群条件，不能因此判为其他人群
    pair = units_of([incompat("a-banxia", object_name="半夏", quote="不宜与半夏同用。", page=102)])[0]
    pair = pair.model_copy(update={"population": None})
    assert classify("附子孕妇能用吗？", pair) is Relevance.UNCONDITIONAL


# ------------------------------------------------------- ②③ 直接结论
@pytest.mark.parametrize(
    "quote,severity_text,expected,first_sentence",
    [
        ("孕妇慎用。", "慎用", Decision.CAUTION, "不能直接判断为可以使用。"),
        ("孕妇忌用。", "忌用", Decision.AVOID, "不宜使用。"),
        ("孕妇禁用。", "禁用", Decision.PROHIBITED, "不能使用。"),
    ],
)
def test_decision_degree_maps_to_its_own_sentence(
    quote, severity_text, expected, first_sentence
):
    """禁用／忌用／慎用三档各自对应一句，且第一句直接回答。

    三档不得互相替代：把「慎用」说成「不能使用」是过度限制，
    把「禁用」说成「慎用」是放行。
    """
    result = answer("附子孕妇能用吗？", [contra("a1", object_name="孕妇", quote=quote, page=231)])

    assert result.decision == expected.value
    assert result.decision_text.startswith(first_sentence)
    assert severity_text in result.decision_text
    assert result.answer_text.startswith(result.decision_text)
    # 第一句不得是元信息
    assert not result.answer_text.startswith("以下结论")


def test_no_caution_evidence_never_becomes_absolute_clearance():
    """只有「慎用」证据时，输出既不能说「可以用」，也不能说「禁止使用」。"""
    result = answer(
        "附子孕妇能用吗？",
        [contra("a1", object_name="孕妇", quote="孕妇慎用。", page=231)],
    )
    text = result.answer_text

    assert "禁止使用" not in text
    assert "可以安全" not in text
    assert "无禁忌" not in text
    assert "不能直接判断为可以使用" in text


def test_unknown_decision_never_reads_as_permission():
    """没有与问题条件匹配的证据时，说「无法判断」，绝不说「可以使用」。"""
    result = answer("附子孕妇能用吗？", [])

    assert result.decision == Decision.UNKNOWN.value
    assert result.decision_text.startswith("根据当前教材证据，无法判断孕妇能否使用")
    for phrase in ("可以同用", "无禁忌", "没有禁忌", "可以服用", "安全使用"):
        assert phrase not in result.answer_text, phrase


# ------------------------------------------------------- ④⑤⑥ 相关性选择
def test_other_condition_evidence_leaves_the_main_answer():
    """④ 孕妇问题的主要 claims／evidence_units 里不得出现「阴虚阳亢者忌用」。"""
    result = answer("附子孕妇能用吗？", fuzi_rows())

    assert result.status is AnswerStatus.GROUNDED
    objects = {claim.object_name for claim in result.claims}
    assert objects == {"孕妇", "妊娠"}, objects
    assert "阴虚阳亢者" not in result.answer_text
    assert all(
        unit.object_name != "阴虚阳亢者" for unit in result.evidence_units
    )
    # 不丢：单独放在默认折叠区，标注与本题人群不同
    assert [unit.object_name for unit in result.other_conditions] == ["阴虚阳亢者"]


def test_exact_population_evidence_outranks_more_severe_unconditional_evidence():
    """⑤ 精确匹配排在无条件通用之前——哪怕后者限制程度更重。

    相关性优先于限制程度。反过来排的话，「忌用（未限定人群）」会排在
    「孕妇慎用」前面，读者先看到的又是一条与所问人群无关的限制。
    """
    result = answer(
        "附子孕妇能用吗？",
        [
            contra("a-p231", object_name="孕妇", quote="孕妇慎用。", page=231),
            contra("a-nobody", object_name="", quote="附子有毒，不宜久服。", page=100),
        ],
    )

    assert len(result.claims) == 2, [c.statement for c in result.claims]
    first = result.claims[0]
    assert first.object_name == "孕妇", [c.object_name for c in result.claims]
    # B 只是排在后面，不是被丢掉
    assert [c.object_name or "" for c in result.claims] == ["孕妇", ""]


def test_other_population_evidence_cannot_decide_or_ground():
    """⑥ 其他人群的证据既不能产生首句，也不能把状态抬成有据。"""
    result = answer("附子哺乳期能用吗？", fuzi_rows())

    assert result.status is AnswerStatus.SOURCE_ONLY
    assert result.status is not AnswerStatus.GROUNDED
    assert result.claims == []
    assert result.decision == Decision.UNKNOWN.value
    assert "哺乳期" in result.decision_text
    assert "阴虚阳亢者" not in result.answer_text
    assert "孕妇慎用" not in result.answer_text


def test_same_population_conflict_still_downgrades():
    """⑦ 同一人群下的程度冲突仍然降级，且不被相关性过滤藏掉。

    相关性只回答「是不是本题的人」，不参与程度取舍。两处来源对孕妇一派
    说慎用、一派说禁用时，必须并列说明并降级——不得挑更严的一档。
    """
    result = answer(
        "附子孕妇能用吗？",
        [
            contra("a-caution", object_name="孕妇", quote="孕妇慎用。", page=231),
            contra("a-prohibit", object_name="孕妇", quote="孕妇禁用。", page=232),
        ],
    )

    assert result.status is AnswerStatus.PARTIAL_GROUNDED
    assert result.decision == Decision.CONFLICT.value
    assert "慎用" in result.decision_text and "禁用" in result.decision_text
    assert "禁" not in result.decision_text.split("在")[0]  # 首句不挑一档
    assert len(result.claims) == 2
    reasons = "；".join(
        item.reason for item in result.sub_questions
    ) + "；".join(result.coverage_notes)
    assert "不一致的限制程度" in reasons
    # 普通用户不该被要求去做人工核对
    assert "人工核对" not in reasons
    assert "暂不作单一结论" in reasons


def test_decide_sees_conflicting_evidence_outside_the_context():
    """⑦ 首句判定必须看到**读到的全部**单元，而不只是进了上下文的那一批。

    真实库实测（川乌孕妇）：第 179 页「孕妇慎用」被 token 预算裁掉，只剩第 67 页
    的「妊娠禁用」，首句因此成了「不能使用。教材明确记载：孕妇禁用川乌。」，
    而同一份回答的下文又写着「存在不一致的限制程度…系统暂不作单一结论」——
    首句与正文自相矛盾，且矛盾的方向是**更严**的那一档。
    """
    units = units_of(
        [
            contra("a-caution", object_name="孕妇", quote="孕妇慎用。", page=179),
            contra("a-prohibit", object_name="妊娠", quote="妊娠禁用。", page=67),
        ]
    )
    by_object = {unit.object_name: unit for unit in units}
    shown = by_object["孕妇"]

    result = decide(
        "附子孕妇能用吗？",
        [shown],
        relevance={shown.unit_id: Relevance.DIRECT},
        read_units=units,
    )

    assert result.decision is Decision.CONFLICT
    assert "慎用" in result.text and "禁用" in result.text
    assert "不能使用" not in result.text


def test_conflict_survives_the_token_budget_in_the_first_sentence():
    """⑦ 预算裁剪之后的整份回答：首句仍是冲突句，不选更严的一档。"""
    rows = [
        contra("a-caution", object_name="孕妇", quote="孕妇慎用。", page=179),
        contra("a-prohibit", object_name="妊娠", quote="妊娠禁用。", page=67),
    ]
    result = service_for(rows, budget=1).answer(
        "附子孕妇能用吗？", linked_entities=linked_fuzi()
    )

    # 预算只够一个单元：这正是「两处来源只剩一处」的情形
    assert len(result.evidence_units) < 2, [u.unit_id for u in result.evidence_units]
    assert result.status is AnswerStatus.PARTIAL_GROUNDED
    assert result.decision == Decision.CONFLICT.value
    assert "慎用" in result.decision_text and "禁用" in result.decision_text
    assert "不能使用" not in result.decision_text


# ------------------------------------------------------- ⑧⑨⑩ 核验说明
def test_grounded_verification_note_sits_after_everything():
    """⑧ 全部成立时，核验说明位于答案末尾。"""
    result = answer("附子孕妇能用吗？", fuzi_rows())

    assert result.verification_note.startswith("以上结论")
    assert result.answer_text.rstrip().endswith(result.verification_note)
    assert not result.answer_text.startswith("以上结论")
    assert not result.answer_text.startswith("以下结论")


def test_partial_verification_note_does_not_claim_everything():
    """⑨ 部分成立时不得写成「均通过」。"""
    result = answer(
        "附子用量是多少？孕妇能用吗？",
        [
            contra("a-p231", object_name="孕妇", quote="孕妇慎用。", page=231),
        ],
    )

    assert result.status is AnswerStatus.PARTIAL_GROUNDED
    assert result.verification_note.startswith("以上已标记为有据的结论通过程序核验")
    assert "均通过" not in result.verification_note


@pytest.mark.parametrize("question", ["附子哺乳期能用吗？", "这个药怎么用？"])
def test_no_verification_note_without_verified_conclusions(question):
    """⑩ 仅来源与拒答都不显示「通过核验」说明。"""
    result = service_for([]).answer(question, linked_entities=linked_fuzi())

    assert result.status in (AnswerStatus.SOURCE_ONLY, AnswerStatus.ABSTAIN)
    assert result.verification_note == ""


# ------------------------------------------------------- ⑪⑭ 文案与标签
def test_quotes_are_labelled_as_origin():
    """⑪ 原文依据有统一标签，并与结论分开。"""
    result = answer("附子孕妇能用吗？", fuzi_rows())

    quotes = [b for b in result.answer_blocks if b.kind.value == "quote"]
    assert quotes, result.answer_blocks
    assert all(block.text.startswith(QUOTE_LABEL) for block in quotes)
    assert all(QUOTE_LABEL in line for line in result.answer_text.splitlines() if "「" in line)
    # 结论块不含原文引文，原文块不含结论行
    conclusion = [b for b in result.answer_blocks if b.kind.value == "conclusion"]
    assert all(QUOTE_LABEL not in b.text for b in conclusion)


def test_unresolved_notes_are_written_for_readers_not_maintainers():
    """⑭ 普通用户文案里不再出现「待核对」「人工复核」。"""
    from graphragexpr.highrisk.render import render_evidence_card

    from .conftest import dosage_row
    from graphragexpr.highrisk.units import build_units_from_assertions
    from graphragexpr.highrisk.models import EvidenceCard

    unit = build_units_from_assertions(
        [dosage_row("a-inv", "9～3g", quote="麻黄 9～3g。")]
    )[0]
    card = EvidenceCard.from_unit(unit)
    text = render_evidence_card(card)

    for word in ("待核对", "人工核对", "人工复核", "维护人员", "人工确认"):
        assert word not in text, word
    assert "当前证据不足以确定" in text
    # 同一件事只说一遍：单元级说明与卡片说明都不得把「上界小于下界」讲两遍
    assert text.count("上界小于下界") == 1, text
    # 解析层内部码不得出现在用户可见文本里
    assert "inverted_range:" not in text
    assert "unit_not_adjacent:" not in text


def test_parser_explanation_is_not_dressed_up_as_a_disagreement():
    """⑭ 解析层**说明**不能被写成「证据存在差异」。

    真实库实测（附子用量，第 231 页）：原文「煎服，3～15g；先煎，久煎，
    口尝至无麻辣感为度」里，后续小句自身不含剂量，解析器把它并入本数值的
    条件，并记了一条**说明**。一律套「证据存在差异…系统暂不作单一结论」时，
    卡片会对着这条已核验的 3～15g 说「存在差异」「不作单一结论」，
    而同一张卡的徽标写着「高风险 · 已核验」——两句话互相打脸，
    且差异并不存在。
    """
    from graphragexpr.highrisk.models import EvidenceCard
    from graphragexpr.highrisk.render import card_flags, unresolved_flags
    from graphragexpr.highrisk.units import build_units_from_assertions

    from .conftest import dosage_row

    # 存量 raw 与教材原文同为一句（真实库里就是这样存的）
    unit = build_units_from_assertions(
        [
            dosage_row(
                "a-merge",
                "煎服，3～15g；先煎，久煎，口尝至无麻辣感为度",
                quote="附子的用法用量：煎服，3～15g；先煎，久煎，口尝至无麻辣感为度。",
            )
        ]
    )[0]
    flags = unresolved_flags(unit)

    assert any(flag.startswith("原文说明：") for flag in flags), flags
    assert not any("证据存在差异" in flag for flag in flags), flags
    assert not any("暂不作单一结论" in flag for flag in flags), flags
    # 卡片显示的是同一批整句，不是另一套模板
    assert card_flags(EvidenceCard.from_unit(unit)) == flags


def test_conflict_wording_survives_only_for_real_disagreements():
    """⑭ 三类说明措辞各自归位：差异／不足／说明，不互相顶替。"""
    from graphragexpr.highrisk.models import EvidenceCard
    from graphragexpr.highrisk.render import card_flags
    from graphragexpr.highrisk.units import build_units_from_assertions

    from .conftest import assertion_row, dosage_row

    # 存量解析值与按原文重解析不一致：这才是真正的「差异」
    stored = build_units_from_assertions(
        [
            assertion_row(
                assertion_id="a-stored",
                predicate="HAS_DOSAGE",
                object_value={
                    "raw": "3～9g",
                    "value_min": 9.0,
                    "value_max": 3.0,
                    "unit": "g",
                },
                evidence_quote="麻黄 3～9g。",
                chunk_texts=["麻黄 3～9g。"],
            )
        ]
    )[0]
    stored_flags = card_flags(EvidenceCard.from_unit(stored))

    assert any(flag.startswith("证据存在差异：") for flag in stored_flags), stored_flags
    assert any("暂不作单一结论" in flag for flag in stored_flags), stored_flags

    # 数值没有紧邻单位：是「确定不了」，不是两处来源打架
    unbound = build_units_from_assertions(
        [dosage_row("a-unbound", "3～9", quote="麻黄 3～9。")]
    )[0]
    unbound_flags = card_flags(EvidenceCard.from_unit(unbound))

    assert any(flag.startswith("当前证据不足以确定：") for flag in unbound_flags)
    assert not any("证据存在差异" in flag for flag in unbound_flags), unbound_flags


# ------------------------------------------------------- 其它问法不受影响
def test_enumeration_question_gets_no_single_drug_decision():
    """反向枚举问的是一个集合，点名其中一味药会把部分清单说成结论。"""
    rows = [
        contra("a-qianjinzi", object_name="孕妇", quote="孕妇禁用。", page=300,
               subject_name="千金子", subject_id="H:千金子"),
        contra("a-xuchangqing", object_name="孕妇", quote="孕妇慎用。", page=301,
               subject_name="徐长卿", subject_id="H:徐长卿"),
    ]
    result = service_for(rows).answer(
        "孕妇禁用的中药有哪些？", linked_entities=[]
    )

    assert result.decision == Decision.NONE.value
    assert result.decision_text == ""
    # 结论行照常列出，且没有把某一味药说成第一句
    assert "千金子" in result.answer_text


def test_multi_population_question_keeps_each_degree():
    """问题同时问了两类人时，两个程度都要在，且不合成一句。"""
    rows = [
        contra("a-pop", object_name="孕妇", quote="孕妇慎用。", page=231,
               subject_name="麻黄", subject_id="H:麻黄"),
        contra("a-child", object_name="儿童", quote="儿童禁用。", page=232,
               subject_name="麻黄", subject_id="H:麻黄"),
    ]
    result = service_for(rows).answer("麻黄孕妇和儿童能用吗？", linked_entities=[
        {"canonical_id": "H:麻黄", "entity_type": "HERB", "preferred_name": "麻黄"}
    ])

    assert result.decision == Decision.NONE.value
    assert {unit.severity_label for unit in result.evidence_units} == {"慎用", "禁用"}
    assert "慎用" in result.answer_text and "禁用" in result.answer_text


def test_dosage_question_has_no_use_or_not_sentence():
    """用量类问题没有「能否使用」这一层，首句不生成。"""
    from .conftest import dosage_row

    result = service_for([dosage_row("a1", "煎服，3～9g。", quote="麻黄煎服，3～9g。")]).answer(
        "麻黄用量是多少？",
        linked_entities=[
            {"canonical_id": "H:麻黄", "entity_type": "HERB", "preferred_name": "麻黄"}
        ],
    )

    assert result.decision == Decision.NONE.value
    assert result.status is AnswerStatus.GROUNDED
    assert "3～9g" in result.answer_text
    assert result.verification_note.startswith("以上结论")
