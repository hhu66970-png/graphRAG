"""旧极性解释的分场景否定作用域测试。

教材里「不宜与藜芦同用」「孕妇慎用」都是**肯定地陈述了一条禁忌**：NEGATIVE 否定的
是「同用」「使用」这个动作，不是禁忌本身不存在。解释反了就会得出「麻黄可以和藜芦
同用」「麻黄孕妇可用」——最危险的错误。反过来，「无配伍禁忌」否定的是关系本身。

因此这里逐场景断言：解释结果三态、歧义标记、以及「判不出来就不判」的保守立场。
"""

from __future__ import annotations

from graphragexpr.domain.high_risk import PolarityVerdict
from graphragexpr.highrisk.polarity import PolarityInterpretation, interpret_polarity


def interpret(
    predicate: str,
    raw_polarity: str | None,
    quote: str,
    object_name: str | None = None,
) -> PolarityInterpretation:
    """按固定参数名调用解释器，使各用例只突出变化的那个维度。"""
    return interpret_polarity(
        predicate=predicate,
        raw_polarity=raw_polarity,
        evidence_quote=quote,
        object_name=object_name,
    )


# ---------------------------------------------------------------- 关系成立
def test_negated_co_use_action_keeps_the_incompatibility():
    """「不宜与藜芦同用」否定的是「同用」这个动作，配伍禁忌关系因此成立。"""
    interpretation = interpret("INCOMPATIBLE_WITH", "NEGATIVE", "不宜与藜芦同用。", "藜芦")

    assert interpretation.verdict is PolarityVerdict.RELATION_HOLDS
    assert interpretation.ambiguous is False
    assert interpretation.relation_holds is True
    # 原始极性必须原样带回，供调用方与解释结果分开保存
    assert interpretation.raw_polarity == "NEGATIVE"
    assert "不宜与" in interpretation.matched_cues


def test_eighteen_antagonism_formula_keeps_the_incompatibility():
    """「人参反藜芦」是十八反的肯定句式，配伍禁忌关系成立。"""
    interpretation = interpret("INCOMPATIBLE_WITH", "POSITIVE", "人参反藜芦。", "藜芦")

    assert interpretation.verdict is PolarityVerdict.RELATION_HOLDS
    assert interpretation.ambiguous is False
    assert "反藜芦" in interpretation.matched_cues


def test_restricted_use_keeps_the_contraindication():
    """「孕妇慎用」没有「同用」句式，但程度词限定的正是「使用」，禁忌成立。"""
    interpretation = interpret("CONTRAINDICATED_IN", "NEGATIVE", "孕妇慎用。", "孕妇")

    assert interpretation.verdict is PolarityVerdict.RELATION_HOLDS
    assert interpretation.ambiguous is False
    assert interpretation.relation_holds is True
    assert "慎用" in interpretation.matched_cues


def test_positive_polarity_without_negation_cues_holds():
    interpretation = interpret(
        "INCOMPATIBLE_WITH", "POSITIVE", "麻黄与藜芦同用需谨慎。", "藜芦"
    )

    assert interpretation.verdict is PolarityVerdict.RELATION_HOLDS
    assert interpretation.ambiguous is False
    assert interpretation.relation_holds is True


# ---------------------------------------------------------------- 关系不成立
def test_relation_negation_means_the_relation_does_not_hold():
    """「无配伍禁忌」否定的是禁忌关系本身，方向与「不宜…同用」相反。"""
    interpretation = interpret("INCOMPATIBLE_WITH", "NEGATIVE", "无配伍禁忌。", "藜芦")

    assert interpretation.verdict is PolarityVerdict.RELATION_DOES_NOT_HOLD
    assert interpretation.ambiguous is False
    assert interpretation.relation_holds is False
    assert "无配伍禁忌" in interpretation.matched_cues


def test_relation_negation_takes_precedence_over_the_polarity_label():
    """「无禁忌」这类写法优先判为关系不成立，即使极性被标成 POSITIVE。"""
    interpretation = interpret("INCOMPATIBLE_WITH", "POSITIVE", "无配伍禁忌。", "藜芦")

    assert interpretation.verdict is PolarityVerdict.RELATION_DOES_NOT_HOLD
    assert interpretation.relation_holds is False


# ---------------------------------------------------------------- 判不出来就不判
def test_predicate_that_does_not_match_the_quote_is_not_guessed():
    """同一句原文换成配伍谓词，就不能套用「使用禁忌」那套解释，只判无法确定。"""
    interpretation = interpret("INCOMPATIBLE_WITH", "NEGATIVE", "孕妇慎用。", "孕妇")

    assert interpretation.verdict is PolarityVerdict.UNDETERMINED
    assert interpretation.ambiguous is True
    assert interpretation.relation_holds is None


def test_missing_polarity_without_cues_is_undetermined():
    """缺极性标注、原文也无线索时判为无法确定，不得默认成立或不成立。"""
    for raw_polarity in (None, "", "   "):
        interpretation = interpret(
            "CONTRAINDICATED_IN", raw_polarity, "麻黄是一味解表药。", "孕妇"
        )

        assert interpretation.verdict is PolarityVerdict.UNDETERMINED
        assert interpretation.ambiguous is True
        assert interpretation.relation_holds is None
        assert interpretation.raw_polarity is None

    # 连原文都没有时同样不猜
    blank = interpret("INCOMPATIBLE_WITH", None, "", "藜芦")
    assert blank.verdict is PolarityVerdict.UNDETERMINED
    assert blank.ambiguous is True


def test_relation_holds_property_matches_the_verdict():
    """relation_holds 是三态：None 表示不得据此输出确定结论。"""
    cases = [
        (interpret("INCOMPATIBLE_WITH", "NEGATIVE", "不宜与藜芦同用。", "藜芦"), True),
        (interpret("INCOMPATIBLE_WITH", "NEGATIVE", "无配伍禁忌。", "藜芦"), False),
        (interpret("INCOMPATIBLE_WITH", "NEGATIVE", "孕妇慎用。", "孕妇"), None),
    ]

    assert [interpretation.relation_holds for interpretation, _ in cases] == [
        True,
        False,
        None,
    ]
    for interpretation, expected in cases:
        assert interpretation.relation_holds is expected
        assert (interpretation.relation_holds is True) is (
            interpretation.verdict is PolarityVerdict.RELATION_HOLDS
        )
        assert (interpretation.relation_holds is False) is (
            interpretation.verdict is PolarityVerdict.RELATION_DOES_NOT_HOLD
        )
