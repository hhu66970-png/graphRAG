"""保守解析的直接测试（任务书 §5.2）。

验收测试只能从响应形态上间接看到解析结果（「未交换」「不借用别的用法」）；这里直接
对解析结果断言，覆盖旧解析器在高风险场景里**不能**做的三件事：

 1 不做倒置范围交换，只标记 `is_inverted`；
 2 单位只与数值紧邻配对，不到片段别处补一个单位；
 3 一处数值一项，条件取该数值的局部窗口，不补「成人」「每日」这类默认值。

条件窗口的边界是「上一处数值之后到下一处数值之前」，因此窗口原文的尾部会带上后一个
数值前面的线索词（render.py 的 `_unstructured_condition` 正是依赖这一事实、剔除整份
线索词表后才判断「窗口里是否还有未结构化内容」）。所以下面断言的是窗口不跨到相邻的
**数值**，以及**槽位**归属本数值，而不是窗口里不出现任何邻近线索词。
"""

from __future__ import annotations

from graphragexpr.domain.high_risk import RestrictionSeverity
from graphragexpr.highrisk.parsing import (
    ALL_CONDITION_CUES,
    classify_severity,
    extract_population,
    extract_preparation,
    parse_dosage_units,
)


# ---------------------------------------------------------------- 一处数值一项
def test_each_dosage_occurrence_gets_its_own_binding():
    """一句话里两处剂量 → 两个绑定，各自带自己窗口内的用法。"""
    bindings = parse_dosage_units("煎服，3～9g；挽救虚脱可用 15～30g。")

    assert [binding.raw for binding in bindings] == ["3～9g", "15～30g"]
    assert (bindings[0].value_min, bindings[0].value_max) == (3.0, 9.0)
    assert (bindings[1].value_min, bindings[1].value_max) == (15.0, 30.0)
    assert [binding.unit for binding in bindings] == ["g", "g"]
    assert bindings[0].route == "煎服"
    # 「挽救虚脱可用」不在用法词表内，第二处数值不得借用第一处的「煎服」
    assert bindings[1].route is None
    assert "煎服" not in (bindings[1].condition_text or "")


def test_two_doses_in_one_clause_keep_their_own_preparation():
    """同一片段内按生制品分档 → 两个绑定，炮制方式不得互换。"""
    bindings = parse_dosage_units("生品 1.5～3g，制品 3～6g。")

    assert [binding.raw for binding in bindings] == ["1.5～3g", "3～6g"]
    assert [binding.preparation for binding in bindings] == ["生品", "制品"]


def test_condition_window_is_bounded_by_the_neighbouring_numbers():
    """窗口只覆盖本数值到相邻数值之间，且槽位（而非窗口原文）归属本数值。"""
    first, second = parse_dosage_units("生品 1.5～3g，制品 3～6g。")

    assert "1.5～3g" in (first.condition_text or "")
    assert "3～6g" not in (first.condition_text or "")
    assert "3～6g" in (second.condition_text or "")
    assert "1.5～3g" not in (second.condition_text or "")
    assert "生品" not in (second.condition_text or "")
    # 槽位归属：每个生制品线索的跨度都落在自己的数值上
    assert first.preparation == "生品"
    assert first.field_spans["preparation"].text == "生品"
    assert second.preparation == "制品"
    assert second.field_spans["preparation"].text == "制品"
    assert first.field_spans["value"].text == "1.5～3g"
    assert second.field_spans["value"].text == "3～6g"


# ---------------------------------------------------------------- 不修正、不补全
def test_inverted_range_is_flagged_and_not_swapped():
    """9～3g 只标记，绝不自动改成 3～9g：擅自交换会把 OCR 错误固化成事实。"""
    binding = parse_dosage_units("麻黄 9～3g。")[0]

    assert binding.is_inverted is True
    assert (binding.value_min, binding.value_max) == (9.0, 3.0)
    assert binding.raw == "9～3g"
    assert any("inverted_range" in note for note in binding.parse_notes)


def test_unit_must_be_adjacent_to_its_number():
    """紧邻处没有单位就留空并记 note，不到片段别处取一个单位拼上来。"""
    bindings = parse_dosage_units("用量 3～9，另服 6g。")

    assert bindings[0].raw == "3～9"
    assert bindings[0].unit is None and bindings[0].unit_raw is None
    assert any("unit_not_adjacent" in note for note in bindings[0].parse_notes)
    # 后面的「6g」属于后一处数值，不能被前一处借走
    assert bindings[1].raw == "6g"
    assert bindings[1].unit == "g" and bindings[1].unit_raw == "g"


def test_lone_count_is_not_a_dose():
    """孤立数字（无范围、紧邻无单位）不是剂量：「3 天」不得被读成 3g。"""
    assert [binding.raw for binding in parse_dosage_units("3 天，每次 9g")] == ["9g"]


def test_page_reference_is_not_a_dose():
    """页码区间与剂量形态相同，靠紧邻的「第／页」区分。"""
    assert parse_dosage_units("第 66-67 页。") == []
    assert [
        binding.raw for binding in parse_dosage_units("见第 66-67 页。煎服，3～9g。")
    ] == ["3～9g"]


def test_no_dosage_yields_empty_list():
    assert parse_dosage_units("") == []
    assert parse_dosage_units("   ") == []
    # 「适量外用」解析不出数值：由组装层保留为不明确单元，解析层不猜数值
    assert parse_dosage_units("适量外用") == []


def test_single_value_binding():
    binding = parse_dosage_units("研末服，0.5g")[0]

    assert binding.is_single_value is True
    assert binding.value_min == binding.value_max == 0.5
    assert binding.unit == "g" and binding.route == "研末服"


# ---------------------------------------------------------------- 限制程度
def test_severity_three_levels_stay_distinct():
    """禁用／忌用／慎用三档不得被抹平：慎用既不是禁用，也不是忌用。"""
    assert classify_severity("孕妇禁用。") == (RestrictionSeverity.PROHIBITED, "禁用")
    assert classify_severity("严禁与藜芦同用。") == (RestrictionSeverity.PROHIBITED, "严禁")
    assert classify_severity("孕妇忌用。") == (RestrictionSeverity.AVOID, "忌用")
    assert classify_severity("孕妇不宜服用。") == (RestrictionSeverity.AVOID, "不宜")

    severity, raw = classify_severity("孕妇慎用。")
    assert severity is RestrictionSeverity.CAUTION
    assert raw == "慎用"
    assert severity is not RestrictionSeverity.PROHIBITED
    assert severity is not RestrictionSeverity.AVOID


def test_severity_without_cue_is_unspecified():
    """教材没给程度词就如实留空，不用近似词顶上。"""
    assert classify_severity("煎服，3～9g。") == (RestrictionSeverity.UNSPECIFIED, None)
    assert classify_severity("") == (RestrictionSeverity.UNSPECIFIED, None)


# ---------------------------------------------------------------- 不补默认条件
def test_population_and_preparation_do_not_invent_defaults():
    """教材没写就不写：不补默认人群，也不推断药物形态。"""
    assert extract_population("煎服，3～9g。") is None
    assert extract_population("成人一次 3g") is None    # 「成人」不是线索词，不补默认人群
    assert extract_population("") is None
    assert extract_preparation("煎服，3～9g。") is None
    assert extract_preparation("") is None
    # 命中时返回原文词，供跨度与依据使用
    assert extract_population("孕妇慎用。") == "孕妇"
    assert extract_preparation("生品 1.5～3g") == "生品"


def test_bound_cues_are_covered_by_the_shared_cue_table():
    """解析能绑定的线索词必须都在 ALL_CONDITION_CUES 里（渲染层另抄一份就会漂移）。"""
    text = "孕妇慎用。生品 1.5～3g；一日三次，开水泡服 3～6g。"

    for binding in parse_dosage_units(text):
        for cue in (binding.route, binding.preparation, binding.frequency, binding.population):
            assert cue is None or cue in ALL_CONDITION_CUES, cue

    severity, raw = classify_severity(text)
    assert severity is RestrictionSeverity.CAUTION
    assert raw in ALL_CONDITION_CUES


# ---------------------------------------------------------------- 线索归属本数值
def test_preparation_of_the_next_dose_is_not_bound_to_the_previous_one():
    """「制品 3～6g，生品 1.5～3g」两个炮制分支必须各归其主。

    窗口注释写明「线索归属哪个数值由 `_nearest_cue` 按距离判定
    （「生品 1.5～3g，制品 3～6g」中两个程度词各归其主）」。这里只是把两个分支的
    先后顺序对调，句式完全相同，因此同样必须各归其主：若前一处数值拿到后一处的
    「生品」，两个分支就都成了生品剂量，核验也看不出错。
    """
    bindings = parse_dosage_units("制品 3～6g，生品 1.5～3g。")

    assert [binding.raw for binding in bindings] == ["3～6g", "1.5～3g"]
    assert [binding.preparation for binding in bindings] == ["制品", "生品"]


def test_route_of_the_next_dose_is_not_bound_to_the_previous_one():
    """同一片段内两种用法必须各归其主，否则 3～9g 会带上 1～2g 的用法。"""
    bindings = parse_dosage_units("煎服 3～9g，研末服 1～2g。")

    assert [binding.raw for binding in bindings] == ["3～9g", "1～2g"]
    assert [binding.route for binding in bindings] == ["煎服", "研末服"]
