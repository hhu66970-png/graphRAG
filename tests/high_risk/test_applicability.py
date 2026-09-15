"""问题—证据适用性核验的单元测试（第二层核验）。

这一层负责的对象错配、条件错配与「错误放行」三件事，都不是单看证据能发现的：
证据本身完全忠实，错的是「它回答的不是用户问的问题」。因此这里的用例全部
围绕**问题问了什么**与**证据声明了什么**的对照来写。

确定性：不连数据库、不调模型，直接构造单元。
"""

from __future__ import annotations

from graphragexpr.domain.high_risk import (
    ConditionalEvidenceUnit,
    RestrictionSeverity,
    RiskCategory,
    SourceKind,
)
from graphragexpr.highrisk.applicability import (
    Applicability,
    assess_applicability,
    is_enumeration_question,
    is_generic_object,
    longer_object_span,
    same_population,
    same_route_family,
    severity_conflicts,
    sort_by_severity,
)


def unit(
    *,
    unit_id: str = "u1",
    subject: str = "麻黄",
    object_name: str | None = None,
    object_id: str | None = None,
    population: str | None = None,
    preparation: str | None = None,
    route: str | None = None,
    severity: RestrictionSeverity = RestrictionSeverity.UNSPECIFIED,
    quote: str = "孕妇慎用",
) -> ConditionalEvidenceUnit:
    """One evidence unit whose only varying dimensions are the ones under test."""
    return ConditionalEvidenceUnit(
        unit_id=unit_id,
        risk_category=RiskCategory.CONTRAINDICATION,
        predicate="CONTRAINDICATED_IN",
        subject_id=f"H:{subject}",
        subject_name=subject,
        object_id=object_id,
        object_name=object_name,
        population=population,
        preparation=preparation,
        route=route,
        severity=severity,
        evidence_quote=quote,
        page_start=1,
        source_kind=SourceKind.TEXT_CHUNK,
    )


# ------------------------------------------------------- 反向枚举问法判定
def test_yes_no_question_is_not_an_enumeration():
    """「布洛芬孕妇能吃吗」问的是某个对象，绝不能走反向枚举读取。"""
    assert is_enumeration_question("布洛芬孕妇能吃吗？") is False
    assert is_enumeration_question("附子哺乳期能用吗？") is False
    assert is_enumeration_question("这个药怎么用？") is False


def test_enumeration_question_is_recognized():
    """测试：枚举型问题为被识别。"""
    assert is_enumeration_question("孕妇禁用的中药有哪些？") is True
    assert is_enumeration_question("孕妇不能吃哪些药？") is True
    assert is_enumeration_question("请列举孕妇慎用的中药") is True


# --------------------------------------------------------------- 泛称对象
def test_generic_class_words_are_not_a_named_object():
    """测试：泛化类别词不具名宾语。"""
    assert is_generic_object("中药") is True
    assert is_generic_object("药材") is True


def test_specific_herb_names_are_not_generic():
    """只做全等匹配，具体药材名不会被误伤。"""
    assert is_generic_object("川乌") is False
    assert is_generic_object("制川乌") is False
    assert is_generic_object("附子") is False
    assert is_generic_object("") is False


# ------------------------------------------------- 名称被截断（对象错配）
def test_patent_medicine_name_is_not_a_single_herb():
    """本体里没有「龙胆泻肝丸」这个实体，靠实体名比对发现不了名称被截断，
    于是它会拿「龙胆」的剂量作答。形态判定必须兜住这一类。"""
    assert longer_object_span("龙胆泻肝丸的用量是多少？", "龙胆") == "龙胆泻肝丸"
    assert longer_object_span("逍遥丸孕妇能吃吗？", "逍遥") == "逍遥丸"


def test_adjacent_dosage_form_after_the_name_is_detected():
    """测试：紧邻的剂量剂型之后名称为被检测到。"""
    assert longer_object_span("六味地黄丸孕妇能用吗？", "地黄") == "地黄丸"


def test_preparation_verbs_are_not_mistaken_for_another_object():
    """「煮汤」「泡酒」说的是用法，不是另一个对象；判成对象错配会凭空拒答。
    「饮片」「切片」指的还是这味药材本身。"""
    assert longer_object_span("附子煮汤怎么用？", "附子") is None
    assert longer_object_span("附子泡酒外用行吗？", "附子") is None
    assert longer_object_span("附子饮片用量是多少？", "附子") is None


def test_ordinary_questions_do_not_trip_the_truncation_check():
    """测试：普通问题不误触发截断检查。"""
    assert longer_object_span("附子孕妇能用吗？", "附子") is None
    assert longer_object_span("附子不宜与什么药同用？", "附子") is None
    assert longer_object_span("附子的用量是多少？", "附子") is None


def test_truncation_check_stops_at_punctuation():
    """名称与剂型字之间跨了标点就不是同一个名称了。"""
    assert longer_object_span("附子的用法；丸剂怎么吃？", "附子") is None


# ----------------------------------------------------------- 人群同义映射
def test_pregnant_and_gestation_are_the_same_population():
    """「妊娠慎用药…附子」是第 67 页对孕妇问题的**正确**证据，不能被判成不匹配。"""
    assert same_population("孕妇", "妊娠") is True
    assert same_population("孕妇", "妊娠期") is True


def test_lactation_is_not_pregnancy():
    """这是本次要拦的核心错配：哺乳期的问题不能用孕妇的证据回答。"""
    assert same_population("哺乳期", "孕妇") is False
    assert same_population("小儿", "妊娠") is False


def test_populations_outside_the_table_do_not_match():
    """表外一律不匹配——「不知道是否同一人群」不能凑成匹配。"""
    assert same_population("孕妇", "脾胃虚寒者") is False
    assert same_population("孕妇", None) is False
    assert same_population(None, "孕妇") is False


def test_unit_population_mentioned_inside_a_longer_phrase_still_matches():
    """单元常写「气虚津亏者及孕妇」，仍应认作孕妇人群。"""
    assert same_population("孕妇", "气虚津亏者及孕妇") is True


# --------------------------------------------------------------- 用法家族
def test_decoction_and_oral_are_the_same_route_family():
    """煎服就是内服的一种：按子串判会把问「内服」与证据「煎服」判成不匹配，
    从而把一个本来正确的剂量答案降级。"""
    assert same_route_family("内服", "煎服") is True


def test_oral_and_topical_are_different_families():
    """测试：内服且外用不同途径族。"""
    assert same_route_family("外用", "内服") is False
    assert same_route_family("内服", "外用") is False


# ----------------------------------------------------- 适用性核验主判定
def test_population_mismatch_blocks_grounded():
    """哺乳期的问题 + 只有孕妇证据 → 不适用，理由必须点名所问人群。"""
    assessment = assess_applicability(
        "附子哺乳期能用吗？", [unit(population="孕妇", severity=RestrictionSeverity.CAUTION)]
    )

    assert assessment.verdict is Applicability.CONDITION_MISMATCH
    assert assessment.allows_grounded is False
    assert "哺乳期" in assessment.reason


def test_matching_population_allows_grounded():
    """测试：匹配的人群允许有证据支撑。"""
    assessment = assess_applicability(
        "附子孕妇能用吗？", [unit(population="孕妇", severity=RestrictionSeverity.CAUTION)]
    )

    assert assessment.verdict is Applicability.APPLICABLE
    assert assessment.allows_grounded is True


def test_population_mismatch_reports_which_populations_the_evidence_covers():
    """测试：人群不一致报告哪些人群证据覆盖。"""
    assessment = assess_applicability(
        "附子小儿能用吗？", [unit(population="妊娠"), unit(population="孕妇")]
    )

    assert "小儿" in assessment.reason
    assert "妊娠" in assessment.reason and "孕妇" in assessment.reason


def test_question_without_conditions_is_applicable():
    """问题没问人群/形态/用法时，不得凭空要求证据声明这些条件。"""
    assessment = assess_applicability("附子不宜与什么药同用？", [unit()])

    assert assessment.verdict is Applicability.APPLICABLE


def test_route_is_only_required_when_the_evidence_distinguishes_routes():
    """证据完全没提用法时，不能因为问题提了用法就判不适用——那会凭空制造拒答。"""
    assert (
        assess_applicability("藜芦外用安全吗？", [unit(route=None)]).verdict
        is Applicability.APPLICABLE
    )
    assert (
        assess_applicability("藜芦外用安全吗？", [unit(route="内服")]).verdict
        is Applicability.CONDITION_MISMATCH
    )


def test_preparation_mismatch_blocks_grounded():
    """生品与制品剂量不可互相套用。"""
    assessment = assess_applicability("半夏生品用量是多少？", [unit(preparation="制品")])

    assert assessment.verdict is Applicability.CONDITION_MISMATCH
    assert "生品" in assessment.reason


def test_unconfirmed_subject_blocks_before_any_condition_check():
    """测试：未确认的主语拦截在…之前任何条件检查。"""
    assessment = assess_applicability(
        "安宫牛黄丸孕妇能用吗？",
        [unit(population="孕妇")],
        subject_unconfirmed_reason="链接只命中了名称子串",
    )

    assert assessment.verdict is Applicability.SUBJECT_UNCONFIRMED
    assert assessment.allows_grounded is False
    assert "子串" in assessment.reason


# --------------------------------------------------------- 程度冲突并列
def test_conflicting_severities_for_the_same_object_are_reported():
    """同一个对象的同一人群下出现「禁用」与「慎用」时，如实并列、不替用户取舍。"""
    units = [
        unit(unit_id="u-prohibited", object_name="孕妇", population="孕妇",
             severity=RestrictionSeverity.PROHIBITED),
        unit(unit_id="u-caution", object_name="妊娠", population="妊娠",
             severity=RestrictionSeverity.CAUTION),
    ]

    conflicts = severity_conflicts(units)

    assert set(conflicts) == {"u-prohibited", "u-caution"}
    note = conflicts["u-prohibited"]
    assert "禁用" in note and "慎用" in note
    assert "未做取舍" in note
    # 不得出现「按更严者」这类自行取舍的说法
    assert "更严" not in note


def test_different_subjects_are_not_a_conflict():
    """反向枚举一次返回多味药：千金子（禁用）与徐长卿（慎用）是两个不同对象
    各自的结论，不是同一对象下的程度冲突。分组键必须带主语。"""
    units = [
        unit(unit_id="u-qianjinzi", subject="千金子", population="孕妇",
             severity=RestrictionSeverity.PROHIBITED),
        unit(unit_id="u-xuchangqing", subject="徐长卿", population="孕妇",
             severity=RestrictionSeverity.CAUTION),
    ]

    assert severity_conflicts(units) == {}


def test_population_class_object_is_grouped_by_population_not_by_id():
    """实测川乌的两处来源抽成了两个不同的 object_id（一个 CONTRAINDICATION
    「孕妇慎用」、一个 INDICATION「妊娠」）。按 id 分组时它们永远比不到一起，
    「慎用」与「禁用」的分歧就不会被标出来。对象名属于人群类说法时，
    比较键必须落在人群维度上。"""
    units = [
        unit(unit_id="u-caution", object_id="CONTRAINDICATION:xxx",
             object_name="孕妇慎用", population="孕妇",
             severity=RestrictionSeverity.CAUTION),
        unit(unit_id="u-prohibited", object_id="INDICATION:yyy",
             object_name="妊娠", population="妊娠",
             severity=RestrictionSeverity.PROHIBITED),
    ]

    conflicts = severity_conflicts(units)

    assert set(conflicts) == {"u-caution", "u-prohibited"}
    assert "禁用" in conflicts["u-caution"] and "慎用" in conflicts["u-caution"]


def test_non_population_objects_still_compare_by_their_own_identity():
    """非人群类的对象仍按各自身份比较，不能因为上面那条就把不相干的对象并到一起。"""
    units = [
        unit(unit_id="u-a", object_id="CONTRAINDICATION:a", object_name="体虚者",
             population="孕妇", severity=RestrictionSeverity.PROHIBITED),
        unit(unit_id="u-b", object_id="CONTRAINDICATION:b", object_name="脾胃虚寒者",
             population="孕妇", severity=RestrictionSeverity.CAUTION),
    ]

    assert severity_conflicts(units) == {}


def test_different_populations_are_not_a_conflict():
    """测试：不同人群不冲突。"""
    units = [
        unit(unit_id="u-a", object_name="孕妇", population="孕妇",
             severity=RestrictionSeverity.PROHIBITED),
        unit(unit_id="u-b", object_name="小儿", population="小儿",
             severity=RestrictionSeverity.CAUTION),
    ]

    assert severity_conflicts(units) == {}


def test_unspecified_severity_never_counts_as_a_conflict():
    """测试：未标明的严重程度绝不算作作为冲突。"""
    units = [
        unit(unit_id="u-a", object_name="孕妇", population="孕妇",
             severity=RestrictionSeverity.PROHIBITED),
        unit(unit_id="u-b", object_name="孕妇", population="孕妇",
             severity=RestrictionSeverity.UNSPECIFIED),
    ]

    assert severity_conflicts(units) == {}


# ------------------------------------------------------------- 展示排序
def test_sort_by_severity_puts_prohibited_first_and_is_stable():
    """测试：排序按严重程度把禁用放在前面且为稳定。"""
    units = [
        unit(unit_id="a", severity=RestrictionSeverity.CAUTION),
        unit(unit_id="b", severity=RestrictionSeverity.PROHIBITED),
        unit(unit_id="c", severity=RestrictionSeverity.CAUTION),
    ]

    ordered = [item.unit_id for item in sort_by_severity(units)]

    # 重的在前；同程度保持原有相对顺序
    assert ordered == ["b", "a", "c"]
