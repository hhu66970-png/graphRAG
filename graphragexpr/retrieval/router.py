"""Rule-first query intent router (design section 8)."""

from __future__ import annotations

from collections.abc import Sequence

from ..config.settings import DEFAULT_RETRIEVAL_SETTINGS, RetrievalSettings
from ..domain.ontology import Ontology, load_ontology
from .models import RetrievalPlan

_COMPARE_TERMS = ("比较", "对比", "区别", "异同", "不同", "相同", "差异")
_INTERSECT_TERMS = ("同时", "兼有", "共同", "都能", "既", "又")
_PATH_TERMS = (
    "为何",
    "为什么",
    "如何",
    "通过什么",
    "关系",
    "关联",
    "路径",
    "治法链",
    "配伍",
)
_REVERSE_TERMS = (
    "哪些药",
    "哪些中药",
    "什么药",
    "哪味药",
    "有何药",
    "用什么药",
)
_INCOMPAT_HINTS = (
    "配伍禁忌",
    "十八反",
    "十九畏",
    "相反",
    "相畏",
    "相恶",
    "不能同用",
    "不宜同用",
    "不可同用",
    "不能同时",
    "不能搭配",
    "不宜与",
    "不可与",
)
_NEGATIVE_COMBO_TERMS = ("不能", "不宜", "不可", "禁止", "忌")
_COMBO_OBJECT_TERMS = ("同用", "合用", "配伍", "搭配", "同时")
_GENERIC_HERBS = frozenset({"中药", "中药材", "药材", "药"})
_HIGH_RISK_TERMS = (
    "剂量",
    "用量",
    "用法",
    "毒",
    "禁忌",
    "慎用",
    "忌用",
    "孕妇",
    "儿童",
    "不良反应",
    "配伍禁忌",
    "相反",
    "相畏",
    "相恶",
    "不能同时",
    "不能搭配",
    "不能同用",
    "不宜同用",
    "不可同用",
)

#: 公开别名：高风险服务层需要在没有 RetrievalPlan 的检索模式（如纯关键词图模式）
#: 下判断同一个问题是否属于高风险。共用同一份词表，避免两处判定标准漂移；
#: 这只是**策略开关**，六意图枚举不变（任务书 §1）。
HIGH_RISK_TERMS: tuple[str, ...] = _HIGH_RISK_TERMS

_PREDICATE_TERMS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("性味", "寒热", "温凉"), ("HAS_NATURE_FLAVOR",)),
    (("归经", "经脉"), ("ENTERS_MERIDIAN",)),
    (("功效", "作用"), ("HAS_EFFICACY",)),
    (("主治", "治疗", "病证", "症状"), ("TREATS",)),
    (("剂量", "用量", "用法", "服用", "中毒", "使用多少", "用多少"), ("HAS_DOSAGE",)),
    (("禁忌", "慎用", "忌用", "孕妇", "儿童", "不良反应"), ("CONTRAINDICATED_IN",)),
    (
        (
            "配伍禁忌",
            "相反",
            "相畏",
            "相恶",
            "不能同用",
            "不宜同用",
            "不可同用",
            "不能同时",
            "不能搭配",
        ),
        ("INCOMPATIBLE_WITH",),
    ),
    (("配伍", "同用"), ("COMBINED_WITH",)),
    (("来源", "基原"), ("DERIVED_FROM",)),
    (("炮制",), ("PROCESSED_AS",)),
    (("成分",), ("CONTAINS_COMPOUND",)),
    (("典籍", "记载"), ("RECORDED_IN",)),
    (("类别", "分类", "属于"), ("BELONGS_TO_CATEGORY",)),
)


def _is_incompatibility_question(question: str) -> bool:
    """问的是「不宜/不能同用」，不是「可以配伍」。"""
    if any(term in question for term in _INCOMPAT_HINTS):
        return True
    return any(term in question for term in _NEGATIVE_COMBO_TERMS) and any(
        term in question for term in _COMBO_OBJECT_TERMS
    )


def _mentioned_predicates(question: str, ontology: Ontology) -> tuple[str, ...]:
    """从问题词里识别提到的谓词。"""
    mentioned: list[str] = []
    for terms, predicates in _PREDICATE_TERMS:
        if any(term in question for term in terms):
            mentioned.extend(
                predicate
                for predicate in predicates
                if predicate in ontology.predicates
            )
    return tuple(dict.fromkeys(mentioned))


def route_query(
    question: str,
    linked_entities: Sequence[dict],
    *,
    ontology: Ontology | None = None,
    settings: RetrievalSettings = DEFAULT_RETRIEVAL_SETTINGS,
) -> RetrievalPlan:
    """Route a normalized question without making an LLM call.

    Comparison and safety rules are evaluated before entity-count heuristics. The returned
    reason/confidence are trace information, not calibrated probabilities.
    """
    normalized = "".join(question.split())
    if not normalized:
        raise ValueError("question must not be blank")

    schema = ontology or load_ontology()
    all_predicates = tuple(schema.predicates)
    mentioned = _mentioned_predicates(normalized, schema)
    entity_ids = tuple(
        str(entity["canonical_id"])
        for entity in linked_entities
        if entity.get("canonical_id")
    )
    herbs = [
        entity
        for entity in linked_entities
        if entity.get("entity_type") == "HERB"
        and (entity.get("name") or entity.get("preferred_name") or "")
        not in _GENERIC_HERBS
    ]
    constraints = [
        entity
        for entity in linked_entities
        if entity.get("entity_type") in {"EFFICACY", "INDICATION"}
    ]
    is_high_risk = any(term in normalized for term in _HIGH_RISK_TERMS)
    require_provenance = is_high_risk or bool(linked_entities)

    if _is_incompatibility_question(normalized) and herbs:
        herb_ids = tuple(str(item["canonical_id"]) for item in herbs[:4])
        multi = len(herb_ids) >= 2
        return RetrievalPlan(
            "compare" if multi else "attr",
            herb_ids,
            ("INCOMPATIBLE_WITH",),
            1,
            True,
            "summary" if multi else "card",
            "matched incompatibility wording for one or more herbs",
            0.96,
            True,
        )

    if any(term in normalized for term in _COMPARE_TERMS):
        return RetrievalPlan(
            "compare",
            entity_ids,
            mentioned or all_predicates,
            1,
            require_provenance,
            "summary",
            "matched explicit comparison wording",
            0.98,
            is_high_risk,
        )

    has_reverse_word = any(term in normalized for term in _REVERSE_TERMS)
    if len(constraints) >= 2 and (
        any(term in normalized for term in _INTERSECT_TERMS) or has_reverse_word
    ):
        return RetrievalPlan(
            "intersect",
            tuple(str(item["canonical_id"]) for item in constraints),
            ("HAS_EFFICACY", "TREATS"),
            1,
            True,
            "matrix",
            "matched multiple constraints and intersection wording",
            0.97,
            is_high_risk,
        )

    has_path_word = any(term in normalized for term in _PATH_TERMS)
    if len(linked_entities) >= 2 and has_path_word:
        return RetrievalPlan(
            "path",
            entity_ids,
            all_predicates,
            max(1, min(settings.path_max_hops, 3)),
            True,
            "paths",
            "matched multi-hop wording with two or more linked entities",
            0.93,
            is_high_risk,
        )

    if constraints and (has_reverse_word or (not herbs and len(constraints) == 1)):
        return RetrievalPlan(
            "reverse",
            (str(constraints[0]["canonical_id"]),),
            ("HAS_EFFICACY", "TREATS"),
            1,
            True,
            "table",
            "matched reverse wording or linked a constraint without a herb",
            0.92,
            is_high_risk,
        )

    if len(herbs) >= 2:
        herb_ids = tuple(str(item["canonical_id"]) for item in herbs[:4])
        return RetrievalPlan(
            "compare",
            herb_ids,
            mentioned or all_predicates,
            1,
            require_provenance,
            "summary",
            "linked multiple herbs; retrieve each instead of the first only",
            0.91,
            is_high_risk,
        )

    if herbs:
        herb_id = str(herbs[0]["canonical_id"])
        return RetrievalPlan(
            "attr",
            (herb_id,),
            mentioned or all_predicates,
            1,
            require_provenance,
            "card",
            "linked a herb for direct typed attributes",
            0.90,
            is_high_risk,
        )

    return RetrievalPlan(
        "fallback",
        entity_ids,
        (),
        0,
        is_high_risk,
        "chunks",
        "no directional graph intent could be resolved",
        0.55,
        is_high_risk,
    )
