"""测试规则意图路由覆盖各题型。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphragexpr.domain.models import EvidenceItem, EvidencePacket
from graphragexpr.retrieval.router import route_query


def _entity(name: str, entity_type: str, suffix: str) -> dict:
    """测试用：构造已链接实体。"""
    return {
        "canonical_id": f"{entity_type}:{suffix}",
        "name": name,
        "entity_type": entity_type,
    }


@pytest.mark.parametrize(
    ("question", "entities", "intent", "structure_type"),
    [
        (
            "人参的性味和功效是什么？",
            [_entity("人参", "HERB", "renshen")],
            "attr",
            "card",
        ),
        (
            "哪些中药具有清热解毒功效？",
            [_entity("清热解毒", "EFFICACY", "qingre")],
            "reverse",
            "table",
        ),
        (
            "哪些药同时具有清热解毒和凉血功效？",
            [
                _entity("清热解毒", "EFFICACY", "qingre"),
                _entity("凉血", "EFFICACY", "liangxue"),
            ],
            "intersect",
            "matrix",
        ),
        (
            "比较人参和黄芪的功效异同",
            [
                _entity("人参", "HERB", "renshen"),
                _entity("黄芪", "HERB", "huangqi"),
            ],
            "compare",
            "summary",
        ),
        (
            "人参与附子配伍后如何发挥作用？",
            [
                _entity("人参", "HERB", "renshen"),
                _entity("附子", "HERB", "fuzi"),
            ],
            "path",
            "paths",
        ),
        ("中药学的发展历史是什么？", [], "fallback", "chunks"),
    ],
)
def test_route_query_covers_all_intents(question, entities, intent, structure_type):
    plan = route_query(question, entities)

    assert plan.intent == intent
    assert plan.structure_type == structure_type
    assert plan.reason
    assert 0.0 <= plan.confidence <= 1.0


def test_compare_rule_precedes_two_entity_path_rule():
    entities = [
        _entity("人参", "HERB", "renshen"),
        _entity("黄芪", "HERB", "huangqi"),
    ]

    plan = route_query("人参和黄芪有什么区别？", entities)

    assert plan.intent == "compare"
    assert plan.max_hops == 1


def test_reverse_wording_precedes_a_generic_herb_link():
    entities = [
        _entity("中药", "HERB", "generic-herb"),
        _entity("大补元气", "EFFICACY", "dabu-yuanqi"),
    ]

    plan = route_query("哪些中药具有大补元气的功效？", entities)

    assert plan.intent == "reverse"
    assert plan.linked_entity_ids == ("EFFICACY:dabu-yuanqi",)


def test_high_risk_words_force_provenance_without_inventing_safety_intent():
    plan = route_query(
        "附子的用量和禁忌是什么？",
        [_entity("附子", "HERB", "fuzi")],
    )

    assert plan.intent == "attr"
    assert plan.require_provenance is True
    assert "HAS_DOSAGE" in plan.allowed_predicates
    assert "CONTRAINDICATED_IN" in plan.allowed_predicates
    assert plan.is_high_risk is True


def test_path_is_bounded_to_three_hops():
    plan = route_query(
        "麻黄通过什么关系影响咳嗽？",
        [
            _entity("麻黄", "HERB", "mahuang"),
            _entity("咳嗽", "INDICATION", "kesou"),
        ],
    )

    assert plan.intent == "path"
    assert plan.max_hops == 3
    assert "FROM_CHUNK" not in plan.allowed_predicates


def test_two_herbs_without_compare_words_still_retrieve_both():
    plan = route_query(
        "人参和黄芪的性味功效是什么？",
        [
            _entity("人参", "HERB", "renshen"),
            _entity("黄芪", "HERB", "huangqi"),
        ],
    )

    assert plan.intent == "compare"
    assert plan.linked_entity_ids == ("HERB:renshen", "HERB:huangqi")
    assert "HAS_NATURE_FLAVOR" in plan.allowed_predicates
    assert "HAS_EFFICACY" in plan.allowed_predicates


def test_reverse_with_two_constraints_becomes_intersection():
    plan = route_query(
        "哪些药具有清热解毒和凉血功效？",
        [
            _entity("清热解毒", "EFFICACY", "qingre"),
            _entity("凉血", "EFFICACY", "liangxue"),
        ],
    )

    assert plan.intent == "intersect"
    assert plan.linked_entity_ids == ("EFFICACY:qingre", "EFFICACY:liangxue")


def test_shared_incompatibility_routes_both_herbs():
    plan = route_query(
        "人参和麻黄均不能同时和什么药搭配使用",
        [
            _entity("人参", "HERB", "renshen"),
            _entity("麻黄", "HERB", "mahuang"),
        ],
    )

    assert plan.intent == "compare"
    assert plan.linked_entity_ids == ("HERB:renshen", "HERB:mahuang")
    assert plan.allowed_predicates == ("INCOMPATIBLE_WITH",)
    assert plan.is_high_risk is True


def test_single_herb_incompatibility_stays_attr():
    plan = route_query(
        "人参不宜与什么药同用？",
        [_entity("人参", "HERB", "renshen")],
    )

    assert plan.intent == "attr"
    assert plan.linked_entity_ids == ("HERB:renshen",)
    assert plan.allowed_predicates == ("INCOMPATIBLE_WITH",)


def test_blank_question_is_rejected():
    with pytest.raises(ValueError, match="question"):
        route_query("  ", [])


def test_evidence_packet_contract_is_strict():
    item = EvidenceItem(
        evidence_id="a1",
        evidence_quote="原文",
        chunk_id="c1",
        page_start=10,
        page_end=10,
    )
    packet = EvidencePacket(
        query_id="q1",
        intent="attr",
        structure_type="card",
        items=[item],
        token_budget=100,
    )

    assert packet.items[0].evidence_id == "a1"
    with pytest.raises(ValidationError):
        EvidencePacket(
            query_id="q1",
            intent="unknown",
            structure_type="card",
            token_budget=100,
        )
