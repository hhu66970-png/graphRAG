"""测试检索服务各意图的证据包与子图。"""

from graphragexpr.domain.ontology import load_ontology
from graphragexpr.retrieval import service


def test_fallback_returns_chunk_evidence_packet(monkeypatch):
    """测试：回退返回文本块证据证据包。"""
    monkeypatch.setattr(service.serving_read, "active_run", lambda store: "run-1")
    monkeypatch.setattr(
        service.serving_read,
        "link_entities_in_text",
        lambda store, text, **kwargs: [],
    )

    def fallback(question):
        """测试用：假的兜底检索。"""
        return (
            [
                {
                    "id": "c1",
                    "text": "中药学总论原文",
                    "page_start": 12,
                    "page_end": 12,
                    "score": 0.8,
                    "matched_terms": ["中药学"],
                },
                {
                    "id": "c2",
                    "text": "第二段中药学总论原文",
                    "page_start": 13,
                    "page_end": 13,
                    "score": 0.7,
                    "matched_terms": ["中药学"],
                },
            ],
            {"nodes": [], "edges": []},
            [],
        )

    result = service.retrieve(
        object(),
        load_ontology(),
        "中药学的发展历史是什么？",
        fallback_retriever=fallback,
        token_budget=50,
    )

    assert result.packet.intent == "fallback"
    assert result.packet.structure_type == "chunks"
    assert result.packet.items[0].chunk_id == "c1"
    assert len(result.packet.items) == 1
    assert "token 预算裁剪" in result.packet.warnings[0]
    assert result.packet.trace["items_before_budget"] == 2
    assert result.packet.trace["route"]["reason"]


def test_path_steps_preserve_relationship_direction():
    """测试：路径步骤保留关系方向。"""
    path = {
        "nodes": [
            {"id": "HERB:a", "name": "甲", "type": "HERB"},
            {"id": "HERB:b", "name": "乙", "type": "HERB"},
        ],
        "predicates": ["COMBINED_WITH"],
        "edge_source_ids": ["HERB:b"],
        "edge_target_ids": ["HERB:a"],
    }

    steps = service._path_steps(path)

    assert steps[0]["source"]["id"] == "HERB:b"
    assert steps[0]["target"]["id"] == "HERB:a"


def test_intersection_subgraph_uses_real_serving_predicate():
    """测试：交集子图使用真实Serving谓词。"""
    rows = [
        {
            "herb_id": "HERB:a",
            "herb": "甲药",
            "matches": [
                {
                    "target_id": "EFFICACY:x",
                    "target_name": "清热",
                    "target_type": "EFFICACY",
                    "predicate": "HAS_EFFICACY",
                }
            ],
        }
    ]

    subgraph = service._constraint_rows_to_subgraph(rows, ["EFFICACY:x"])

    assert subgraph["edges"] == [
        {
            "source": "HERB:a",
            "target": "EFFICACY:x",
            "type": "HAS_EFFICACY",
        }
    ]


def test_incompatibility_compare_reads_assertions_not_serving_edges(monkeypatch):
    """测试：配伍禁忌对比读取断言不Serving边。"""
    linked = [
        {"canonical_id": "HERB:renshen", "name": "人参", "entity_type": "HERB"},
        {"canonical_id": "HERB:mahuang", "name": "麻黄", "entity_type": "HERB"},
    ]
    serving_calls = []
    monkeypatch.setattr(service.serving_read, "active_run", lambda store: "run-1")
    monkeypatch.setattr(
        service.serving_read,
        "link_entities_in_text",
        lambda store, text, **kwargs: linked,
    )
    monkeypatch.setattr(
        service.serving_read,
        "load_entity_facts",
        lambda *args, **kwargs: serving_calls.append("facts") or [],
    )
    monkeypatch.setattr(
        service.serving_read,
        "get_herb_card",
        lambda *args, **kwargs: serving_calls.append("card") or None,
    )

    def fake_load(store, *, run_id, subject_ids, predicates, limit=60):
        """测试用：假的断言加载。"""
        assert run_id == "run-1"
        assert set(subject_ids) == {"HERB:renshen", "HERB:mahuang"}
        assert predicates == ("INCOMPATIBLE_WITH",)
        return [
            {
                "assertion_id": "a-rs",
                "fact_key": "fk-rs",
                "predicate": "INCOMPATIBLE_WITH",
                "subject_id": "HERB:renshen",
                "subject_name": "人参",
                "object_id": "HERB:lilu",
                "object_name": "藜芦",
                "evidence_quote": "人参反藜芦。",
                "chunk_ids": ["c1"],
                "page_start": 379,
                "page_end": 379,
                "final_score": 0.9,
            }
        ]

    monkeypatch.setattr(service, "load_high_risk_assertions", fake_load)

    result = service.retrieve(
        object(),
        load_ontology(),
        "人参和麻黄均不能同时和什么药搭配使用",
    )

    assert serving_calls == []
    assert result.packet.intent == "compare"
    assert [item.predicate for item in result.packet.items] == ["INCOMPATIBLE_WITH"]
    assert result.packet.items[0].qualifiers["subject_name"] == "人参"
    assert result.packet.items[0].qualifiers["object_name"] == "藜芦"
    warnings = "；".join(result.packet.warnings)
    assert "麻黄" in warnings
    assert "共同" in warnings
    assert result.packet.trace["incompatibility_by_herb"]["人参"] == ["藜芦"]
    assert result.packet.trace["incompatibility_by_herb"]["麻黄"] == []
    assert result.packet.trace["shared_incompatible_objects"] == []


def test_attr_dosage_reads_assertions_when_serving_card_empty(monkeypatch):
    """测试：属性意图剂量读取断言当Serving卡片空。"""
    linked = [{"canonical_id": "HERB:fuzi", "name": "附子", "entity_type": "HERB"}]
    monkeypatch.setattr(service.serving_read, "active_run", lambda store: "run-1")
    monkeypatch.setattr(
        service.serving_read,
        "link_entities_in_text",
        lambda store, text, **kwargs: linked,
    )
    monkeypatch.setattr(service.serving_read, "get_herb_card", lambda *args, **kwargs: None)

    def fake_load(store, *, run_id, subject_ids, predicates, limit=60):
        """测试用：假的断言加载。"""
        assert "HAS_DOSAGE" in predicates
        assert subject_ids == ["HERB:fuzi"]
        return [
            {
                "assertion_id": "a-dose",
                "fact_key": "fk-dose",
                "predicate": "HAS_DOSAGE",
                "subject_id": "HERB:fuzi",
                "subject_name": "附子",
                "object_id": None,
                "object_name": None,
                "evidence_quote": "煎服，3～15g。",
                "chunk_ids": ["c1"],
                "page_start": 236,
                "page_end": 236,
                "final_score": 0.9,
            }
        ]

    monkeypatch.setattr(service, "load_high_risk_assertions", fake_load)

    result = service.retrieve(object(), load_ontology(), "附子的用量是多少？")

    assert result.packet.intent == "attr"
    assert result.packet.items[0].predicate == "HAS_DOSAGE"
    assert "3～15g" in result.packet.items[0].evidence_quote


def test_path_reads_assertion_edges_when_serving_has_none(monkeypatch):
    """测试：路径读取断言边当Serving具有没有边。"""
    linked = [
        {"canonical_id": "HERB:gancao", "name": "甘草", "entity_type": "HERB"},
        {"canonical_id": "HERB:yuanhua", "name": "芫花", "entity_type": "HERB"},
    ]
    monkeypatch.setattr(service.serving_read, "active_run", lambda store: "run-1")
    monkeypatch.setattr(
        service.serving_read,
        "link_entities_in_text",
        lambda store, text, **kwargs: linked,
    )
    monkeypatch.setattr(
        service.serving_read, "find_supported_paths", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        service,
        "load_assertion_edges_between",
        lambda store, **kwargs: [
            {
                "assertion_id": "a-fan",
                "predicate": "INCOMPATIBLE_WITH",
                "subject_id": "HERB:gancao",
                "subject_name": "甘草",
                "object_id": "HERB:yuanhua",
                "object_name": "芫花",
                "object_type": "HERB",
                "evidence_quote": "甘草反芫花。",
                "final_score": 0.88,
                "chunk_ids": ["c9"],
                "page_start": 64,
                "page_end": 64,
            }
        ],
    )
    monkeypatch.setattr(
        service.serving_read,
        "load_evidence",
        lambda store, ids: [
            {
                "assertion_id": "a-fan",
                "quote": "甘草反芫花。",
                "predicate": "INCOMPATIBLE_WITH",
                "subject_id": "HERB:gancao",
                "subject_name": "甘草",
                "object_id": "HERB:yuanhua",
                "object_name": "芫花",
                "chunk_ids": ["c9"],
                "page_start": 64,
                "page_end": 64,
                "support_score": 0.88,
            }
        ],
    )

    result = service.retrieve(object(), load_ontology(), "甘草与芫花如何关联？")

    assert result.packet.intent == "path"
    assert result.packet.items[0].predicate == "INCOMPATIBLE_WITH"
    assert result.packet.trace["path_count"] == 1
    assert result.subgraph["edges"][0]["type"] == "INCOMPATIBLE_WITH"
