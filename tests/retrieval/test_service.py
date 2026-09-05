from graphragexpr.domain.ontology import load_ontology
from graphragexpr.retrieval import service


def test_fallback_returns_chunk_evidence_packet(monkeypatch):
    monkeypatch.setattr(service.serving_read, "active_run", lambda store: "run-1")
    monkeypatch.setattr(
        service.serving_read,
        "link_entities_in_text",
        lambda store, text, **kwargs: [],
    )

    def fallback(question):
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
