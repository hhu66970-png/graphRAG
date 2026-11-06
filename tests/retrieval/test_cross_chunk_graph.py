"""测试问答子图跨块扩展与关系遍历。"""

from graphragexpr.retrieval.cross_chunk_graph import (
    AUDIT_REL_TYPES,
    expand_cross_chunk_neighborhood,
    merge_extra_chunks,
)


def test_merge_extra_chunks_keeps_hits_and_adds_other_pages():
    ranked = [
        {"id": "c1", "text": "人参性味", "page_start": 378, "matched_terms": ["人参"]},
        {"id": "c2", "text": "相邻块", "page_start": 379, "matched_terms": ["相邻文本块"]},
    ]
    extra = [
        {"id": "c1", "text": "重复命中", "page_start": 378},
        {"id": "c3", "text": "人参功效在另一页", "page_start": 380},
        {"id": "c4", "text": "人参归经", "page_start": 68},
    ]

    merged = merge_extra_chunks(ranked, extra, limit=4)

    assert [chunk["id"] for chunk in merged] == ["c1", "c2", "c3", "c4"]
    assert merged[0]["matched_terms"] == ["人参"]
    assert "跨块图谱" in merged[2]["matched_terms"]
    assert merged[2]["page_start"] == 380


def test_merge_extra_chunks_respects_context_limit():
    ranked = [{"id": "c1", "text": "a"}]
    extra = [{"id": f"x{i}", "text": str(i)} for i in range(10)]

    merged = merge_extra_chunks(ranked, extra, limit=3)

    assert len(merged) == 3
    assert merged[0]["id"] == "c1"


def test_expand_prefers_serving_edges_and_other_chunk_text():
    class FakeResult(list):
        def single(self):
            """测试替身：返回唯一记录。"""
            return self[0] if self else None

    class FakeSession:
        def run(self, cypher, **params):
            """测试替身：按 Cypher 参数返回预制的 FakeResult。"""
            if params.get("predicates") is not None:
                return FakeResult([])
            if params.get("names") is not None:
                assert "人参" in params["names"]
                assert params["run"] == "run-1"
                return FakeResult(
                    [
                        {
                            "source_id": "HERB:renshen",
                            "source_name": "人参",
                            "source_type": "HERB",
                            "rel_type": "HAS_EFFICACY",
                            "target_id": "EFFICACY:buqi",
                            "target_name": "大补元气",
                            "target_type": "EFFICACY",
                            "evidence_ids": ["a-1"],
                        }
                    ]
                )
            if params.get("assertion_ids") is not None:
                assert "a-1" in params["assertion_ids"]
                assert "hit-1" in params["known_ids"]
                return FakeResult(
                    [
                        {
                            "id": "chunk-other",
                            "index": 44,
                            "text": "大补元气，复脉固脱",
                            "page_start": 378,
                            "page_end": 378,
                            "section": "人参",
                        }
                    ]
                )
            if params.get("chunk_ids") == ["hit-1"]:
                return FakeResult(
                    [
                        {
                            "entity_id": "old-1",
                            "entity_label": "中药",
                            "entity_name": "人参",
                        }
                    ]
                )
            raise AssertionError(cypher)

    nodes, edges, triples, extra = expand_cross_chunk_neighborhood(
        FakeSession(),
        ["hit-1"],
        run_id="run-1",
        node_limit=10,
        edge_limit=10,
        triple_limit=10,
        extra_chunk_limit=4,
    )

    assert triples == [("人参", "HAS_EFFICACY", "大补元气")]
    assert edges == [
        {"source": "HERB:renshen", "target": "EFFICACY:buqi", "type": "HAS_EFFICACY"}
    ]
    assert extra[0]["id"] == "chunk-other"
    assert extra[0]["page_start"] == 378
    assert {node["id"] for node in nodes} == {"HERB:renshen", "EFFICACY:buqi"}


def test_expand_falls_back_to_legacy_cross_chunk_when_no_serving_graph():
    class FakeSession:
        def run(self, cypher, **params):
            """测试替身：按 Cypher 参数返回预制的 FakeResult。"""
            if params.get("audit"):
                assert set(params["audit"]) == set(AUDIT_REL_TYPES)
                return [
                    {
                        "source_id": "old-1",
                        "source_label": "中药",
                        "source_name": "人参",
                        "target_id": "old-2",
                        "target_label": "功效",
                        "target_name": "大补元气",
                        "rel_type": "具有功效",
                    }
                ]
            if params.get("entity_ids"):
                return [
                    {
                        "id": "chunk-other",
                        "index": 9,
                        "text": "另一块里的功效",
                        "page_start": 380,
                        "page_end": 380,
                        "section": None,
                    }
                ]
            if params.get("chunk_ids") == ["hit-1"]:
                return [
                    {
                        "entity_id": "old-1",
                        "entity_label": "中药",
                        "entity_name": "人参",
                    }
                ]
            raise AssertionError(cypher)

    _nodes, _edges, triples, extra = expand_cross_chunk_neighborhood(
        FakeSession(),
        ["hit-1"],
        run_id=None,
        node_limit=10,
        edge_limit=10,
        triple_limit=10,
        extra_chunk_limit=4,
    )

    assert triples == [("人参", "具有功效", "大补元气")]
    assert extra[0]["page_start"] == 380


def test_expand_also_follows_incompatible_assertions():
    class FakeResult(list):
        def single(self):
            """测试替身：返回唯一记录。"""
            return self[0] if self else None

    class FakeSession:
        def run(self, cypher, **params):
            """测试替身：按 Cypher 参数返回预制的 FakeResult。"""
            if params.get("predicates") is not None:
                assert "INCOMPATIBLE_WITH" in params["predicates"]
                return FakeResult(
                    [
                        {
                            "source_id": "HERB:renshen",
                            "source_name": "人参",
                            "source_type": "HERB",
                            "rel_type": "INCOMPATIBLE_WITH",
                            "target_id": "HERB:lilu",
                            "target_name": "藜芦",
                            "target_type": "HERB",
                            "assertion_id": "a-incompat",
                        }
                    ]
                )
            if params.get("names") is not None:
                return FakeResult([])
            if params.get("assertion_ids") is not None:
                assert "a-incompat" in params["assertion_ids"]
                return FakeResult(
                    [
                        {
                            "id": "chunk-incompat",
                            "index": 12,
                            "text": "人参反藜芦。",
                            "page_start": 65,
                            "page_end": 65,
                            "section": "十八反",
                        }
                    ]
                )
            if params.get("chunk_ids") == ["hit-1"]:
                return FakeResult(
                    [
                        {
                            "entity_id": "old-1",
                            "entity_label": "中药",
                            "entity_name": "人参",
                        }
                    ]
                )
            raise AssertionError(cypher)

    nodes, edges, triples, extra = expand_cross_chunk_neighborhood(
        FakeSession(),
        ["hit-1"],
        run_id="run-1",
        node_limit=10,
        edge_limit=10,
        triple_limit=10,
        extra_chunk_limit=4,
    )

    assert ("人参", "INCOMPATIBLE_WITH", "藜芦") in triples
    assert extra[0]["id"] == "chunk-incompat"
    assert extra[0]["page_start"] == 65
    assert {node["id"] for node in nodes} >= {"HERB:renshen", "HERB:lilu"}


def test_audit_relationship_types_are_not_traversed():
    assert "FROM_CHUNK" in AUDIT_REL_TYPES
    assert "EVIDENCED_BY" in AUDIT_REL_TYPES
    assert "HAS_EFFICACY" not in AUDIT_REL_TYPES
