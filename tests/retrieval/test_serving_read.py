from __future__ import annotations

from graphragexpr.storage import serving_read


class FakeStore:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, **params):
        self.calls.append((query, params))
        return self.rows


def test_intersection_groups_exact_and_partial_matches():
    store = FakeStore(
        [
            {
                "herb": "金银花",
                "herb_id": "HERB:jinyinhua",
                "matched_ids": ["EFFICACY:qingre", "EFFICACY:liangxue"],
                "matched_names": ["清热解毒", "凉血"],
                "matches": [
                    {
                        "target_id": "EFFICACY:qingre",
                        "target_name": "清热解毒",
                        "target_type": "EFFICACY",
                        "predicate": "HAS_EFFICACY",
                    },
                    {
                        "target_id": "EFFICACY:liangxue",
                        "target_name": "凉血",
                        "target_type": "EFFICACY",
                        "predicate": "HAS_EFFICACY",
                    },
                ],
                "evidence_ids": ["a1", "a2"],
                "pages": [201, 202],
                "best_score": 0.98,
            },
            {
                "herb": "连翘",
                "herb_id": "HERB:lianqiao",
                "matched_ids": ["EFFICACY:qingre"],
                "matched_names": ["清热解毒"],
                "matches": [
                    {
                        "target_id": "EFFICACY:qingre",
                        "target_name": "清热解毒",
                        "target_type": "EFFICACY",
                        "predicate": "HAS_EFFICACY",
                    }
                ],
                "evidence_ids": ["a3"],
                "pages": [205],
                "best_score": 0.92,
            },
        ]
    )

    rows = serving_read.find_herbs_by_constraints(
        store,
        ["EFFICACY:qingre", "EFFICACY:liangxue"],
        run_id="run-1",
    )

    assert rows[0]["exact_match"] is True
    assert rows[0]["coverage"] == 1.0
    assert rows[1]["exact_match"] is False
    assert rows[1]["coverage"] == 0.5
    assert store.calls[0][1]["constraint_count"] == 2


def test_intersection_uses_compound_efficacy_names_when_ids_miss():
    class SwitchingStore:
        def __init__(self):
            self.calls: list[dict] = []

        def run(self, query, **params):
            self.calls.append(params)
            if params.get("names"):
                return [
                    {
                        "herb": "黄芪",
                        "herb_id": "HERB:huangqi",
                        "matches": [
                            {
                                "target_id": "EFFICACY:buqilishui",
                                "target_name": "补气利水",
                                "target_type": "EFFICACY",
                                "predicate": "HAS_EFFICACY",
                            }
                        ],
                        "matched_ids": ["EFFICACY:buqilishui"],
                        "matched_names": ["补气利水"],
                        "evidence_ids": ["a1"],
                        "pages": [289],
                        "best_score": 0.9,
                    }
                ]
            return [
                {
                    "herb": "黄芪",
                    "herb_id": "HERB:huangqi",
                    "matches": [
                        {
                            "target_id": "EFFICACY:buqi",
                            "target_name": "补气",
                            "target_type": "EFFICACY",
                            "predicate": "HAS_EFFICACY",
                        }
                    ],
                    "matched_ids": ["EFFICACY:buqi"],
                    "matched_names": ["补气"],
                    "evidence_ids": ["a0"],
                    "pages": [289],
                    "best_score": 0.8,
                }
            ]

    store = SwitchingStore()
    rows = serving_read.find_herbs_by_constraints(
        store,
        ["EFFICACY:buqi", "EFFICACY:lishui"],
        run_id="run-1",
        constraint_names=["补气", "利水"],
    )

    assert rows[0]["exact_match"] is True
    assert set(rows[0]["matched_ids"]) == {"EFFICACY:buqi", "EFFICACY:lishui"}
    assert store.calls[1]["names"] == ["补气", "利水"]


def test_path_query_caps_hops_and_requires_serving_provenance():
    store = FakeStore([])

    serving_read.find_supported_paths(
        store,
        "HERB:renshen",
        "INDICATION:qixu",
        run_id="run-1",
        max_hops=99,
    )

    query, params = store.calls[0]
    assert "*1..3" in query
    assert "fact_key IS NOT NULL" in query
    assert "active_run_id = $run" in query
    assert params["allowed_predicates"]
    assert "FROM_CHUNK" not in params["allowed_predicates"]


def test_path_query_rejects_same_endpoint():
    store = FakeStore([])

    try:
        serving_read.find_supported_paths(store, "HERB:x", "HERB:x", run_id="run-1")
    except ValueError as exc:
        assert "different" in str(exc)
    else:
        raise AssertionError("same endpoints must be rejected")


def test_load_evidence_uses_assertion_relationships_and_decodes_qualifiers():
    store = FakeStore(
        [
            {
                "assertion_id": "a1",
                "subject_id": "HERB:renshen",
                "object_id": "EFFICACY:buqi",
                "qualifiers_json": '{"condition": "气虚"}',
            }
        ]
    )

    rows = serving_read.load_evidence(store, ["a1"])

    query, params = store.calls[0]
    assert "(a)-[:SUBJECT]->(subject:Entity)" in query
    assert "(a)-[:OBJECT]->(object:Entity)" in query
    assert "(a)-[:EVIDENCED_BY]->(chunk:Chunk)" in query
    assert "a.subject_id" not in query
    assert params == {"ids": ["a1"]}
    assert rows[0]["qualifiers"] == {"condition": "气虚"}


def test_load_entity_sections_uses_assertion_provenance_relationships():
    store = FakeStore([])

    serving_read.load_entity_sections(
        store, ["HERB:renshen"], run_id="run-1"
    )

    query, _params = store.calls[0]
    assert "[:SUBJECT]->(entity)" in query
    assert "[:EVIDENCED_BY]->(chunk:Chunk)" in query
    assert "assertion.subject_id" not in query
