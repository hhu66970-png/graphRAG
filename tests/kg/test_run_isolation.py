"""Version identity and transaction failure regressions without a database."""

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from graphragexpr.domain.errors import RunStateError
from graphragexpr.domain.ids import assertion_version_id
from graphragexpr.domain.models import AssertionRecord
from graphragexpr.domain.ontology import load_ontology
from graphragexpr.storage import assertions, runs, serving_projection


class TransactionStore:
    """Record statements submitted through a managed write transaction."""

    def __init__(self, states):
        self.states = states
        self.rows = {}
        self.calls = []
        self.tx = MagicMock()
        self.tx.run.side_effect = self.run_tx
        self.sessions = 0

    def run(self, query, **params):
        # Schema creation is the only operation allowed outside the write transaction.
        assert query.startswith("CREATE CONSTRAINT")
        return []

    def run_tx(self, query, **params):
        self.calls.append((query, params))
        result = MagicMock()
        if "RETURN r.status AS status" in query:
            state = self.states.get(params["run_id"])
            result.single.return_value = {"status": state} if state else None
        elif "source_assertion_id" in query:
            for row in params["batch"]:
                self.rows[row["assertion_id"]] = row.copy()
        return result

    @contextmanager
    def session(self):
        self.sessions += 1
        owner = self

        class Session:
            def execute_write(self, callback):
                return callback(owner.tx)

        yield Session()


def _assertion(run, score=0.9):
    return AssertionRecord(
        assertion_id="logical-evidence", fact_key="same-fact", run_id=run,
        subject_id="HERB:example", predicate="HAS_EFFICACY", object_id="EFFICACY:example",
        chunk_ids=["same-chunk"], evidence_quote="发汗解表", evidence_start=0, evidence_end=4,
        match_mode="raw_exact", page_start=1, page_end=1, final_score=score,
        rule_score=score, validation_status="ACCEPTED", schema_version="v1",
    )


def test_repeated_evidence_is_idempotent_within_a_run_and_distinct_between_runs():
    store = TransactionStore({"first": "BUILDING", "second": "BUILDING"})
    assertions.upsert_assertions(store, [_assertion("first")], batch_size=1)
    old = dict(store.rows)
    assertions.upsert_assertions(store, [_assertion("second", score=0.8)], batch_size=1)
    assertions.upsert_assertions(store, [_assertion("second", score=0.8)], batch_size=1)
    assert len(store.rows) == 2
    assert store.rows[assertion_version_id("logical-evidence", "first")] == next(iter(old.values()))
    assert {row["source_assertion_id"] for row in store.rows.values()} == {"logical-evidence"}
    assert {row["run_id"] for row in store.rows.values()} == {"first", "second"}


@pytest.mark.parametrize("state", ["ACTIVE", "SUPERSEDED", "FAILED", None])
def test_assertion_write_refuses_finished_or_missing_build_before_writing(state):
    store = TransactionStore({"target": state})
    with pytest.raises(RunStateError):
        assertions.upsert_assertions(store, [_assertion("target")], batch_size=10)
    assert store.rows == {}
    assert not any("MERGE (a:Assertion" in query for query, _ in store.calls)


@pytest.mark.parametrize("transition,state", [
    ("activate", "ACTIVE"), ("activate", "FAILED"),
    ("rollback", "BUILDING"), ("failed", "ACTIVE"),
])
def test_invalid_transition_does_not_demote_a_serving_version(transition, state):
    store = TransactionStore({"target": state})
    with pytest.raises(RunStateError):
        if transition == "activate":
            runs.activate(store, "target", quality_report={})
        elif transition == "rollback":
            runs.rollback_to(store, "target")
        else:
            runs.mark_failed(store, "target", "failure")
    assert not any("SET old.status" in query for query, _ in store.calls)
    assert not any("SET target.status" in query for query, _ in store.calls)


def test_publication_checks_and_state_changes_share_one_transaction():
    store = TransactionStore({"target": "BUILDING"})
    runs.activate(store, "target", quality_report={"evidence_coverage": 1.0})
    assert store.sessions == 1
    queries = [query for query, _ in store.calls]
    assert "GraphPublication" in queries[0]
    assert "RETURN r.status" in queries[1]
    assert "SET old.status" in queries[2]
    assert "SET target.status" in queries[3]


def test_projection_refuses_published_run_before_deleting_edges():
    store = TransactionStore({"target": "ACTIVE"})
    with pytest.raises(RunStateError):
        serving_projection.rebuild_projection(store, load_ontology(), run_id="target", batch_size=10)
    assert not any("DELETE e" in query for query, _ in store.calls)


def test_storage_integration_fixture_does_not_fall_back_to_business_database(monkeypatch):
    from tests.kg import test_storage_integration as integration

    monkeypatch.setenv("ALLOW_DESTRUCTIVE_TESTS", "true")
    for key in ("URL", "USER", "PASSWORD", "DB"):
        monkeypatch.setenv("NEO4J_" + key, "configured-business-value")
        monkeypatch.delenv("NEO4J_TEST_" + key, raising=False)
    constructor = MagicMock(side_effect=AssertionError("must not create a business connection"))
    monkeypatch.setattr(integration, "Neo4jStore", constructor)
    with pytest.raises(pytest.skip.Exception, match="独立 NEO4J_TEST"):
        next(integration.store.__wrapped__())
    constructor.assert_not_called()
