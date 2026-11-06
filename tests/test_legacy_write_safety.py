"""Prevent the baseline builder from replacing evidence used by versioned runs."""

from types import SimpleNamespace
from unittest.mock import MagicMock
import os

import pytest

from graphragexpr.extract import build_kg_dyn as legacy


@pytest.fixture
def record():
    return legacy.ChunkRecord(0, "新版原文", 2, 2, "section", "source.md")


def fake_driver(protected):
    session = MagicMock()
    session.__enter__.return_value = session
    session.run.return_value.single.return_value = {"protected": protected}
    driver = MagicMock()
    driver.session.return_value = session
    return driver, session


@pytest.mark.parametrize("setting", [None, "false", "0", ""])
def test_versioned_database_is_rejected_before_embedding_or_writing(monkeypatch, record, setting):
    if setting is None:
        monkeypatch.delenv("CLEAR_DATABASE", raising=False)
    else:
        monkeypatch.setenv("CLEAR_DATABASE", setting)
    driver, session = fake_driver(True)
    embedder = MagicMock(side_effect=AssertionError("must not load a model"))
    monkeypatch.setattr(legacy, "ExternalEmbedder", embedder)
    with pytest.raises(RuntimeError, match="旧管线不能覆盖"):
        legacy.write_graph(driver, [record], [{"entities": [], "relationships": []}])
    assert session.run.call_count == 1
    assert session.run.call_args.args[0].startswith("RETURN EXISTS")
    embedder.assert_not_called()


def test_missing_clear_option_does_not_delete_baseline_database(monkeypatch, record):
    monkeypatch.delenv("CLEAR_DATABASE", raising=False)
    driver, session = fake_driver(False)
    monkeypatch.setattr(legacy, "ExternalEmbedder", lambda **kw: SimpleNamespace(embed_document=lambda text: [0.0]))
    legacy.write_graph(driver, [record], [{"entities": [], "relationships": []}])
    queries = [call.args[0] for call in session.run.call_args_list]
    assert not any("DELETE" in query for query in queries)
    assert any("SET c.text" in query for query in queries)


def test_explicit_clear_still_reinitializes_database(monkeypatch, record):
    monkeypatch.setenv("CLEAR_DATABASE", "true")
    driver, session = fake_driver(True)
    monkeypatch.setattr(legacy, "ExternalEmbedder", lambda **kw: SimpleNamespace(embed_document=lambda text: [0.0]))
    legacy.write_graph(driver, [record], [{"entities": [], "relationships": []}])
    queries = [call.args[0] for call in session.run.call_args_list]
    assert queries[0] == "MATCH (n) DETACH DELETE n"
    assert any("SET c.text" in query for query in queries)


def test_main_checks_database_before_constructing_llm(monkeypatch, record):
    driver, session = fake_driver(True)
    monkeypatch.setenv("CLEAR_DATABASE", "false")
    monkeypatch.setattr(legacy, "_required_environment", lambda: ("bolt://example", "user", "pw", "token", "model"))
    monkeypatch.setattr(legacy, "records_from_environment", lambda: [record])
    monkeypatch.setattr(legacy.GraphDatabase, "driver", lambda *args, **kwargs: driver)
    llm = MagicMock(side_effect=AssertionError("must not call the LLM"))
    monkeypatch.setattr(legacy, "OpenAI", llm)
    with pytest.raises(RuntimeError, match="旧管线不能覆盖"):
        legacy.build_knowledge_graph()
    llm.assert_not_called()
    driver.close.assert_called_once()
    assert session.run.call_count == 1


@pytest.mark.integration
@pytest.mark.parametrize("seed", [
    "CREATE (:IngestRun {run_id: 'protected', status: 'ACTIVE'})",
    "CREATE (:Assertion {assertion_id: 'protected'})",
    "CREATE (:Chunk {chunk_id: 'protected', index: 0, text: '原始证据'})",
])
def test_real_database_guard_preserves_versioned_content(neo4j_driver, clean_database, monkeypatch, record, seed):
    monkeypatch.setenv("CLEAR_DATABASE", "false")
    monkeypatch.setenv("NEO4J_DB", os.environ["NEO4J_TEST_DB"])
    monkeypatch.setattr(legacy, "ExternalEmbedder", MagicMock(side_effect=AssertionError("must not load model")))
    with neo4j_driver.session(database=os.environ["NEO4J_TEST_DB"]) as session:
        session.run(seed).consume()
        before = session.run("MATCH (n) RETURN labels(n) AS labels, properties(n) AS props").data()
    with pytest.raises(RuntimeError, match="旧管线不能覆盖"):
        legacy.write_graph(neo4j_driver, [record], [{"entities": [], "relationships": []}])
    with neo4j_driver.session(database=os.environ["NEO4J_TEST_DB"]) as session:
        assert session.run("MATCH (n) RETURN labels(n) AS labels, properties(n) AS props").data() == before
