"""Bundle validation, replacement and transaction failure regressions."""

from __future__ import annotations

import copy
import json
import os
import zipfile
from pathlib import Path

import pytest

from scripts import export_kg_bundle as export
from scripts import import_kg_bundle as restore


def write_bundle(path, *, nodes=None, relationships=None, manifest=None):
    nodes = copy.deepcopy(nodes if nodes is not None else [
        {"id": "n1", "labels": ["Chunk"], "props": {"text": "麻黄", "embedding": [0.5, 0.5]}},
        {"id": "n2", "labels": ["Entity"], "props": {"canonical_id": "HERB:mahuang"}},
    ])
    relationships = copy.deepcopy(relationships if relationships is not None else [
        {"source_id": "n1", "target_id": "n2", "type": "MENTIONS", "props": {}},
    ])
    metadata = {"format": restore.BUNDLE_FORMAT, "node_count": len(nodes),
                "relationship_count": len(relationships), "embed_dim": 2, "active_run_id": None}
    metadata.update(manifest or {})
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(metadata))
        archive.writestr("nodes.jsonl", "\n".join(json.dumps(row) for row in nodes))
        archive.writestr("relationships.jsonl", "\n".join(json.dumps(row) for row in relationships))
    return path


class Result(list):
    def consume(self):
        return None

    def single(self):
        return self[0] if self else None


class FakeTransaction:
    def __init__(self, driver):
        self.driver = driver
        self.nodes = copy.deepcopy(driver.nodes)
        self.relationships = copy.deepcopy(driver.relationships)
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if not self.committed:
            self.driver.rollbacks += 1

    def run(self, query, **params):
        self.driver.events.append(query)
        if query == "MATCH (n) RETURN count(n) AS count":
            return Result([{"count": len(self.nodes)}])
        if "DETACH DELETE" in query:
            self.nodes.clear()
            self.relationships.clear()
            return Result()
        if "RETURN row.id AS old_id" in query:
            if self.driver.failure == "nodes":
                raise RuntimeError("node constraint failure")
            rows = []
            for node in params["batch"]:
                new = copy.deepcopy(node)
                new["id"] = f"new-{node['id']}"
                self.nodes.append(new)
                rows.append({"old_id": node["id"], "new_id": new["id"]})
            return Result(rows)
        if "RETURN count(r) AS created" in query:
            if self.driver.failure == "relationships":
                raise RuntimeError("relationship write failure")
            self.relationships.extend(copy.deepcopy(params["batch"]))
            return Result([{"created": len(params["batch"])}])
        raise AssertionError(f"Unexpected transaction query: {query}")

    def commit(self):
        if self.driver.failure == "commit":
            raise RuntimeError("commit failure")
        self.driver.nodes = self.nodes
        self.driver.relationships = self.relationships
        self.committed = True
        self.driver.commits += 1


class FakeSession:
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def begin_transaction(self):
        self.driver.transactions += 1
        return FakeTransaction(self.driver)

    def run(self, query, **params):
        self.driver.events.append(query)
        if query.startswith("SHOW INDEXES"):
            return Result(self.driver.indexes)
        if query.startswith("CREATE ") or query.startswith("CALL db.awaitIndexes"):
            if self.driver.failure == "index":
                raise RuntimeError("index creation failure")
            if query.startswith("CREATE ") and self.driver.failure != "index_noop":
                is_vector = query.startswith("CREATE VECTOR")
                name = "text_embeddings" if is_vector else "text_fulltext"
                if not any(row["name"] == name for row in self.driver.indexes):
                    self.driver.indexes.append({
                        "name": name, "type": "VECTOR" if is_vector else "FULLTEXT",
                        "entityType": "NODE", "labelsOrTypes": ["Chunk"],
                        "properties": ["embedding" if is_vector else "text"],
                        "options": {"indexConfig": {"vector.dimensions": params["dimensions"]}}
                        if is_vector else {},
                    })
            return Result()
        if query == export._ACTIVE_RUN:
            return Result([{"run_id": "active"}])
        if query in (export._NODES_ALL, export._NODES_ACTIVE_ONLY):
            return Result(self.driver.nodes)
        if query in (export._RELS_ALL, export._RELS_ACTIVE_ONLY):
            if self.driver.failure == "export_relationships":
                raise RuntimeError("export read failure")
            return Result(self.driver.relationships)
        raise AssertionError(f"Unexpected session query: {query}")


class FakeDriver:
    def __init__(self, *, failure=None):
        self.failure = failure
        self.nodes = [{"id": "original", "labels": ["Chunk"], "props": {"text": "original"}}]
        self.relationships = []
        self.indexes = []
        self.events = []
        self.transactions = 0
        self.commits = 0
        self.rollbacks = 0
        self.closed = 0

    def verify_connectivity(self):
        if self.failure == "connection":
            raise ConnectionError("offline")

    def session(self, **kwargs):
        return FakeSession(self)

    def close(self):
        self.closed += 1


@pytest.mark.parametrize("field,value", [
    ("format", "future-format"), ("node_count", 999), ("node_count", True),
    ("relationship_count", -1), ("embed_dim", 0), ("embed_dim", "512"),
    ("active_run_id", "not-in-bundle"),
])
def test_manifest_rejected_before_opening_database(tmp_path, monkeypatch, field, value):
    path = write_bundle(tmp_path / "bad.zip", manifest={field: value})
    monkeypatch.setattr(restore, "_driver", lambda: pytest.fail("Invalid bundle opened database"))
    with pytest.raises(ValueError):
        restore.import_bundle(path, clear=True)


@pytest.mark.parametrize("mutation", [
    lambda rows: rows[0].pop("id"),
    lambda rows: rows.append(copy.deepcopy(rows[0])),
    lambda rows: rows[0].update(labels="Chunk"),
    lambda rows: rows[0].update(labels=["Chunk` malicious"]),
    lambda rows: rows[0].update(props={"nested": {"x": 1}}),
    lambda rows: rows[0].update(props={"values": [1, None]}),
    lambda rows: rows[0].update(props={"values": [1, "one"]}),
    lambda rows: rows[0].update(props={"integer": 2**63}),
    lambda rows: rows[0].update(props={"number": float("nan")}),
    lambda rows: rows[0].update(props={"embedding": [0.1]}),
    lambda rows: rows[0].update(props={"embedding": [True, False]}),
])
def test_node_errors_rejected_before_clear(tmp_path, monkeypatch, mutation):
    nodes = [{"id": "n1", "labels": ["Chunk"], "props": {}}]
    mutation(nodes)
    path = write_bundle(tmp_path / "bad.zip", nodes=nodes, relationships=[])
    monkeypatch.setattr(restore, "_driver", lambda: pytest.fail("Invalid bundle opened database"))
    with pytest.raises(ValueError):
        restore.import_bundle(path, clear=True)


@pytest.mark.parametrize("mutation", [
    lambda row: row.pop("source_id"),
    lambda row: row.update(target_id="missing"),
    lambda row: row.update(type="bad`type"),
    lambda row: row.update(props={"bad": [[1]]}),
])
def test_relation_errors_rejected_before_clear(tmp_path, monkeypatch, mutation):
    rel = {"source_id": "n1", "target_id": "n2", "type": "MENTIONS", "props": {}}
    mutation(rel)
    path = write_bundle(tmp_path / "bad.zip", relationships=[rel])
    monkeypatch.setattr(restore, "_driver", lambda: pytest.fail("Invalid bundle opened database"))
    with pytest.raises(ValueError):
        restore.import_bundle(path, clear=True)


@pytest.mark.parametrize("failure", ["nodes", "relationships", "commit"])
def test_import_failure_rolls_back_clear_and_partial_writes(tmp_path, monkeypatch, failure):
    driver = FakeDriver(failure=failure)
    before = copy.deepcopy(driver.nodes)
    monkeypatch.setattr(restore, "_driver", lambda: (driver, "test"))
    with pytest.raises(RuntimeError):
        restore.import_bundle(write_bundle(tmp_path / "bundle.zip"), clear=True)
    assert driver.nodes == before
    assert driver.relationships == []
    assert driver.commits == 0
    assert driver.rollbacks == 1
    assert any("DETACH DELETE" in query for query in driver.events)
    assert driver.closed == 1


def test_import_commits_nodes_and_relationships_together(tmp_path, monkeypatch):
    driver = FakeDriver()
    monkeypatch.setattr(restore, "_driver", lambda: (driver, "test"))
    restore.import_bundle(write_bundle(tmp_path / "bundle.zip"), clear=True)
    assert {row["id"] for row in driver.nodes} == {"new-n1", "new-n2"}
    assert driver.relationships == [{"source": "new-n1", "target": "new-n2", "props": {}}]
    assert (driver.transactions, driver.commits, driver.rollbacks) == (1, 1, 0)
    assert driver.closed == 1


@pytest.mark.parametrize("existing_dimension", [1, 3])
def test_incompatible_index_rejected_before_data_transaction(tmp_path, monkeypatch, existing_dimension):
    driver = FakeDriver()
    driver.indexes = [{"name": "text_embeddings", "type": "VECTOR", "entityType": "NODE", "labelsOrTypes": ["Chunk"],
                       "properties": ["embedding"], "options": {"indexConfig": {"vector.dimensions": existing_dimension}}}]
    monkeypatch.setattr(restore, "_driver", lambda: (driver, "test"))
    with pytest.raises(ValueError, match="index"):
        restore.import_bundle(write_bundle(tmp_path / "bundle.zip"), clear=True)
    assert driver.transactions == 0
    assert driver.nodes[0]["id"] == "original"


def test_index_creation_failure_preserves_existing_graph(tmp_path, monkeypatch):
    driver = FakeDriver(failure="index")
    monkeypatch.setattr(restore, "_driver", lambda: (driver, "test"))
    with pytest.raises(RuntimeError, match="index"):
        restore.import_bundle(write_bundle(tmp_path / "bundle.zip"), clear=True)
    assert driver.transactions == 0
    assert driver.nodes[0]["id"] == "original"


@pytest.mark.parametrize("failure", ["connection", "export_relationships", "invalid_archive", "fsync", "replace"])
def test_export_failure_preserves_existing_backup_and_cleans_temp(tmp_path, monkeypatch, failure):
    output = write_bundle(tmp_path / "backup.zip")
    before = output.read_bytes()
    driver = FakeDriver(failure=failure)
    if failure == "invalid_archive":
        driver.relationships = [{"source_id": "original", "target_id": "missing", "type": "MENTIONS", "props": {}}]
    if failure == "replace":
        monkeypatch.setattr(Path, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk failure")))
    if failure == "fsync":
        monkeypatch.setattr(export.os, "fsync", lambda *args: (_ for _ in ()).throw(OSError("sync failure")))
    monkeypatch.setattr(export, "_driver", lambda: (driver, "test"))
    with pytest.raises((ConnectionError, RuntimeError, ValueError, OSError)):
        export.export_bundle(output)
    assert output.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["backup.zip"]
    assert driver.closed == 1


def test_successful_export_is_valid_and_uses_actual_vector_dimensions(tmp_path, monkeypatch):
    output = write_bundle(tmp_path / "backup.zip")
    driver = FakeDriver()
    driver.nodes[0]["props"]["embedding"] = [0.2, 0.3, 0.4]
    monkeypatch.setenv("EMBED_DIM", "512")
    monkeypatch.setattr(export, "_driver", lambda: (driver, "test"))
    export.export_bundle(output)
    manifest, nodes, rels = restore.read_bundle(output)
    assert manifest["embed_dim"] == 3
    assert nodes == driver.nodes
    assert rels == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["backup.zip"]


def test_export_syncs_a_writable_descriptor_before_replacing_backup(tmp_path, monkeypatch):
    output = write_bundle(tmp_path / "backup.zip")
    before = output.read_bytes()
    driver = FakeDriver()
    real_fsync = os.fsync
    synced = []

    def windows_compatible_fsync(descriptor):
        # A zero-byte write exposes read-only descriptors on both Windows and Linux.
        # Windows fsync/_commit rejects them with EBADF even for an already closed zip.
        assert output.read_bytes() == before
        os.write(descriptor, b"")
        real_fsync(descriptor)
        synced.append(descriptor)

    monkeypatch.setattr(export, "_driver", lambda: (driver, "test"))
    monkeypatch.setattr(export.os, "fsync", windows_compatible_fsync)
    export.export_bundle(output)
    assert len(synced) == 1
    _, nodes, _ = restore.read_bundle(output)
    assert nodes == driver.nodes
    assert sorted(p.name for p in tmp_path.iterdir()) == ["backup.zip"]


@pytest.mark.integration
def test_active_export_preserves_shared_entities(neo4j_driver, clean_database, tmp_path, monkeypatch):
    database = os.getenv("NEO4J_TEST_DB") or "neo4j"

    class BorrowedDriver:
        def __getattr__(self, name):
            return getattr(neo4j_driver, name)

        def close(self):
            pass

    with neo4j_driver.session(database=database) as session:
        session.run("""
            CREATE (old:IngestRun {run_id:'bundle-old', status:'SUPERSEDED'})
            CREATE (current:IngestRun {run_id:'bundle-active', status:'ACTIVE'})
            CREATE (s:Entity {canonical_id:'bundle-subject', run_id:'bundle-old'})
            CREATE (o:Entity {canonical_id:'bundle-object', run_id:'bundle-old'})
            CREATE (a:Assertion {assertion_id:'bundle-assertion', run_id:'bundle-active'})
            CREATE (s)-[:IN_RUN]->(current)
            CREATE (a)-[:SUBJECT]->(s)
            CREATE (a)-[:OBJECT]->(o)
            CREATE (s)-[:HAS_EFFICACY {fact_key:'bundle-fact', active_run_id:'bundle-active'}]->(o)
        """).consume()
    monkeypatch.setattr(export, "_driver", lambda: (BorrowedDriver(), database))
    output = export.export_bundle(tmp_path / "active.zip", active_only=True)
    manifest, nodes, relationships = restore.read_bundle(output)
    assert manifest["active_run_id"] == "bundle-active"
    assert {row["props"]["canonical_id"] for row in nodes if "Entity" in row["labels"]} == {
        "bundle-subject", "bundle-object",
    }
    assert not any(row["props"].get("run_id") == "bundle-old" and "IngestRun" in row["labels"] for row in nodes)
    assert any(row["type"] == "HAS_EFFICACY" for row in relationships)


def test_nonempty_target_requires_explicit_clear(tmp_path, monkeypatch):
    driver = FakeDriver()
    before = copy.deepcopy(driver.nodes)
    monkeypatch.setattr(restore, "_driver", lambda: (driver, "test"))
    with pytest.raises(ValueError, match="not empty"):
        restore.import_bundle(write_bundle(tmp_path / "bundle.zip"), clear=False)
    assert driver.nodes == before
    assert driver.commits == 0
    assert not any("DETACH DELETE" in query for query in driver.events)


@pytest.mark.integration
def test_real_constraint_failure_rolls_back_import_clear(neo4j_driver, clean_database, tmp_path, monkeypatch):
    database = os.getenv("NEO4J_TEST_DB") or "neo4j"

    class BorrowedDriver:
        def __getattr__(self, name):
            return getattr(neo4j_driver, name)

        def close(self):
            pass

    with neo4j_driver.session(database=database) as session:
        session.run("CREATE CONSTRAINT bundle_import_unique IF NOT EXISTS "
                    "FOR (n:BundleImportTest) REQUIRE n.unique_id IS UNIQUE").consume()
        session.run("CREATE (:BundleImportTest {unique_id: 'original'})").consume()
        index = session.run("SHOW INDEXES YIELD name, options WHERE name='text_embeddings' RETURN options").single()
        dimensions = index["options"]["indexConfig"]["vector.dimensions"] if index else 2
    nodes = [
        {"id": "one", "labels": ["BundleImportTest"], "props": {"unique_id": "duplicate"}},
        {"id": "two", "labels": ["BundleImportTest"], "props": {"unique_id": "duplicate"}},
    ]
    bundle = write_bundle(tmp_path / "constraint.zip", nodes=nodes, relationships=[], manifest={"embed_dim": dimensions})
    monkeypatch.setattr(restore, "_driver", lambda: (BorrowedDriver(), database))
    from neo4j.exceptions import ConstraintError

    try:
        with pytest.raises(ConstraintError):
            restore.import_bundle(bundle, clear=True)
        with neo4j_driver.session(database=database) as session:
            values = session.run("MATCH (n:BundleImportTest) RETURN n.unique_id AS value")
            assert [row["value"] for row in values] == ["original"]
    finally:
        with neo4j_driver.session(database=database) as session:
            session.run("DROP CONSTRAINT bundle_import_unique IF EXISTS").consume()


@pytest.mark.integration
def test_real_import_export_round_trip(neo4j_driver, clean_database, tmp_path, monkeypatch):
    database = os.getenv("NEO4J_TEST_DB") or "neo4j"

    class BorrowedDriver:
        def __getattr__(self, name):
            return getattr(neo4j_driver, name)

        def close(self):
            pass

    with neo4j_driver.session(database=database) as session:
        index = session.run("SHOW INDEXES YIELD name, options WHERE name='text_embeddings' RETURN options").single()
        dimensions = index["options"]["indexConfig"]["vector.dimensions"] if index else 2
    nodes = [
        {"id": "one", "labels": ["Chunk"], "props": {"text": "麻黄", "embedding": [0.5] * dimensions}},
        {"id": "two", "labels": ["Entity"], "props": {"canonical_id": "bundle-roundtrip"}},
    ]
    rels = [{"source_id": "one", "target_id": "two", "type": "MENTIONS", "props": {"source": "test"}}]
    bundle = write_bundle(tmp_path / "input.zip", nodes=nodes, relationships=rels, manifest={"embed_dim": dimensions})
    monkeypatch.setattr(restore, "_driver", lambda: (BorrowedDriver(), database))
    monkeypatch.setattr(export, "_driver", lambda: (BorrowedDriver(), database))
    restore.import_bundle(bundle, clear=False)
    output = export.export_bundle(tmp_path / "output.zip")
    _, exported_nodes, exported_rels = restore.read_bundle(output)
    assert sorted((row["labels"], row["props"]) for row in exported_nodes) == sorted((row["labels"], row["props"]) for row in nodes)
    assert [(row["type"], row["props"]) for row in exported_rels] == [("MENTIONS", {"source": "test"})]


@pytest.mark.parametrize("index_type,property_name", [("VECTOR", "embedding"), ("FULLTEXT", "text")])
def test_equivalent_index_with_another_name_is_rejected_before_data_changes(
    tmp_path, monkeypatch, index_type, property_name,
):
    driver = FakeDriver()
    driver.indexes = [{
        "name": "legacy_index", "type": index_type, "entityType": "NODE",
        "labelsOrTypes": ["Chunk"], "properties": [property_name],
        "options": {"indexConfig": {"vector.dimensions": 2}},
    }]
    before = copy.deepcopy(driver.indexes)
    monkeypatch.setattr(restore, "_driver", lambda: (driver, "test"))
    with pytest.raises(ValueError, match="legacy_index.*already covers"):
        restore.import_bundle(write_bundle(tmp_path / "bundle.zip"), clear=True)
    assert driver.transactions == 0
    assert driver.nodes[0]["id"] == "original"
    assert driver.indexes == before
    assert not any(query.startswith(("CREATE ", "DROP ")) for query in driver.events)


def test_index_creation_noop_is_detected_before_clear(tmp_path, monkeypatch):
    driver = FakeDriver(failure="index_noop")
    monkeypatch.setattr(restore, "_driver", lambda: (driver, "test"))
    with pytest.raises(RuntimeError, match="did not create.*text_embeddings"):
        restore.import_bundle(write_bundle(tmp_path / "bundle.zip"), clear=True)
    assert driver.transactions == 0
    assert driver.nodes[0]["id"] == "original"


def test_same_named_relationship_vector_index_is_not_accepted(tmp_path, monkeypatch):
    driver = FakeDriver()
    driver.indexes = [{
        "name": "text_embeddings", "type": "VECTOR", "entityType": "RELATIONSHIP",
        "labelsOrTypes": ["Chunk"], "properties": ["embedding"],
        "options": {"indexConfig": {"vector.dimensions": 2}},
    }]
    monkeypatch.setattr(restore, "_driver", lambda: (driver, "test"))
    with pytest.raises(ValueError, match="does not match"):
        restore.import_bundle(write_bundle(tmp_path / "bundle.zip"), clear=True)
    assert driver.transactions == 0
    assert driver.nodes[0]["id"] == "original"
