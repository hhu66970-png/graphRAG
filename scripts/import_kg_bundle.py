"""Validate and atomically import a portable GraphRAG bundle into Neo4j."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 200
BUNDLE_FORMAT = "zhongyaoxue-kg-bundle-v1"
SAFE_NAME = re.compile(r"^[\w\u4e00-\u9fff]+$", re.UNICODE)


def _driver():
    load_dotenv(ROOT / ".env")
    url = os.getenv("NEO4J_URL")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    if not all([url, user, password]):
        raise RuntimeError("缺少 NEO4J_URL / NEO4J_USER / NEO4J_PASSWORD")
    return GraphDatabase.driver(url, auth=(user, password)), os.getenv("NEO4J_DB") or None


def _read_jsonl(archive: zipfile.ZipFile, name: str):
    with archive.open(name, "r") as handle:
        for number, raw in enumerate(handle, start=1):
            line = raw.decode("utf-8").strip()
            if line:
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{name}:{number}: invalid JSON") from exc


def _identifier(value, location: str) -> str:
    if not isinstance(value, str) or not value or not SAFE_NAME.fullmatch(value):
        raise ValueError(f"{location}: invalid label or relationship type")
    return value


def _record_id(value, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: expected a nonempty string ID")
    return value


def _scalar_kind(value, location: str) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        if not -(2**63) <= value < 2**63:
            raise ValueError(f"{location}: integer exceeds Neo4j's 64-bit range")
        return "number"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location}: non-finite number")
        return "number"
    if isinstance(value, str):
        return "string"
    raise ValueError(f"{location}: unsupported Neo4j property type")


def _properties(props, location: str) -> None:
    if not isinstance(props, dict):
        raise ValueError(f"{location}: expected an object")
    for key, value in props.items():
        if not isinstance(key, str) or not key or "\x00" in key:
            raise ValueError(f"{location}: invalid property name")
        field = f"{location}.{key}"
        if value is None:
            continue
        if isinstance(value, list):
            kinds = {_scalar_kind(item, field) for item in value}
            if len(kinds) > 1:
                raise ValueError(f"{field}: property arrays must have one scalar type")
        else:
            _scalar_kind(value, field)


def read_bundle(bundle: Path) -> tuple[dict, list[dict], list[dict]]:
    """Validate the complete archive before opening a database connection."""
    with zipfile.ZipFile(bundle, "r") as archive:
        for name in ("manifest.json", "nodes.jsonl", "relationships.jsonl"):
            if archive.namelist().count(name) != 1:
                raise ValueError(f"Bundle must contain exactly one {name}")
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        nodes = list(_read_jsonl(archive, "nodes.jsonl"))
        relationships = list(_read_jsonl(archive, "relationships.jsonl"))
        if archive.testzip() is not None:
            raise ValueError("Bundle ZIP checksum failed")

    if not isinstance(manifest, dict) or manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"Unsupported bundle format; expected {BUNDLE_FORMAT}")
    for key, actual in (("node_count", len(nodes)), ("relationship_count", len(relationships))):
        count = manifest.get(key)
        if type(count) is not int or count < 0 or count != actual:
            raise ValueError(f"manifest.{key} does not match the archive")
    dimensions = manifest.get("embed_dim")
    if type(dimensions) is not int or dimensions <= 0:
        raise ValueError("manifest.embed_dim must be a positive integer")
    active_run = manifest.get("active_run_id")
    if active_run is not None:
        _record_id(active_run, "manifest.active_run_id")

    ids: set[str] = set()
    active_runs: list[str] = []
    for index, node in enumerate(nodes):
        location = f"nodes[{index}]"
        if not isinstance(node, dict):
            raise ValueError(f"{location}: expected an object")
        node_id = _record_id(node.get("id"), f"{location}.id")
        if node_id in ids:
            raise ValueError(f"{location}: duplicate node ID {node_id!r}")
        ids.add(node_id)
        labels = node.get("labels")
        if not isinstance(labels, list):
            raise ValueError(f"{location}.labels: expected an array")
        for label in labels:
            _identifier(label, f"{location}.labels")
        if len(set(labels)) != len(labels):
            raise ValueError(f"{location}.labels: duplicate label")
        _properties(node.get("props"), f"{location}.props")
        if "Chunk" in labels and node["props"].get("embedding") is not None:
            vector = node["props"]["embedding"]
            if not isinstance(vector, list) or len(vector) != dimensions or any(
                type(value) not in (int, float) for value in vector
            ):
                raise ValueError(f"{location}.props.embedding: inconsistent embedding dimensions or values")
        if "IngestRun" in labels and node["props"].get("status") == "ACTIVE":
            active_runs.append(_record_id(node["props"].get("run_id"), f"{location}.props.run_id"))
    if len(active_runs) > 1:
        raise ValueError("Bundle contains more than one ACTIVE run")
    if active_run is not None and active_runs != [active_run]:
        raise ValueError("manifest.active_run_id does not match the bundled ACTIVE run")

    for index, rel in enumerate(relationships):
        location = f"relationships[{index}]"
        if not isinstance(rel, dict):
            raise ValueError(f"{location}: expected an object")
        for key in ("source_id", "target_id"):
            endpoint = _record_id(rel.get(key), f"{location}.{key}")
            if endpoint not in ids:
                raise ValueError(f"{location}.{key}: unknown node {endpoint!r}")
        _identifier(rel.get("type"), f"{location}.type")
        _properties(rel.get("props"), f"{location}.props")
    return manifest, nodes, relationships


_SHOW_BUNDLE_INDEXES = (
    "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, options "
    "RETURN name, type, entityType, labelsOrTypes, properties, options"
)


def _check_bundle_indexes(rows: list[dict], dimensions: int, *, required: bool) -> None:
    for name, index_type, property_name in (
        ("text_embeddings", "VECTOR", "embedding"),
        ("text_fulltext", "FULLTEXT", "text"),
    ):
        named = next((row for row in rows if row["name"] == name), None)
        if named is not None:
            if (named["type"] != index_type or named["entityType"] != "NODE"
                    or named["labelsOrTypes"] != ["Chunk"]
                    or named["properties"] != [property_name]):
                raise ValueError(f"Existing {name} index does not match the bundle")
            if index_type == "VECTOR":
                config = named["options"].get("indexConfig", {})
                if (config.get("vector.dimensions") != dimensions
                        or str(config.get("vector.similarity_function", "cosine")).lower() != "cosine"):
                    raise ValueError(f"Existing {name} index does not match the bundle")
            continue
        equivalent = next((row for row in rows if
                           row["type"] == index_type and row["entityType"] == "NODE"
                           and row["labelsOrTypes"] == ["Chunk"]
                           and row["properties"] == [property_name]), None)
        if equivalent is not None:
            raise ValueError(
                f"Index {equivalent['name']!r} already covers Chunk.{property_name}; "
                f"the application requires the name {name!r}. "
                "Resolve the index name conflict before importing; no index was removed."
            )
        if required:
            raise RuntimeError(f"Neo4j did not create the required {name} index")


def _prepare_indexes(session, dimensions: int) -> None:
    """Check/create schema before the data transaction; never remove existing indexes."""
    _check_bundle_indexes(list(session.run(_SHOW_BUNDLE_INDEXES)), dimensions, required=False)
    session.run(
        "CREATE VECTOR INDEX text_embeddings IF NOT EXISTS "
        "FOR (n:Chunk) ON (n.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: $dimensions, "
        "`vector.similarity_function`: 'cosine'}}",
        dimensions=dimensions,
    ).consume()
    session.run(
        "CREATE FULLTEXT INDEX text_fulltext IF NOT EXISTS FOR (n:Chunk) ON EACH [n.text]"
    ).consume()
    # IF NOT EXISTS can succeed without creating the requested name when an equivalent
    # schema appears concurrently. Verify the named indexes before any data changes.
    _check_bundle_indexes(list(session.run(_SHOW_BUNDLE_INDEXES)), dimensions, required=True)
    session.run("CALL db.awaitIndexes(300)").consume()


def _write_graph(tx, nodes: list[dict], relationships: list[dict], *, clear: bool) -> None:
    if clear:
        tx.run("MATCH (n) DETACH DELETE n").consume()
    elif tx.run("MATCH (n) RETURN count(n) AS count").single()["count"]:
        raise ValueError("Target database is not empty; use an empty database or explicitly pass --clear")

    id_map: dict[str, str] = {}
    grouped_nodes: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for node in nodes:
        grouped_nodes[tuple(node["labels"])].append(node)
    for labels, rows in grouped_nodes.items():
        label_clause = "".join(f":`{label}`" for label in labels)
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start : start + BATCH_SIZE]
            result = tx.run(
                f"UNWIND $batch AS row CREATE (n{label_clause}) SET n += row.props "
                "RETURN row.id AS old_id, elementId(n) AS new_id",
                batch=batch,
            )
            for record in result:
                id_map[record["old_id"]] = record["new_id"]
    if len(id_map) != len(nodes):
        raise RuntimeError("Neo4j did not create every bundled node")

    grouped_rels: dict[str, list[dict]] = defaultdict(list)
    for rel in relationships:
        grouped_rels[rel["type"]].append({
            "source": id_map[rel["source_id"]],
            "target": id_map[rel["target_id"]],
            "props": rel["props"],
        })
    for rel_type, rows in grouped_rels.items():
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start : start + BATCH_SIZE]
            result = tx.run(
                "UNWIND $batch AS row MATCH (a), (b) "
                "WHERE elementId(a) = row.source AND elementId(b) = row.target "
                f"CREATE (a)-[r:`{rel_type}`]->(b) SET r += row.props "
                "RETURN count(r) AS created",
                batch=batch,
            ).single()
            if result is None or result["created"] != len(batch):
                raise RuntimeError("Neo4j did not create every bundled relationship")


def import_bundle(bundle: Path, *, clear: bool) -> None:
    bundle = bundle.expanduser().resolve()
    manifest, nodes, relationships = read_bundle(bundle)
    driver, database = _driver()
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            _prepare_indexes(session, manifest["embed_dim"])
            # All data changes, including deletion, commit together or roll back together.
            with session.begin_transaction() as tx:
                _write_graph(tx, nodes, relationships, clear=clear)
                tx.commit()
        print(f"imported {len(nodes)} nodes and {len(relationships)} relationships")
    finally:
        driver.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "dist" / "zhongyaoxue-kg-bundle.zip")
    parser.add_argument("--clear", action="store_true", help="Replace all target data in one transaction")
    args = parser.parse_args()
    import_bundle(args.bundle, clear=args.clear)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
