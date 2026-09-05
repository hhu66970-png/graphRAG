"""Import a portable GraphRAG zip into Neo4j and recreate vector/fulltext indexes."""

from __future__ import annotations

import argparse
import json
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 200
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
        for raw in handle:
            line = raw.decode("utf-8").strip()
            if line:
                yield json.loads(line)


def _safe_token(value: str, fallback: str) -> str:
    value = (value or "").strip() or fallback
    if not SAFE_NAME.match(value):
        cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "_", value, flags=re.UNICODE).strip("_")
        return cleaned or fallback
    return value


def import_bundle(bundle: Path, *, clear: bool) -> None:
    bundle = bundle.expanduser().resolve()
    if not bundle.is_file():
        raise FileNotFoundError(bundle)

    driver, database = _driver()
    try:
        driver.verify_connectivity()
        with zipfile.ZipFile(bundle, "r") as archive:
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            print(
                f"importing {bundle.name}: "
                f"{manifest.get('node_count')} nodes, "
                f"{manifest.get('relationship_count')} relationships"
            )

            nodes = list(_read_jsonl(archive, "nodes.jsonl"))
            relationships = list(_read_jsonl(archive, "relationships.jsonl"))

            id_map: dict[str, str] = {}
            with driver.session(database=database) as session:
                if clear:
                    print("清空当前 Neo4j 数据库...")
                    session.run("MATCH (n) DETACH DELETE n").consume()

                grouped_nodes: dict[tuple[str, ...], list[dict]] = defaultdict(list)
                for node in nodes:
                    labels = tuple(
                        _safe_token(label, "概念")
                        for label in (node.get("labels") or ["概念"])
                        if label
                    ) or ("概念",)
                    grouped_nodes[labels].append(
                        {"id": node["id"], "props": node.get("props") or {}}
                    )

                imported_nodes = 0
                for labels, rows in grouped_nodes.items():
                    label_cypher = ":".join(f"`{label}`" for label in labels)
                    for start in range(0, len(rows), BATCH_SIZE):
                        batch = rows[start : start + BATCH_SIZE]
                        result = session.run(
                            f"""
                            UNWIND $batch AS row
                            CREATE (n:{label_cypher})
                            SET n += row.props
                            RETURN row.id AS old_id, elementId(n) AS new_id
                            """,
                            batch=batch,
                        )
                        for record in result:
                            id_map[record["old_id"]] = record["new_id"]
                            imported_nodes += 1
                        print(f"imported nodes: {imported_nodes}/{len(nodes)}", flush=True)

                grouped_rels: dict[str, list[dict]] = defaultdict(list)
                for rel in relationships:
                    rel_type = _safe_token(rel.get("type") or "相关", "相关")
                    source = id_map.get(rel["source_id"])
                    target = id_map.get(rel["target_id"])
                    if not source or not target:
                        continue
                    grouped_rels[rel_type].append(
                        {
                            "source": source,
                            "target": target,
                            "props": rel.get("props") or {},
                        }
                    )

                imported_rels = 0
                total_rels = sum(len(rows) for rows in grouped_rels.values())
                for rel_type, rows in grouped_rels.items():
                    for start in range(0, len(rows), BATCH_SIZE):
                        batch = rows[start : start + BATCH_SIZE]
                        session.run(
                            f"""
                            UNWIND $batch AS row
                            MATCH (a), (b)
                            WHERE elementId(a) = row.source AND elementId(b) = row.target
                            CREATE (a)-[r:`{rel_type}`]->(b)
                            SET r += row.props
                            """,
                            batch=batch,
                        ).consume()
                        imported_rels += len(batch)
                        print(
                            f"imported relationships: {imported_rels}/{total_rels}",
                            flush=True,
                        )

            dimensions = int(manifest.get("embed_dim") or os.getenv("EMBED_DIM", "512"))
            print("creating vector index text_embeddings...")
            with driver.session(database=database) as session:
                session.run(
                    """
                    CREATE VECTOR INDEX text_embeddings IF NOT EXISTS
                    FOR (n:Chunk) ON (n.embedding)
                    OPTIONS {indexConfig: {
                      `vector.dimensions`: $dimensions,
                      `vector.similarity_function`: 'cosine'
                    }}
                    """,
                    dimensions=dimensions,
                ).consume()
                print("vector index ready")
                session.run(
                    """
                    CREATE FULLTEXT INDEX text_fulltext IF NOT EXISTS
                    FOR (n:Chunk) ON EACH [n.text]
                    """
                ).consume()
            print("fulltext index ready")
            print("import completed")
    finally:
        driver.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Import GraphRAG Neo4j bundle")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=ROOT / "dist" / "zhongyaoxue-kg-bundle.zip",
        help="Bundle zip created by export_kg_bundle.py",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the target Neo4j database before import",
    )
    args = parser.parse_args()
    import_bundle(args.bundle, clear=args.clear)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
