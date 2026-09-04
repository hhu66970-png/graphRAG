"""Export Neo4j knowledge graph (nodes, relationships, embeddings) to a portable zip.

The zip can be shared offline and restored with scripts/import_kg_bundle.py.
Runtime QA does not depend on this script; it only packages an already-built database.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]


def _driver():
    load_dotenv(ROOT / ".env")
    url = os.getenv("NEO4J_URL")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    if not all([url, user, password]):
        raise RuntimeError("缺少 NEO4J_URL / NEO4J_USER / NEO4J_PASSWORD")
    return GraphDatabase.driver(url, auth=(user, password)), os.getenv("NEO4J_DB") or None


def export_bundle(output: Path) -> Path:
    driver, database = _driver()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session, zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            node_count = 0
            with archive.open("nodes.jsonl", "w") as handle:
                for record in session.run(
                    """
                    MATCH (n)
                    RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props
                    """
                ):
                    payload = {
                        "id": record["id"],
                        "labels": list(record["labels"]),
                        "props": dict(record["props"]),
                    }
                    handle.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
                    node_count += 1
                    if node_count % 1000 == 0:
                        print(f"exported nodes: {node_count}", flush=True)

            rel_count = 0
            with archive.open("relationships.jsonl", "w") as handle:
                for record in session.run(
                    """
                    MATCH (a)-[r]->(b)
                    RETURN elementId(a) AS source_id, elementId(b) AS target_id,
                           type(r) AS type, properties(r) AS props
                    """
                ):
                    payload = {
                        "source_id": record["source_id"],
                        "target_id": record["target_id"],
                        "type": record["type"],
                        "props": dict(record["props"]),
                    }
                    handle.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
                    rel_count += 1
                    if rel_count % 5000 == 0:
                        print(f"exported relationships: {rel_count}", flush=True)

            meta = {
                "format": "zhongyaoxue-kg-bundle-v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "node_count": node_count,
                "relationship_count": rel_count,
                "embed_dim": int(os.getenv("EMBED_DIM", "512")),
                "embed_model": os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
                "notes": "Contains Chunk.embedding vectors and domain entities. Recreate indexes after import.",
            }
            archive.writestr("manifest.json", json.dumps(meta, ensure_ascii=False, indent=2))
            print(f"done: {output} ({node_count} nodes, {rel_count} relationships)")
    finally:
        driver.close()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Neo4j GraphRAG bundle")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / "zhongyaoxue-kg-bundle.zip",
        help="Output zip path",
    )
    args = parser.parse_args()
    export_bundle(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
