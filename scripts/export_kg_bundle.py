"""Export Neo4j knowledge graph (nodes, relationships, embeddings) to a portable zip.

The zip can be shared offline and restored with scripts/import_kg_bundle.py.
Runtime QA does not depend on this script; it only packages an already-built database.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

if __package__:
    from .import_kg_bundle import BUNDLE_FORMAT, read_bundle
else:
    from import_kg_bundle import BUNDLE_FORMAT, read_bundle

ROOT = Path(__file__).resolve().parents[1]


def _driver():
    """按环境变量创建 Neo4j Driver。"""
    load_dotenv(ROOT / ".env")
    url = os.getenv("NEO4J_URL")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    if not all([url, user, password]):
        raise RuntimeError("缺少 NEO4J_URL / NEO4J_USER / NEO4J_PASSWORD")
    return GraphDatabase.driver(url, auth=(user, password)), os.getenv("NEO4J_DB") or None


# Chunks and legacy nodes are shared; entities can belong to several runs.
_ACTIVE_RUN = "MATCH (r:IngestRun {status:'ACTIVE'}) RETURN r.run_id AS run_id"


def _in_active_scope(alias: str) -> str:
    return f"""(
        NOT ({alias}:Assertion OR {alias}:Entity OR {alias}:IngestRun)
        OR {alias}.run_id = $run
        OR ({alias}:Entity AND (
            EXISTS {{ MATCH ({alias})-[:IN_RUN]->(:IngestRun {{run_id: $run}}) }}
            OR EXISTS {{ MATCH (:Assertion {{run_id: $run}})-[:SUBJECT|OBJECT]->({alias}) }}
        ))
    )"""


_NODES_ALL = """
MATCH (n)
RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props
"""

_NODES_ACTIVE_ONLY = f"""
MATCH (n) WHERE {_in_active_scope('n')}
RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props
"""

_RELS_ALL = """
MATCH (a)-[r]->(b)
RETURN elementId(a) AS source_id, elementId(b) AS target_id,
       type(r) AS type, properties(r) AS props
"""

_RELS_ACTIVE_ONLY = f"""
MATCH (a)-[r]->(b)
WHERE {_in_active_scope('a')} AND {_in_active_scope('b')}
  AND (r.active_run_id IS NULL OR r.active_run_id = $run)
RETURN elementId(a) AS source_id, elementId(b) AS target_id,
       type(r) AS type, properties(r) AS props
"""


def export_bundle(output: Path, *, active_only: bool = False) -> Path:
    """把库中的图与向量导出为 zip。"""
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    driver, database = _driver()
    temporary: Path | None = None

    try:
        driver.verify_connectivity()
        with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
        with driver.session(database=database) as session, zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            run_id = None
            if active_only:
                rows = list(session.run(_ACTIVE_RUN))
                if len(rows) != 1:
                    raise RuntimeError("按 --active-only 导出需要且仅能有一个 ACTIVE 批次")
                run_id = rows[0]["run_id"]
                print(f"only exporting ACTIVE run: {run_id}")

            node_count = 0
            dimensions = None
            with archive.open("nodes.jsonl", "w") as handle:
                for record in session.run(
                    _NODES_ACTIVE_ONLY if active_only else _NODES_ALL,
                    **({"run": run_id} if active_only else {}),
                ):
                    payload = {
                        "id": record["id"],
                        "labels": list(record["labels"]),
                        "props": dict(record["props"]),
                    }
                    if "Chunk" in payload["labels"] and payload["props"].get("embedding") is not None:
                        actual = len(payload["props"]["embedding"])
                        if dimensions is not None and dimensions != actual:
                            raise ValueError("Chunk embeddings have inconsistent dimensions")
                        dimensions = actual
                    handle.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
                    node_count += 1
                    if node_count % 1000 == 0:
                        print(f"exported nodes: {node_count}", flush=True)

            rel_count = 0
            with archive.open("relationships.jsonl", "w") as handle:
                for record in session.run(
                    _RELS_ACTIVE_ONLY if active_only else _RELS_ALL,
                    **({"run": run_id} if active_only else {}),
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
                "format": BUNDLE_FORMAT,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "node_count": node_count,
                "relationship_count": rel_count,
                "embed_dim": dimensions if dimensions is not None else int(os.getenv("EMBED_DIM", "512")),
                "embed_model": os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
                "active_run_id": run_id,
                "notes": "Contains Chunk.embedding vectors and domain entities. Recreate indexes after import.",
            }
            archive.writestr("manifest.json", json.dumps(meta, ensure_ascii=False, indent=2))
        read_bundle(temporary)
        # Replacing only a validated archive also preserves the previous backup on read/write failures.
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(output)
        print(f"done: {output} ({node_count} nodes, {rel_count} relationships)")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
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
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="导出 ACTIVE 批次及其共享文本与实体，排除历史批次",
    )
    args = parser.parse_args()
    export_bundle(args.output, active_only=args.active_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
