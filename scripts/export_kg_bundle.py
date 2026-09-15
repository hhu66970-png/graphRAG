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
    """按环境变量创建 Neo4j Driver。"""
    load_dotenv(ROOT / ".env")
    url = os.getenv("NEO4J_URL")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    if not all([url, user, password]):
        raise RuntimeError("缺少 NEO4J_URL / NEO4J_USER / NEO4J_PASSWORD")
    return GraphDatabase.driver(url, auth=(user, password)), os.getenv("NEO4J_DB") or None


# 只保留 ACTIVE run 的 Assertion/Entity/IngestRun；Document、Chunk 与旧图实体
# 属于共享数据，始终导出（旧图是前端四种对照检索模式的数据来源）。
_ACTIVE_RUN = "MATCH (r:IngestRun {status:'ACTIVE'}) RETURN r.run_id AS run_id LIMIT 1"

_NODES_ALL = """
MATCH (n)
RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props
"""

_NODES_ACTIVE_ONLY = """
MATCH (n)
WHERE NOT (n:Assertion OR n:Entity OR n:IngestRun) OR n.run_id = $run
RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props
"""

_RELS_ALL = """
MATCH (a)-[r]->(b)
RETURN elementId(a) AS source_id, elementId(b) AS target_id,
       type(r) AS type, properties(r) AS props
"""

# 关系两端都必须在导出的节点集合内，否则导入时会因找不到端点被静默丢弃
_RELS_ACTIVE_ONLY = """
MATCH (a)-[r]->(b)
WHERE (NOT (a:Assertion OR a:Entity OR a:IngestRun) OR a.run_id = $run)
  AND (NOT (b:Assertion OR b:Entity OR b:IngestRun) OR b.run_id = $run)
  AND (r.active_run_id IS NULL OR r.active_run_id = $run)
RETURN elementId(a) AS source_id, elementId(b) AS target_id,
       type(r) AS type, properties(r) AS props
"""


def export_bundle(output: Path, *, active_only: bool = False) -> Path:
    """把库中的图与向量导出为 zip。"""
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
            run_id = None
            if active_only:
                rows = list(session.run(_ACTIVE_RUN))
                if not rows:
                    raise RuntimeError("没有 ACTIVE 的 IngestRun，无法按 --active-only 导出")
                run_id = rows[0]["run_id"]
                print(f"only exporting ACTIVE run: {run_id}")

            node_count = 0
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
                "format": "zhongyaoxue-kg-bundle-v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "node_count": node_count,
                "relationship_count": rel_count,
                "embed_dim": int(os.getenv("EMBED_DIM", "512")),
                "embed_model": os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
                "active_run_id": run_id,
                "notes": "Contains Chunk.embedding vectors and domain entities. Recreate indexes after import.",
            }
            archive.writestr("manifest.json", json.dumps(meta, ensure_ascii=False, indent=2))
            print(f"done: {output} ({node_count} nodes, {rel_count} relationships)")
    finally:
        driver.close()
    return output


def main() -> int:
    """命令行入口：解析参数并执行本脚本的主流程。"""
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
        help="只导出 ACTIVE run 的 Assertion/Entity/IngestRun，排除已废弃的历史 run",
    )
    args = parser.parse_args()
    export_bundle(args.output, active_only=args.active_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
