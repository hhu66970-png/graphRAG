"""从 Assertion 层重建类型化 Serving 边。

为 BUILDING 批次预备 ACCEPTED、POSITIVE 的断言投影，发布后由读取端使用。版本、冲突和完整来源
保留在 Assertion 层，Serving 边用于检索，可随时重建。
关系类型经过本体白名单校验后才拼入 Cypher；其余值通过参数传入。
"""

from __future__ import annotations

from collections import defaultdict

from ..domain.ontology import Ontology
from ..domain.errors import RunStateError
from .neo4j_store import Neo4jStore

# 每条 Serving Edge 携带的证据摘要
_PROJECT_TEMPLATE = """
UNWIND $batch AS row
MATCH (s:Entity {{canonical_id: row.subject_id}})
MATCH (o:Entity {{canonical_id: row.object_id}})
MERGE (s)-[e:`{predicate}` {{fact_key: row.fact_key, active_run_id: row.run_id}}]->(o)
SET e.best_support_score = row.best_support_score,
    e.evidence_count     = row.evidence_count,
    e.evidence_ids       = row.evidence_ids,
    e.page_start         = row.page_start,
    e.page_end           = row.page_end,
    e.active_run_id      = row.run_id
"""

_COLLECT_FACTS = """
MATCH (a:Assertion {run_id: $run_id, validation_status: 'ACCEPTED'})
MATCH (a)-[:SUBJECT]->(s:Entity)
MATCH (a)-[:OBJECT]->(o:Entity)
WHERE a.polarity = 'POSITIVE'
WITH a.fact_key AS fact_key, a.predicate AS predicate,
     s.canonical_id AS subject_id, o.canonical_id AS object_id,
     max(a.final_score) AS best_support_score,
     count(a) AS evidence_count,
     collect(a.assertion_id)[..10] AS evidence_ids,
     min(a.page_start) AS page_start,
     max(a.page_end) AS page_end
RETURN fact_key, predicate, subject_id, object_id,
       best_support_score, evidence_count, evidence_ids, page_start, page_end
"""

_DROP_PROJECTION = """
MATCH ()-[e]->() WHERE e.active_run_id = $run_id AND e.fact_key IS NOT NULL
DELETE e
"""


def rebuild_projection(
    store: Neo4jStore, ontology: Ontology, *, run_id: str, batch_size: int
) -> dict:
    """Rebuild typed serving edges for one run.

    Args:
        store: 连接句柄。
        ontology: 谓词白名单来源；未声明谓词直接跳过并计入 skipped。
        run_id: 只投影该 run 下 ACCEPTED + POSITIVE 的断言。
        batch_size: UNWIND 批大小。

    Returns:
        {"edges": 写入边数, "predicates": 涉及谓词数, "skipped": 被拒谓词数}

    Side effects: 先删除该 run 的旧投影边，再重建；对 Assertion 层只读。

    Note:
        NEGATIVE 极性的断言不投影为正向边——「不宜与藜芦同用」不能变成
        一条肯定的配伍边。否定事实保留在 Assertion 层供检索侧单独处理。
    """
    def rebuild(tx):
        state = tx.run(
            "MATCH (r:IngestRun {run_id: $run_id}) "
            "SET r.write_lock = coalesce(r.write_lock, 0) + 1 RETURN r.status AS status",
            run_id=run_id,
        ).single()
        if state is None or state["status"] != "BUILDING":
            raise RunStateError("只能重建 BUILDING 批次的投影")
        facts = list(tx.run(_COLLECT_FACTS, run_id=run_id))
        grouped: dict[str, list[dict]] = defaultdict(list)
        skipped = 0
        for fact in facts:
            predicate = fact["predicate"]
            if predicate not in ontology.predicates:
                skipped += 1
                continue
            grouped[predicate].append({**dict(fact), "run_id": run_id})

        tx.run(_DROP_PROJECTION, run_id=run_id).consume()
        written = 0
        for predicate, rows in grouped.items():
            cypher = _PROJECT_TEMPLATE.format(predicate=predicate)
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                tx.run(cypher, batch=batch).consume()
                written += len(batch)
        return {"edges": written, "predicates": len(grouped), "skipped": skipped}

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    with store.session() as session:
        return session.execute_write(rebuild)
