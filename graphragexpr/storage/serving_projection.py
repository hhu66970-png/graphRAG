"""Typed serving projection rebuilt from the assertion layer (设计 §4.2).

检索侧需要「快」，所以把 `ACTIVE + ACCEPTED` 的断言投影成类型化直接边；
但审计、版本、冲突与多来源仍由 Assertion 层承担。Serving Edge 是**可重建的派生物**，
不是事实源——这正是建图文档与检索文档冲突的化解方式。

关系类型只能来自本体白名单：先用 Ontology 校验，再按谓词分组拼接语句
（SKILL §9.2 允许的唯一动态拼接方式），杜绝把模型输出直接拼进 Cypher。
"""

from __future__ import annotations

from collections import defaultdict

from ..domain.ontology import Ontology
from .neo4j_store import Neo4jStore

# 每条 Serving Edge 携带的证据摘要（设计 §4.2 边属性）
_PROJECT_TEMPLATE = """
UNWIND $batch AS row
MATCH (s:Entity {{canonical_id: row.subject_id}})
MATCH (o:Entity {{canonical_id: row.object_id}})
MERGE (s)-[e:`{predicate}` {{fact_key: row.fact_key}}]->(o)
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
    facts = store.run(_COLLECT_FACTS, run_id=run_id)

    grouped: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    for fact in facts:
        predicate = fact["predicate"]
        if predicate not in ontology.predicates:
            skipped += 1
            continue
        grouped[predicate].append({**fact, "run_id": run_id})

    store.run(_DROP_PROJECTION, run_id=run_id)

    written = 0
    for predicate, rows in grouped.items():
        cypher = _PROJECT_TEMPLATE.format(predicate=predicate)
        written += store.execute_batches(cypher, rows, batch_size=batch_size)

    return {"edges": written, "predicates": len(grouped), "skipped": skipped}
