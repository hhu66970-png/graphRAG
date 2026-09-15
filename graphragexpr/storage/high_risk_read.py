"""三类高风险证据的专用读取入口（任务书 §4）。

为什么不能复用服务图读取
------------------------
`serving_projection` 只投影「ACCEPTED + POSITIVE + 两端都是 Entity」的断言
（见 `serving_projection.py` 的 `_COLLECT_FACTS`）。这带来两个后果：

1. `HAS_DOSAGE` 的宾语是值对象，**永远不进服务图**，剂量事实在普通读取里消失；
2. 否定极性的禁忌断言被过滤，而教材里禁忌最典型的写法恰恰带否定词
   （「不宜与藜芦同用」）。

于是普通读取既读不到剂量，也读不到禁忌。本模块直接读 **Assertion 事实源层**，
并按任务书要求：

- 剂量读取**不要求宾语必须是实体节点**；
- 保留剂量为值对象，不为让普通图查询工作而创建共享剂量实体；
- 参数化查询 + 限定 run + 限定结果数，避免全库扫描；
- 同时返回原始极性、校验状态、限定条件、原文、来源块、文档与页码，
  使上层能够沿 `EVIDENCED_BY` 一路回溯（SKILL §3.1）。

只读模块，不写入任何数据。
"""

from __future__ import annotations

import json

from ..domain.high_risk import TASK_HIGH_RISK_PREDICATES
from .neo4j_store import Neo4jStore

# 只读校验通过的断言。待复核／拒绝的抽取结果**不能**被冒充为通过的事实
# （任务书 §4）；它们所对应的教材原文仍可经文本证据通道独立核验后使用。
_LOAD_ASSERTIONS = """
MATCH (a:Assertion {run_id: $run})
WHERE a.predicate IN $predicates
  AND a.validation_status = 'ACCEPTED'
  AND ($subject_ids IS NULL OR EXISTS {
        MATCH (a)-[:SUBJECT]->(subject:Entity)
        WHERE subject.canonical_id IN $subject_ids
      })
OPTIONAL MATCH (a)-[:SUBJECT]->(subject:Entity)
OPTIONAL MATCH (a)-[:OBJECT]->(object:Entity)
OPTIONAL MATCH (a)-[:EVIDENCED_BY]->(chunk:Chunk)
WITH a, subject, object, collect(DISTINCT chunk) AS chunks
RETURN a.assertion_id        AS assertion_id,
       a.fact_key            AS fact_key,
       a.predicate           AS predicate,
       a.polarity            AS polarity,
       a.qualifiers_json     AS qualifiers_json,
       a.object_value_json   AS object_value_json,
       a.evidence_quote      AS evidence_quote,
       a.evidence_start      AS evidence_start,
       a.evidence_end        AS evidence_end,
       a.match_mode          AS match_mode,
       a.validation_status   AS validation_status,
       a.is_cross_chunk      AS is_cross_chunk,
       a.final_score         AS final_score,
       a.page_start          AS page_start,
       a.page_end            AS page_end,
       subject.canonical_id  AS subject_id,
       subject.preferred_name AS subject_name,
       object.canonical_id   AS object_id,
       object.preferred_name AS object_name,
       [chunk IN chunks | chunk.chunk_id]     AS chunk_ids,
       [chunk IN chunks | chunk.text]         AS chunk_texts,
       [chunk IN chunks | chunk.page_start]   AS chunk_page_starts,
       [chunk IN chunks | chunk.page_end]     AS chunk_page_ends,
       [chunk IN chunks | chunk.section_path] AS chunk_sections,
       [chunk IN chunks | chunk.document_id]  AS chunk_document_ids
ORDER BY a.final_score DESC, a.assertion_id
LIMIT $limit
"""

# 高风险证据数量上限：这是「限定结果数」的具体落点，
# 不是为了性能而牺牲完整性——超出上限时上层会记录覆盖限制。
DEFAULT_ASSERTION_LIMIT = 60

# 关联断言读取：用于反向／交集题，从「给定人群、病证或配伍对象」出发找候选药物。
_LOAD_ASSERTIONS_BY_OBJECT = """
MATCH (a:Assertion {run_id: $run})
WHERE a.predicate IN $predicates
  AND a.validation_status = 'ACCEPTED'
  AND (
    EXISTS {
      MATCH (a)-[:OBJECT]->(object:Entity)
      WHERE object.canonical_id IN $object_ids
    }
    OR any(keyword IN $object_keywords
           WHERE a.evidence_quote CONTAINS keyword)
  )
MATCH (a)-[:SUBJECT]->(subject:Entity)
OPTIONAL MATCH (a)-[:EVIDENCED_BY]->(chunk:Chunk)
WITH a, subject, collect(DISTINCT chunk) AS chunks
RETURN a.assertion_id        AS assertion_id,
       a.fact_key            AS fact_key,
       a.predicate           AS predicate,
       a.polarity            AS polarity,
       a.object_value_json   AS object_value_json,
       a.evidence_quote      AS evidence_quote,
       a.validation_status   AS validation_status,
       a.final_score         AS final_score,
       subject.canonical_id  AS subject_id,
       subject.preferred_name AS subject_name,
       [chunk IN chunks | chunk.chunk_id]   AS chunk_ids,
       [chunk IN chunks | chunk.text]       AS chunk_texts,
       [chunk IN chunks | chunk.page_start] AS chunk_page_starts,
       [chunk IN chunks | chunk.page_end]   AS chunk_page_ends
ORDER BY a.final_score DESC, a.assertion_id
LIMIT $limit
"""

# 按证据 ID 精确回表：核验阶段必须能拿回「最终选中的那几条」的原文与页码。
_LOAD_BY_ASSERTION_IDS = """
UNWIND $assertion_ids AS assertion_id
MATCH (a:Assertion {assertion_id: assertion_id})
OPTIONAL MATCH (a)-[:SUBJECT]->(subject:Entity)
OPTIONAL MATCH (a)-[:OBJECT]->(object:Entity)
OPTIONAL MATCH (a)-[:EVIDENCED_BY]->(chunk:Chunk)
WITH a, subject, object, collect(DISTINCT chunk) AS chunks
RETURN a.assertion_id      AS assertion_id,
       a.predicate         AS predicate,
       a.polarity          AS polarity,
       a.object_value_json AS object_value_json,
       a.validation_status AS validation_status,
       a.page_start        AS page_start,
       a.page_end          AS page_end,
       subject.preferred_name AS subject_name,
       object.preferred_name  AS object_name,
       [chunk IN chunks | chunk.chunk_id]   AS chunk_ids,
       [chunk IN chunks | chunk.text]       AS chunk_texts,
       [chunk IN chunks | chunk.page_start] AS chunk_page_starts,
       [chunk IN chunks | chunk.page_end]   AS chunk_page_ends
"""

# 文本证据通道：向量／全文检索命中原文后，经此适配进入同一证据处理流程。
_LOAD_CHUNKS = """
UNWIND $chunk_ids AS chunk_id
MATCH (chunk:Chunk {chunk_id: chunk_id})
RETURN chunk.chunk_id     AS chunk_id,
       chunk.text         AS text,
       chunk.page_start   AS page_start,
       chunk.page_end     AS page_end,
       chunk.section_path AS section_path,
       chunk.document_id  AS document_id,
       chunk.entry_id     AS entry_id
"""


def _decode_json_column(row: dict, column: str, target: str) -> dict:
    """把 Neo4j 中的 JSON 字符串列解码成对象。"""
    raw = row.pop(column, None)
    if raw is None:
        row[target] = {}
        return row
    if isinstance(raw, dict):
        row[target] = raw
        return row
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        value = {}
    row[target] = value if isinstance(value, dict) else {}
    return row


def load_high_risk_assertions(
    store: Neo4jStore,
    *,
    run_id: str,
    subject_ids: list[str] | None = None,
    predicates: tuple[str, ...] = TASK_HIGH_RISK_PREDICATES,
    limit: int = DEFAULT_ASSERTION_LIMIT,
) -> list[dict]:
    """Read the three high-risk predicates from the assertion layer.

    Args:
        store: 连接句柄。
        run_id: 只读该批次的断言；不跨批次混合，避免把已下线的数据当成现役事实。
        subject_ids: 限定主语药物；None 表示不限定（用于反向／交集题）。
        predicates: 谓词白名单，默认本任务的三类。
        limit: 结果上限，防止全库扫描。

    Returns:
        断言行列表。每行含原始极性、校验状态、值对象、原文、来源块与页码。
        剂量断言的 `object_id` 通常为 None——这是设计使然，值对象不建实体身份。

    Side effects: 只读。
    """
    rows = store.run(
        _LOAD_ASSERTIONS,
        run=run_id,
        predicates=list(predicates),
        subject_ids=list(subject_ids) if subject_ids else None,
        limit=max(1, int(limit)),
    )
    for row in rows:
        _decode_json_column(row, "object_value_json", "object_value")
        _decode_json_column(row, "qualifiers_json", "qualifiers")
    return rows


def load_high_risk_assertions_by_object(
    store: Neo4jStore,
    *,
    run_id: str,
    object_ids: list[str] | None = None,
    object_keywords: list[str] | None = None,
    predicates: tuple[str, ...] = TASK_HIGH_RISK_PREDICATES,
    limit: int = DEFAULT_ASSERTION_LIMIT,
) -> list[dict]:
    """Find high-risk assertions whose object (or evidence text) matches a constraint.

    用于反向／交集题：「孕妇禁用的药有哪些」。注意返回的是**候选**——
    是否列为候选取决于是否存在明确支持的风险证据，而不是「没查到禁忌」，
    后者绝不能反过来被解读为安全（任务书 §3）。

    Side effects: 只读。`object_keywords` 只做原文包含匹配，不构成事实判定。
    """
    if not object_ids and not object_keywords:
        return []
    rows = store.run(
        _LOAD_ASSERTIONS_BY_OBJECT,
        run=run_id,
        predicates=list(predicates),
        object_ids=list(object_ids or []),
        object_keywords=list(object_keywords or []),
        limit=max(1, int(limit)),
    )
    for row in rows:
        _decode_json_column(row, "object_value_json", "object_value")
    return rows


def load_assertions_by_ids(
    store: Neo4jStore, assertion_ids: list[str]
) -> list[dict]:
    """Re-read the exact assertions selected for the final answer.

    核验与渲染只允许使用**最终选中的证据**，因此需要按 ID 精确回表，
    而不是复用检索阶段的宽集合（任务书 §6）。
    """
    unique = list(dict.fromkeys(item for item in assertion_ids if item))
    if not unique:
        return []
    rows = store.run(_LOAD_BY_ASSERTION_IDS, assertion_ids=unique)
    for row in rows:
        _decode_json_column(row, "object_value_json", "object_value")
    return rows


_LOAD_EDGES_BETWEEN = """
MATCH (a:Assertion {run_id: $run})
WHERE a.predicate IN $predicates
  AND a.validation_status = 'ACCEPTED'
MATCH (a)-[:SUBJECT]->(subject:Entity)
MATCH (a)-[:OBJECT]->(object:Entity)
WHERE subject.canonical_id IN $endpoint_ids
  AND object.canonical_id IN $endpoint_ids
  AND subject.canonical_id <> object.canonical_id
OPTIONAL MATCH (a)-[:EVIDENCED_BY]->(chunk:Chunk)
WITH a, subject, object, collect(DISTINCT chunk) AS chunks
RETURN a.assertion_id        AS assertion_id,
       a.fact_key            AS fact_key,
       a.predicate           AS predicate,
       a.evidence_quote      AS evidence_quote,
       a.final_score         AS final_score,
       a.page_start          AS page_start,
       a.page_end            AS page_end,
       subject.canonical_id  AS subject_id,
       subject.preferred_name AS subject_name,
       object.canonical_id   AS object_id,
       object.preferred_name AS object_name,
       object.entity_type    AS object_type,
       [chunk IN chunks | chunk.chunk_id] AS chunk_ids
ORDER BY a.final_score DESC, a.assertion_id
LIMIT $limit
"""


def load_assertion_edges_between(
    store: Neo4jStore,
    *,
    run_id: str,
    endpoint_ids: list[str],
    predicates: tuple[str, ...] = TASK_HIGH_RISK_PREDICATES,
    limit: int = 40,
) -> list[dict]:
    """Direct assertion-layer edges whose both ends are the given entities.

    Serving paths cannot see NEGATIVE incompatibilities. This is the 1-hop
    fallback for questions such as 甘草与芫花十八反.
    """
    unique = list(dict.fromkeys(item for item in endpoint_ids if item))
    if len(unique) < 2:
        return []
    return store.run(
        _LOAD_EDGES_BETWEEN,
        run=run_id,
        predicates=list(predicates),
        endpoint_ids=unique,
        limit=max(1, int(limit)),
    )


def load_chunks(store: Neo4jStore, chunk_ids: list[str]) -> list[dict]:
    """Load chunk text for the text-evidence channel (向量／全文命中的原文)."""
    unique = list(dict.fromkeys(item for item in chunk_ids if item))
    if not unique:
        return []
    return store.run(_LOAD_CHUNKS, chunk_ids=unique)
