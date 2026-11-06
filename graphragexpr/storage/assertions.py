"""Assertion / Evidence graph writes — the fact source.

事实源是 Assertion，而不是实体间直接边。每条 ACCEPTED 断言都必须带
chunk_ids、页码与可定位 evidence_quote；同一事实可以有多条断言/证据，
冲突事实不互相覆盖。

所有写入按 UNWIND 批量执行且幂等。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from ..domain.models import AssertionRecord, ChunkRecord, EntityRecord
from .neo4j_store import Neo4jStore

_UPSERT_ENTITIES = """
UNWIND $batch AS row
MERGE (e:Entity {canonical_id: row.canonical_id})
SET e.entity_type      = row.entity_type,
    e.preferred_name   = row.preferred_name,
    e.normalized_name  = row.normalized_name,
    e.aliases          = row.aliases,
    e.source_authority = row.source_authority,
    e.link_confidence  = row.link_confidence,
    e.link_method      = row.link_method,
    e.review_status    = row.review_status,
    e.run_id           = row.run_id
"""

_UPSERT_CHUNKS = """
UNWIND $batch AS row
MERGE (c:Chunk {chunk_id: row.chunk_id})
SET c.document_id      = row.document_id,
    c.parent_chunk_id  = row.parent_chunk_id,
    c.entry_id         = row.entry_id,
    c.section_path     = row.section_path,
    c.order_in_entry   = row.order_in_entry,
    c.text             = row.text,
    c.normalized_text  = row.normalized_text,
    c.page_start       = row.page_start,
    c.page_end         = row.page_end,
    c.content_hash     = row.content_hash,
    c.ocr_quality      = row.ocr_quality,
    c.run_id           = row.run_id
WITH c, row
MATCH (d:Document {document_id: row.document_id})
MERGE (c)-[:PART_OF]->(d)
"""

_UPSERT_ASSERTIONS = """
UNWIND $batch AS row
MERGE (a:Assertion {assertion_id: row.assertion_id})
SET a.fact_key          = row.fact_key,
    a.predicate         = row.predicate,
    a.polarity          = row.polarity,
    a.qualifiers_json   = row.qualifiers_json,
    a.object_value_json = row.object_value_json,
    a.evidence_quote    = row.evidence_quote,
    a.evidence_start    = row.evidence_start,
    a.evidence_end      = row.evidence_end,
    a.match_mode        = row.match_mode,
    a.page_start        = row.page_start,
    a.page_end          = row.page_end,
    a.rule_score        = row.rule_score,
    a.final_score       = row.final_score,
    a.validation_status = row.validation_status,
    a.prompt_version    = row.prompt_version,
    a.schema_version    = row.schema_version,
    a.run_id            = row.run_id,
    a.is_cross_chunk    = row.is_cross_chunk,
    a.created_at        = row.created_at
WITH a, row
MATCH (s:Entity {canonical_id: row.subject_id})
MERGE (a)-[:SUBJECT]->(s)
WITH a, row
CALL (a, row) {
    WITH a, row
    WHERE row.object_id IS NOT NULL
    MATCH (o:Entity {canonical_id: row.object_id})
    MERGE (a)-[:OBJECT]->(o)
}
WITH a, row
UNWIND row.chunk_ids AS cid
MATCH (c:Chunk {chunk_id: cid})
MERGE (a)-[:EVIDENCED_BY]->(c)
WITH a, row
MATCH (r:IngestRun {run_id: row.run_id})
MERGE (a)-[:IN_RUN]->(r)
"""


def upsert_document(store: Neo4jStore, *, document_id: str, name: str, source: str, run_id: str) -> None:
    """写入或更新 Document 节点。"""
    store.run(
        """
        MERGE (d:Document {document_id: $document_id})
        SET d.name = $name, d.source = $source, d.run_id = $run_id
        """,
        document_id=document_id, name=name, source=source, run_id=run_id,
    )


def upsert_entities(
    store: Neo4jStore, entities: Sequence[EntityRecord], *, run_id: str, batch_size: int
) -> int:
    """批量写入实体节点。"""
    rows = [
        {
            "canonical_id": e.canonical_id,
            "entity_type": e.entity_type,
            "preferred_name": e.preferred_name,
            "normalized_name": e.normalized_name,
            "aliases": list(e.aliases),
            "source_authority": e.source_authority,
            "link_confidence": e.link_confidence,
            "link_method": e.link_method,
            "review_status": e.review_status,
            "run_id": run_id,
        }
        for e in entities
    ]
    return store.execute_batches(_UPSERT_ENTITIES, rows, batch_size=batch_size)


def upsert_chunks(
    store: Neo4jStore, chunks: Sequence[ChunkRecord], *, run_id: str, batch_size: int
) -> int:
    """批量写入文本块节点。"""
    rows = [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "parent_chunk_id": c.parent_chunk_id,
            "entry_id": c.entry_id,
            "section_path": list(c.section_path),
            "order_in_entry": c.order_in_entry,
            "text": c.text,
            "normalized_text": c.normalized_text,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "content_hash": c.content_hash,
            "ocr_quality": c.ocr_quality,
            "run_id": run_id,
        }
        for c in chunks
    ]
    return store.execute_batches(_UPSERT_CHUNKS, rows, batch_size=batch_size)


def upsert_assertions(
    store: Neo4jStore, assertions: Sequence[AssertionRecord], *, batch_size: int
) -> int:
    """Write assertions. ACCEPTED ones are re-checked for evidence before writing."""
    import json

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for a in assertions:
        a.assert_publishable()  # 硬门槛
        rows.append(
            {
                "assertion_id": a.assertion_id,
                "fact_key": a.fact_key,
                "subject_id": a.subject_id,
                "object_id": a.object_id,
                "predicate": a.predicate,
                "polarity": a.polarity,
                "qualifiers_json": json.dumps(a.qualifiers, ensure_ascii=False, sort_keys=True),
                "object_value_json": (
                    json.dumps(a.object_value, ensure_ascii=False, sort_keys=True)
                    if a.object_value else None
                ),
                "evidence_quote": a.evidence_quote,
                "evidence_start": a.evidence_start,
                "evidence_end": a.evidence_end,
                "match_mode": a.match_mode,
                "chunk_ids": list(a.chunk_ids),
                "page_start": a.page_start,
                "page_end": a.page_end,
                "rule_score": a.rule_score,
                "final_score": a.final_score,
                "validation_status": a.validation_status,
                "prompt_version": a.prompt_version,
                "schema_version": a.schema_version,
                "run_id": a.run_id,
                "is_cross_chunk": a.is_cross_chunk,
                "created_at": now,
            }
        )
    return store.execute_batches(_UPSERT_ASSERTIONS, rows, batch_size=batch_size)


def quality_report(store: Neo4jStore, run_id: str) -> dict:
    """Compute the publication gate metrics for one run."""
    rows = store.run(
        """
        MATCH (a:Assertion {run_id: $run_id})
        WITH count(a) AS total,
             sum(CASE WHEN a.validation_status = 'ACCEPTED' THEN 1 ELSE 0 END) AS accepted,
             sum(CASE WHEN a.validation_status = 'ACCEPTED'
                       AND a.evidence_quote IS NOT NULL
                       AND a.match_mode <> 'not_found' THEN 1 ELSE 0 END) AS accepted_with_evidence,
             sum(CASE WHEN a.validation_status = 'REVIEW' THEN 1 ELSE 0 END) AS review,
             sum(CASE WHEN a.validation_status = 'REJECTED' THEN 1 ELSE 0 END) AS rejected,
             sum(CASE WHEN a.is_cross_chunk THEN 1 ELSE 0 END) AS cross_chunk
        RETURN total, accepted, accepted_with_evidence, review, rejected, cross_chunk
        """,
        run_id=run_id,
    )
    row = rows[0] if rows else {}
    accepted = row.get("accepted", 0) or 0
    with_evidence = row.get("accepted_with_evidence", 0) or 0
    return {
        "total_assertions": row.get("total", 0) or 0,
        "accepted": accepted,
        "review": row.get("review", 0) or 0,
        "rejected": row.get("rejected", 0) or 0,
        "cross_chunk": row.get("cross_chunk", 0) or 0,
        "evidence_coverage": (with_evidence / accepted) if accepted else 1.0,
    }
