"""Cross-chunk graph expansion for native retrieval modes.

Hit chunks only mention the local neighborhood. Textbook facts for one herb
are often split across pages, so expansion must follow canonical entities
instead of requiring the neighbor to share the same FROM_CHUNK link.
"""

from __future__ import annotations

AUDIT_REL_TYPES: tuple[str, ...] = (
    "FROM_CHUNK",
    "PART_OF",
    "EVIDENCED_BY",
    "IN_RUN",
    "SUBJECT",
    "OBJECT",
)

DEFAULT_SEED_LIMIT = 24
DEFAULT_CONTEXT_CHUNK_LIMIT = 8
HIGH_RISK_EXPAND_PREDICATES: tuple[str, ...] = (
    "INCOMPATIBLE_WITH",
    "CONTRAINDICATED_IN",
    "HAS_DOSAGE",
)


def read_active_run_id(session) -> str | None:
    """读取当前 ACTIVE 的 IngestRun ID。"""
    record = session.run(
        "MATCH (r:IngestRun {status: 'ACTIVE'}) RETURN r.run_id AS run_id LIMIT 1"
    ).single()
    return record["run_id"] if record else None


def merge_extra_chunks(
    ranked_chunks: list[dict],
    extra_chunks: list[dict],
    *,
    limit: int = DEFAULT_CONTEXT_CHUNK_LIMIT,
) -> list[dict]:
    """Append graph-reached chunks that the first-stage retriever missed."""
    merged = list(ranked_chunks)
    seen = {chunk["id"] for chunk in merged if chunk.get("id")}
    for chunk in extra_chunks:
        chunk_id = chunk.get("id")
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        item = dict(chunk)
        item.setdefault("score", 0.15)
        item.setdefault("lexical_score", None)
        item.setdefault("vector_score", None)
        terms = list(item.get("matched_terms") or [])
        if "跨块图谱" not in terms:
            terms.append("跨块图谱")
        item["matched_terms"] = terms
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def _add_node(nodes: dict[str, dict], *, node_id, label, node_type, limit: int) -> bool:
    """把实体加入问答子图节点表。"""
    if not node_id:
        return False
    if node_id in nodes:
        return True
    if len(nodes) >= limit:
        return False
    nodes[node_id] = {"id": node_id, "label": label, "type": node_type}
    return True


def _add_edge(
    nodes: dict[str, dict],
    edges: list[dict],
    triples: list[tuple[str, str, str]],
    *,
    source_id,
    source_name,
    source_type,
    target_id,
    target_name,
    target_type,
    rel_type,
    node_limit: int,
    edge_limit: int,
    triple_limit: int,
) -> None:
    """把关系加入问答子图边表与三元组。"""
    if not target_id or not rel_type:
        return
    if not _add_node(
        nodes, node_id=source_id, label=source_name, node_type=source_type, limit=node_limit
    ):
        return
    if not _add_node(
        nodes, node_id=target_id, label=target_name, node_type=target_type, limit=node_limit
    ):
        return
    edge = {"source": source_id, "target": target_id, "type": rel_type}
    if edge not in edges and len(edges) < edge_limit:
        edges.append(edge)
    triple = (source_name, rel_type, target_name)
    if triple not in triples and len(triples) < triple_limit:
        triples.append(triple)


def _seed_entities(session, chunk_ids: list[str], *, limit: int) -> list[dict]:
    """用检索命中块上的实体作为扩图种子。"""
    return list(
        session.run(
            """
            UNWIND $chunk_ids AS chunk_id
            MATCH (chunk:Chunk) WHERE elementId(chunk) = chunk_id
            MATCH (entity)-[:FROM_CHUNK]->(chunk)
            WHERE NOT entity:Chunk AND NOT entity:Document
              AND NOT entity:Assertion AND NOT entity:IngestRun
            RETURN DISTINCT
                   elementId(entity) AS entity_id,
                   labels(entity)[0] AS entity_label,
                   coalesce(entity.preferred_name, entity.name) AS entity_name
            LIMIT $limit
            """,
            chunk_ids=chunk_ids,
            limit=limit,
        )
    )


def _expand_serving_edges(session, names: list[str], *, run_id: str, limit: int):
    """沿 Serving 投影边扩展邻域。"""
    return list(
        session.run(
            """
            UNWIND $names AS name
            MATCH (entity:Entity)
            WHERE (entity.preferred_name = name OR name IN coalesce(entity.aliases, []))
              AND (entity.run_id = $run OR EXISTS {
                MATCH (entity)-[serving]-()
                WHERE serving.fact_key IS NOT NULL AND serving.active_run_id = $run
              })
            MATCH (entity)-[serving]->(object:Entity)
            WHERE serving.fact_key IS NOT NULL AND serving.active_run_id = $run
            RETURN entity.canonical_id AS source_id,
                   entity.preferred_name AS source_name,
                   entity.entity_type AS source_type,
                   type(serving) AS rel_type,
                   object.canonical_id AS target_id,
                   object.preferred_name AS target_name,
                   object.entity_type AS target_type,
                   serving.evidence_ids AS evidence_ids
            ORDER BY coalesce(serving.best_support_score, 0.0) DESC
            LIMIT $limit
            """,
            names=names,
            run=run_id,
            limit=limit,
        )
    )


def _chunks_for_assertions(session, assertion_ids: list[str], *, known_ids: list[str], limit: int):
    """为断言补上对应教材块文本。"""
    if not assertion_ids:
        return []
    return [
        {
            "id": record["id"],
            "index": record["index"],
            "text": record["text"],
            "page_start": record["page_start"],
            "page_end": record["page_end"],
            "section": record["section"],
        }
        for record in session.run(
            """
            UNWIND $assertion_ids AS assertion_id
            MATCH (assertion:Assertion {assertion_id: assertion_id})
                  -[:EVIDENCED_BY]->(chunk:Chunk)
            WHERE NOT elementId(chunk) IN $known_ids
            RETURN DISTINCT
                   elementId(chunk) AS id, chunk.index AS index, chunk.text AS text,
                   chunk.page_start AS page_start, chunk.page_end AS page_end,
                   chunk.section AS section
            ORDER BY chunk.page_start
            LIMIT $limit
            """,
            assertion_ids=assertion_ids,
            known_ids=known_ids,
            limit=limit,
        )
    ]


def _expand_assertion_edges(session, names: list[str], *, run_id: str, limit: int):
    """Assertion-layer neighbors that serving projection never contains."""
    return list(
        session.run(
            """
            UNWIND $names AS name
            MATCH (entity:Entity)
            WHERE (entity.preferred_name = name OR name IN coalesce(entity.aliases, []))
            MATCH (assertion:Assertion {run_id: $run, validation_status: 'ACCEPTED'})
                  -[:SUBJECT]->(entity)
            WHERE assertion.predicate IN $predicates
            OPTIONAL MATCH (assertion)-[:OBJECT]->(object:Entity)
            RETURN entity.canonical_id AS source_id,
                   entity.preferred_name AS source_name,
                   entity.entity_type AS source_type,
                   assertion.predicate AS rel_type,
                   coalesce(object.canonical_id, assertion.assertion_id) AS target_id,
                   coalesce(object.preferred_name, assertion.predicate) AS target_name,
                   coalesce(object.entity_type, 'VALUE') AS target_type,
                   assertion.assertion_id AS assertion_id
            ORDER BY coalesce(assertion.final_score, 0.0) DESC
            LIMIT $limit
            """,
            names=names,
            run=run_id,
            predicates=list(HIGH_RISK_EXPAND_PREDICATES),
            limit=limit,
        )
    )


def _expand_legacy_edges(session, chunk_ids: list[str], *, limit: int):
    """旧图无 Serving 边时按名称扩邻域。"""
    return list(
        session.run(
            """
            UNWIND $chunk_ids AS chunk_id
            MATCH (chunk:Chunk) WHERE elementId(chunk) = chunk_id
            MATCH (entity)-[:FROM_CHUNK]->(chunk)
            WHERE NOT entity:Chunk AND NOT entity:Document
              AND NOT entity:Assertion AND NOT entity:IngestRun
            MATCH (entity)-[relationship]-(neighbor)
            WHERE NOT neighbor:Chunk AND NOT neighbor:Document
              AND NOT neighbor:Assertion AND NOT neighbor:IngestRun
              AND NOT type(relationship) IN $audit
            RETURN DISTINCT
                   elementId(entity) AS source_id,
                   labels(entity)[0] AS source_label,
                   coalesce(entity.preferred_name, entity.name) AS source_name,
                   elementId(neighbor) AS target_id,
                   labels(neighbor)[0] AS target_label,
                   coalesce(neighbor.preferred_name, neighbor.name) AS target_name,
                   type(relationship) AS rel_type
            LIMIT $limit
            """,
            chunk_ids=chunk_ids,
            audit=list(AUDIT_REL_TYPES),
            limit=limit,
        )
    )


def _chunks_for_entities(session, entity_ids: list[str], *, known_ids: list[str], limit: int):
    """按实体取出相关文本块。"""
    if not entity_ids:
        return []
    return [
        {
            "id": record["id"],
            "index": record["index"],
            "text": record["text"],
            "page_start": record["page_start"],
            "page_end": record["page_end"],
            "section": record["section"],
        }
        for record in session.run(
            """
            UNWIND $entity_ids AS entity_id
            MATCH (entity) WHERE elementId(entity) = entity_id
            MATCH (entity)-[:FROM_CHUNK]->(chunk:Chunk)
            WHERE NOT elementId(chunk) IN $known_ids
            RETURN DISTINCT
                   elementId(chunk) AS id, chunk.index AS index, chunk.text AS text,
                   chunk.page_start AS page_start, chunk.page_end AS page_end,
                   chunk.section AS section
            ORDER BY chunk.page_start
            LIMIT $limit
            """,
            entity_ids=entity_ids,
            known_ids=known_ids,
            limit=limit,
        )
    ]


def expand_cross_chunk_neighborhood(
    session,
    chunk_ids: list[str],
    *,
    run_id: str | None,
    node_limit: int,
    edge_limit: int,
    triple_limit: int,
    extra_chunk_limit: int,
    seed_limit: int = DEFAULT_SEED_LIMIT,
    seed_names: list[str] | None = None,
) -> tuple[list[dict], list[dict], list[tuple[str, str, str]], list[dict]]:
    """Follow entities out of the hit chunks and collect their other evidence pages."""
    if not chunk_ids and not seed_names:
        return [], [], [], []

    seeds = _seed_entities(session, chunk_ids, limit=seed_limit) if chunk_ids else []
    chunk_names = [row["entity_name"] for row in seeds if row.get("entity_name")]
    asked_names = [name for name in (seed_names or []) if name]
    # 问句已锚定药名时，只沿这些实体扩图，避免块上挂着的「功效/防风/头虱」
    # 把子图画成别的药。
    names = list(dict.fromkeys(asked_names or chunk_names))
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    triples: list[tuple[str, str, str]] = []
    extra_chunks: list[dict] = []

    serving_rows = []
    if run_id and names:
        serving_rows = _expand_serving_edges(
            session, names[:seed_limit], run_id=run_id, limit=edge_limit
        )
    for row in serving_rows:
        _add_edge(
            nodes,
            edges,
            triples,
            source_id=row["source_id"],
            source_name=row["source_name"],
            source_type=row["source_type"],
            target_id=row["target_id"],
            target_name=row["target_name"],
            target_type=row["target_type"],
            rel_type=row["rel_type"],
            node_limit=node_limit,
            edge_limit=edge_limit,
            triple_limit=triple_limit,
        )

    evidence_ids = [
        evidence_id
        for row in serving_rows
        for evidence_id in (row.get("evidence_ids") or [])
        if evidence_id
    ]
    extra_chunks.extend(
        _chunks_for_assertions(
            session,
            list(dict.fromkeys(evidence_ids)),
            known_ids=chunk_ids,
            limit=extra_chunk_limit,
        )
    )

    assertion_rows = []
    if run_id and names:
        assertion_rows = _expand_assertion_edges(
            session, names[:seed_limit], run_id=run_id, limit=edge_limit
        )
    assertion_ids: list[str] = []
    for row in assertion_rows:
        _add_edge(
            nodes,
            edges,
            triples,
            source_id=row["source_id"],
            source_name=row["source_name"],
            source_type=row["source_type"],
            target_id=row["target_id"],
            target_name=row["target_name"],
            target_type=row["target_type"],
            rel_type=row["rel_type"],
            node_limit=node_limit,
            edge_limit=edge_limit,
            triple_limit=triple_limit,
        )
        if row.get("assertion_id"):
            assertion_ids.append(row["assertion_id"])
    extra_chunks.extend(
        _chunks_for_assertions(
            session,
            list(dict.fromkeys(assertion_ids)),
            known_ids=chunk_ids + [chunk["id"] for chunk in extra_chunks if chunk.get("id")],
            limit=extra_chunk_limit,
        )
    )

    if not triples:
        legacy_rows = _expand_legacy_edges(session, chunk_ids, limit=edge_limit)
        neighbor_ids: list[str] = []
        for row in legacy_rows:
            _add_edge(
                nodes,
                edges,
                triples,
                source_id=row["source_id"],
                source_name=row["source_name"],
                source_type=row["source_label"],
                target_id=row["target_id"],
                target_name=row["target_name"],
                target_type=row["target_label"],
                rel_type=row["rel_type"],
                node_limit=node_limit,
                edge_limit=edge_limit,
                triple_limit=triple_limit,
            )
            if row.get("target_id"):
                neighbor_ids.append(row["target_id"])
        extra_chunks.extend(
            _chunks_for_entities(
                session,
                list(dict.fromkeys(neighbor_ids)),
                known_ids=chunk_ids,
                limit=extra_chunk_limit,
            )
        )

    return list(nodes.values()), edges, triples, extra_chunks
