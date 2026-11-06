"""Typed retrieval orchestration producing EvidencePacket values."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from ..config.settings import RetrievalSettings
from ..domain.models import EvidenceItem, EvidencePacket
from ..domain.ontology import Ontology
from ..domain.high_risk import TASK_HIGH_RISK_PREDICATES
from ..storage import serving_read
from ..storage.high_risk_read import (
    load_assertion_edges_between,
    load_high_risk_assertions,
)
from ..storage.neo4j_store import Neo4jStore
from .comparison import reduce_section_summaries
from .models import RetrievalPlan, RetrievalResult
from .router import route_query

FallbackRetriever = Callable[[str], tuple[list[dict], dict, list[tuple[str, str, str]]]]


def _query_id(question: str) -> str:
    """为一次检索生成查询 ID。"""
    normalized = "".join(question.split())
    return f"query:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:20]}"


def _evidence_items(
    rows: list[dict],
    *,
    retrieval_scores: dict[str, float] | None = None,
    path_by_evidence: dict[str, list[dict]] | None = None,
) -> list[EvidenceItem]:
    """把存储行转成 EvidencePacket 条目。"""
    scores = retrieval_scores or {}
    paths = path_by_evidence or {}
    items: list[EvidenceItem] = []
    seen: set[str] = set()
    for row in rows:
        evidence_id = str(row.get("assertion_id") or row.get("evidence_id") or "")
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        chunk_ids = row.get("chunk_ids") or []
        page_start = int(row.get("page_start") or 0)
        page_end = int(row.get("page_end") or page_start)
        qualifiers = dict(row.get("qualifiers") or {})
        if row.get("subject_name"):
            qualifiers.setdefault("subject_name", row["subject_name"])
        if row.get("object_name"):
            qualifiers.setdefault("object_name", row["object_name"])
        if chunk_ids:
            qualifiers.setdefault("chunk_ids", list(chunk_ids))
        items.append(
            EvidenceItem(
                evidence_id=evidence_id,
                claim_key=row.get("fact_key"),
                subject_id=row.get("subject_id"),
                predicate=row.get("predicate"),
                object_id=row.get("object_id"),
                path=paths.get(evidence_id),
                evidence_quote=str(row.get("quote") or ""),
                chunk_id=str(chunk_ids[0] if chunk_ids else ""),
                page_start=page_start,
                page_end=page_end,
                support_score=float(row.get("support_score") or 0.0),
                retrieval_score=float(
                    scores.get(evidence_id, row.get("retrieval_score", 1.0))
                ),
                qualifiers=qualifiers,
            )
        )
    return items


def _load_evidence_for_ids(store: Neo4jStore, evidence_ids: list[str]) -> list[dict]:
    """按证据 ID 批量加载原文。"""
    return serving_read.load_evidence(store, list(dict.fromkeys(evidence_ids)))


def _apply_token_budget(
    items: list[EvidenceItem], token_budget: int
) -> tuple[list[EvidenceItem], int]:
    """Keep whole evidence quotes within a conservative one-character/token budget.

    Round-robin by subject so one herb's pages cannot crowd out the others.
    """
    buckets: dict[str, list[EvidenceItem]] = {}
    order: list[str] = []
    for item in items:
        key = str(item.subject_id or item.chunk_id or item.evidence_id)
        if key not in buckets:
            order.append(key)
            buckets[key] = []
        buckets[key].append(item)
    selected: list[EvidenceItem] = []
    used = 0
    pending = True
    while pending:
        pending = False
        for key in order:
            if not buckets[key]:
                continue
            item = buckets[key][0]
            estimated = len(item.evidence_quote) + 32
            if selected and used + estimated > token_budget:
                buckets[key].pop(0)
                pending = True
                continue
            buckets[key].pop(0)
            selected.append(item)
            used += estimated
            pending = True
    return selected, used


def _path_score(path: dict) -> float:
    """给一条受限路径打检索分。"""
    edge_scores = [float(score or 0.0) for score in path.get("edge_scores") or []]
    groups = path.get("evidence_id_groups") or []
    edge_count = max(len(path.get("predicates") or []), 1)
    min_support = min(edge_scores, default=0.0)
    evidence_coverage = sum(bool(group) for group in groups) / edge_count
    length_decay = 1.0 / edge_count
    # PathRAG-inspired ranking. Endpoints and predicates have
    # already passed exact entity linking and the ontology whitelist.
    return round(
        0.25
        + 0.25 * min_support
        + 0.20 * evidence_coverage
        + 0.15
        + 0.15 * length_decay,
        6,
    )


def _is_incompat_plan(plan: RetrievalPlan) -> bool:
    """判断当前计划是否在查配伍禁忌。"""
    return plan.allowed_predicates == ("INCOMPATIBLE_WITH",)


def _high_risk_predicates(plan: RetrievalPlan) -> tuple[str, ...]:
    """取出检索侧关心的高风险谓词集合。"""
    return tuple(
        predicate
        for predicate in plan.allowed_predicates
        if predicate in TASK_HIGH_RISK_PREDICATES
    )


def _map_assertion_rows(rows: list[dict]) -> list[dict]:
    """把断言查询行映射成内部事实结构。"""
    mapped: list[dict] = []
    for row in rows:
        mapped.append(
            {
                **row,
                "quote": row.get("quote") or row.get("evidence_quote") or "",
                "support_score": float(
                    row.get("support_score") or row.get("final_score") or 0.0
                ),
                "object_type": row.get("object_type") or "ENTITY",
            }
        )
    return mapped


def _load_assertion_facts(
    store: Neo4jStore,
    *,
    run_id: str,
    subject_ids: list[str],
    predicates: tuple[str, ...],
) -> list[dict]:
    """按实体与谓词加载断言层事实。"""
    if not subject_ids or not predicates:
        return []
    return _map_assertion_rows(
        load_high_risk_assertions(
            store,
            run_id=run_id,
            subject_ids=subject_ids,
            predicates=predicates,
        )
    )


def _retrieve_incompatibilities(
    store: Neo4jStore,
    linked: list[dict],
    plan: RetrievalPlan,
    run_id: str,
) -> tuple[list[EvidenceItem], dict, dict, list[str]]:
    """Read INCOMPATIBLE_WITH from the assertion layer, not the serving graph.

    Serving projection drops NEGATIVE polarity, so「不宜同用」never appears as
    a COMBINED_WITH neighbour. Multi-herb questions must list every subject and
    say explicitly when there is no shared incompatible partner.
    """
    herb_ids = list(plan.linked_entity_ids)
    if not herb_ids:
        return [], {"nodes": [], "edges": []}, {}, ["未链接到中药实体"]
    herb_by_id = {}
    for item in linked:
        canonical_id = item.get("canonical_id")
        if canonical_id in herb_ids:
            herb_by_id[canonical_id] = (
                item.get("name") or item.get("preferred_name") or canonical_id
            )
    rows = load_high_risk_assertions(
        store,
        run_id=run_id,
        subject_ids=herb_ids,
        predicates=("INCOMPATIBLE_WITH",),
    )
    evidence_rows = []
    objects_by_subject: dict[str, set[str]] = {herb_id: set() for herb_id in herb_ids}
    for row in rows:
        evidence_rows.append(
            {
                **row,
                "quote": row.get("evidence_quote") or "",
                "support_score": float(row.get("final_score") or 0.0),
                "object_type": row.get("object_type") or "HERB",
            }
        )
        subject_id = row.get("subject_id")
        object_name = row.get("object_name")
        if subject_id in objects_by_subject and object_name:
            objects_by_subject[subject_id].add(object_name)
    items = _evidence_items(evidence_rows)
    warnings: list[str] = []
    missing = [
        herb_by_id.get(herb_id, herb_id)
        for herb_id in herb_ids
        if not objects_by_subject.get(herb_id)
    ]
    if missing:
        warnings.append(
            "以下中药未检索到配伍禁忌（INCOMPATIBLE_WITH）断言："
            + "、".join(missing)
        )
    shared: list[str] = []
    if len(herb_ids) >= 2:
        present = [herb_id for herb_id in herb_ids if objects_by_subject.get(herb_id)]
        if len(present) == len(herb_ids):
            shared = sorted(
                set.intersection(*(objects_by_subject[herb_id] for herb_id in herb_ids))
            )
        if shared:
            warnings.append("共同不宜同用对象：" + "、".join(shared))
        else:
            warnings.append(
                "现有证据中没有两药共同的不宜同用对象；"
                "不得把单味药的禁忌推广为两药共有"
            )
    subgraph_rows = [row for row in evidence_rows if row.get("object_id")]
    linked_subjects = [
        item for item in linked if item.get("canonical_id") in set(herb_ids)
    ]
    return (
        items,
        _fact_rows_to_subgraph(subgraph_rows, linked_subjects),
        {
            "incompatibility_by_herb": {
                herb_by_id.get(herb_id, herb_id): sorted(objects_by_subject[herb_id])
                for herb_id in herb_ids
            },
            "shared_incompatible_objects": shared,
            "herbs_without_incompatibility": missing,
        },
        warnings,
    )


def _retrieve_attr(
    store: Neo4jStore,
    ontology: Ontology,
    linked: list[dict],
    plan: RetrievalPlan,
    run_id: str,
) -> tuple[list[EvidenceItem], dict, dict, list[str]]:
    """单味药属性卡片检索。"""
    planned_id = plan.linked_entity_ids[0] if plan.linked_entity_ids else None
    herb = next(
        (
            item
            for item in linked
            if planned_id and item.get("canonical_id") == planned_id
        ),
        None,
    ) or next((item for item in linked if item.get("entity_type") == "HERB"), None)
    if not herb:
        return [], {"nodes": [], "edges": []}, {}, ["未链接到中药实体"]
    card = serving_read.get_herb_card(store, ontology, herb["name"], run_id=run_id)
    items: list[EvidenceItem] = []
    subgraph = {"nodes": [], "edges": []}
    selected_slots: list[str] = []
    if card is not None and not card.is_empty():
        slots = [
            slot for slot in card.slots if slot.predicate in plan.allowed_predicates
        ]
        selected_slots = [slot.predicate for slot in slots]
        evidence_ids = [
            evidence_id for slot in slots for evidence_id in slot.evidence_ids
        ]
        items = _evidence_items(_load_evidence_for_ids(store, evidence_ids))
        subgraph = card.subgraph
    risk_predicates = _high_risk_predicates(plan)
    extra_rows = _load_assertion_facts(
        store,
        run_id=run_id,
        subject_ids=[str(herb.get("canonical_id") or "")],
        predicates=risk_predicates,
    )
    extra_items = _evidence_items(extra_rows)
    seen = {item.evidence_id for item in items}
    items.extend(item for item in extra_items if item.evidence_id not in seen)
    if extra_rows:
        subgraph = _merge_subgraphs(
            subgraph, _fact_rows_to_subgraph(extra_rows, [herb])
        )
    if not items:
        return (
            [],
            subgraph,
            {},
            [f"{herb['name']} 没有 ACTIVE 属性事实"],
        )
    trace = {
        "herb_id": (card.canonical_id if card else herb.get("canonical_id")),
        "selected_slots": selected_slots,
        "assertion_predicates": list(risk_predicates),
        "assertion_items": len(extra_items),
    }
    return items, subgraph, trace, []


def _retrieve_reverse(
    store: Neo4jStore,
    linked: list[dict],
    run_id: str,
    settings: RetrievalSettings,
) -> tuple[list[EvidenceItem], dict, dict, list[str]]:
    """按功效/主治反向查药。"""
    target = next(
        (
            item
            for item in linked
            if item.get("entity_type") in {"EFFICACY", "INDICATION"}
        ),
        None,
    )
    if not target:
        return [], {"nodes": [], "edges": []}, {}, ["未链接到功效或主治实体"]
    rows = serving_read.find_herbs_by_effect(
        store,
        target["name"],
        run_id=run_id,
        limit=settings.directed_result_limit,
    )
    evidence_ids = [item for row in rows for item in (row.get("evidence_ids") or [])]
    scores = {
        evidence_id: float(row.get("score") or 0.0)
        for row in rows
        for evidence_id in (row.get("evidence_ids") or [])
    }
    evidence = _load_evidence_for_ids(store, evidence_ids)
    items = _evidence_items(evidence, retrieval_scores=scores)
    subgraph = _rows_to_subgraph(rows, target=target)
    return items, subgraph, {"ranked_herbs": [row["herb_id"] for row in rows]}, []


def _retrieve_intersection(
    store: Neo4jStore,
    linked: list[dict],
    plan: RetrievalPlan,
    run_id: str,
    settings: RetrievalSettings,
) -> tuple[list[EvidenceItem], dict, dict, list[str]]:
    """多约束硬交集检索。"""
    names_by_id = {
        str(item.get("canonical_id")): item.get("name") or item.get("preferred_name")
        for item in linked
        if item.get("canonical_id")
    }
    constraint_names = [
        str(names_by_id.get(entity_id) or "")
        for entity_id in plan.linked_entity_ids
    ]
    rows = serving_read.find_herbs_by_constraints(
        store,
        list(plan.linked_entity_ids),
        run_id=run_id,
        limit=settings.directed_result_limit,
        constraint_names=constraint_names,
    )
    evidence_ids = [item for row in rows for item in (row.get("evidence_ids") or [])]
    scores = {
        evidence_id: float(row["coverage"])
        for row in rows
        for evidence_id in (row.get("evidence_ids") or [])
    }
    items = _evidence_items(
        _load_evidence_for_ids(store, evidence_ids), retrieval_scores=scores
    )
    exact = [row["herb_id"] for row in rows if row["exact_match"]]
    partial = [row["herb_id"] for row in rows if not row["exact_match"]]
    matrix = [
        {
            "herb_id": row["herb_id"],
            "herb": row["herb"],
            "matched_constraints": row.get("matched_names") or [],
            "coverage": row["coverage"],
            "exact_match": row["exact_match"],
        }
        for row in rows
    ]
    warnings = [] if exact else ["没有中药满足全部约束；返回结果均为部分匹配"]
    return (
        items,
        _constraint_rows_to_subgraph(rows, list(plan.linked_entity_ids)),
        {
            "exact_matches": exact,
            "partial_matches": partial,
            "constraint_matrix": matrix,
        },
        warnings,
    )


def _retrieve_compare(
    store: Neo4jStore,
    linked: list[dict],
    plan: RetrievalPlan,
    run_id: str,
    question: str,
    settings: RetrievalSettings,
) -> tuple[list[EvidenceItem], dict, dict, list[str]]:
    """多药对比检索与章节摘要。"""
    entity_ids = list(plan.linked_entity_ids[:4])
    if len(entity_ids) < 2:
        return [], {"nodes": [], "edges": []}, {}, ["比较问题需要至少两个已链接实体"]
    rows = serving_read.load_entity_facts(
        store, entity_ids, run_id=run_id, limit=settings.comparison_fact_limit
    )
    allowed = set(plan.allowed_predicates)
    rows = [row for row in rows if row.get("predicate") in allowed]
    extra_rows = _load_assertion_facts(
        store,
        run_id=run_id,
        subject_ids=entity_ids,
        predicates=_high_risk_predicates(plan),
    )
    seen_ids = {
        str(row.get("assertion_id") or row.get("evidence_id") or "") for row in rows
    }
    for row in extra_rows:
        evidence_id = str(row.get("assertion_id") or "")
        if evidence_id and evidence_id not in seen_ids:
            rows.append(row)
            seen_ids.add(evidence_id)
    section_rows = serving_read.load_entity_sections(
        store, entity_ids, run_id=run_id, limit=settings.comparison_chunk_limit
    )
    summaries = reduce_section_summaries(
        question,
        section_rows,
        per_entity=settings.comparison_per_entity,
        total_limit=settings.comparison_total,
    )
    items = _evidence_items(rows) + _evidence_items(summaries)
    slots: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        object_name = row.get("object_name")
        if not object_name:
            object_name = (row.get("quote") or row.get("evidence_quote") or "")[:24]
        slots.setdefault(row["predicate"], {}).setdefault(row["subject_id"], []).append(
            object_name
        )
    return (
        items,
        _fact_rows_to_subgraph(rows, linked),
        {
            "comparison_slots": slots,
            "map_entities": entity_ids,
            "reduce": "predicate_alignment_and_balanced_section_merge",
            "section_summary_items": len(summaries),
        },
        [] if items else ["比较对象没有共同的 ACTIVE 类型化事实"],
    )


def _retrieve_path(
    store: Neo4jStore,
    plan: RetrievalPlan,
    run_id: str,
    settings: RetrievalSettings,
) -> tuple[list[EvidenceItem], dict, dict, list[str]]:
    """白名单关系上的受限多跳路径。"""
    if len(plan.linked_entity_ids) < 2:
        return [], {"nodes": [], "edges": []}, {}, ["路径问题需要至少两个已链接实体"]
    paths = serving_read.find_supported_paths(
        store,
        plan.linked_entity_ids[0],
        plan.linked_entity_ids[1],
        run_id=run_id,
        allowed_predicates=list(plan.allowed_predicates),
        max_hops=plan.max_hops,
        limit=settings.path_limit,
    )
    assertion_paths = _assertion_paths_between(
        store, plan, run_id, settings.path_limit
    )
    paths = sorted(paths + assertion_paths, key=_path_score, reverse=True)
    evidence_ids: list[str] = []
    score_by_evidence: dict[str, float] = {}
    path_by_evidence: dict[str, list[dict]] = {}
    for path in paths:
        score = _path_score(path)
        path_steps = _path_steps(path)
        for group in path.get("evidence_id_groups") or []:
            for evidence_id in group or []:
                evidence_ids.append(evidence_id)
                score_by_evidence[evidence_id] = max(
                    score, score_by_evidence.get(evidence_id, 0.0)
                )
                path_by_evidence.setdefault(evidence_id, path_steps)
    items = _evidence_items(
        _load_evidence_for_ids(store, evidence_ids),
        retrieval_scores=score_by_evidence,
        path_by_evidence=path_by_evidence,
    )
    return (
        items,
        _paths_to_subgraph(paths),
        {
            "path_count": len(paths),
            "path_scores": [_path_score(path) for path in paths],
        },
        [] if paths else ["领域关系白名单内未找到三跳以内的证据路径"],
    )


def _retrieve_fallback(
    question: str,
    fallback_retriever: FallbackRetriever | None,
) -> tuple[list[EvidenceItem], dict, dict, list[str]]:
    """总论/兜底：走文本块检索。"""
    if fallback_retriever is None:
        return [], {"nodes": [], "edges": []}, {}, ["未配置文本块兜底检索器"]
    chunks, subgraph, _triples = fallback_retriever(question)
    items = [
        EvidenceItem(
            evidence_id=f"chunk:{chunk['id']}",
            evidence_quote=str(chunk.get("text") or ""),
            chunk_id=str(chunk.get("id") or chunk.get("index") or ""),
            page_start=int(chunk.get("page_start") or 0),
            page_end=int(chunk.get("page_end") or chunk.get("page_start") or 0),
            retrieval_score=float(chunk.get("score") or 0.0),
            support_score=0.0,
            qualifiers={"matched_terms": chunk.get("matched_terms") or []},
        )
        for chunk in chunks
    ]
    return items, subgraph, {"chunk_count": len(chunks)}, []


def retrieve(
    store: Neo4jStore,
    ontology: Ontology,
    question: str,
    *,
    run_id: str | None = None,
    fallback_retriever: FallbackRetriever | None = None,
    token_budget: int | None = None,
    settings: RetrievalSettings | None = None,
) -> RetrievalResult:
    """Link, route and retrieve one question without generating natural language."""
    settings = settings or RetrievalSettings()
    active_run_id = run_id or serving_read.active_run(store)
    if not active_run_id:
        raise RuntimeError("没有 ACTIVE 的 IngestRun")
    linked = serving_read.link_entities_in_text(
        store,
        question,
        run_id=active_run_id,
        types=tuple(ontology.entity_types),
        limit=settings.entity_link_limit,
    )
    plan = route_query(question, linked, ontology=ontology, settings=settings)
    handlers: dict[
        str, Callable[[], tuple[list[EvidenceItem], dict, dict, list[str]]]
    ] = {
        "attr": lambda: (
            _retrieve_incompatibilities(store, linked, plan, active_run_id)
            if _is_incompat_plan(plan)
            else _retrieve_attr(store, ontology, linked, plan, active_run_id)
        ),
        "reverse": lambda: _retrieve_reverse(store, linked, active_run_id, settings),
        "intersect": lambda: _retrieve_intersection(
            store, linked, plan, active_run_id, settings
        ),
        "compare": lambda: (
            _retrieve_incompatibilities(store, linked, plan, active_run_id)
            if _is_incompat_plan(plan)
            else _retrieve_compare(
                store, linked, plan, active_run_id, question, settings
            )
        ),
        "path": lambda: _retrieve_path(store, plan, active_run_id, settings),
        "fallback": lambda: _retrieve_fallback(question, fallback_retriever),
    }
    items, subgraph, details, warnings = handlers[plan.intent]()
    effective_budget = (
        token_budget if token_budget is not None else settings.token_budget
    )
    items_before_budget = len(items)
    items, estimated_tokens = _apply_token_budget(items, effective_budget)
    if len(items) < items_before_budget:
        warnings.append(
            f"证据按 token 预算裁剪：保留 {len(items)}/{items_before_budget} 项"
        )
    if plan.is_high_risk:
        warnings.append("高风险问题：检索结果必须保留逐字证据，证据不足时生成层应拒答")
    packet = EvidencePacket(
        query_id=_query_id(question),
        intent=plan.intent,
        linked_entities=linked,
        structure_type=plan.structure_type,
        items=items,
        warnings=warnings,
        token_budget=effective_budget,
        trace={
            "run_id": active_run_id,
            "route": {
                "intent": plan.intent,
                "reason": plan.reason,
                "confidence": plan.confidence,
                "allowed_predicates": list(plan.allowed_predicates),
                "max_hops": plan.max_hops,
                "require_provenance": plan.require_provenance,
                "is_high_risk": plan.is_high_risk,
            },
            "retrieved_items": len(items),
            "items_before_budget": items_before_budget,
            "estimated_evidence_tokens": estimated_tokens,
            **details,
        },
    )
    return RetrievalResult(packet=packet, subgraph=subgraph)


def _rows_to_subgraph(rows: list[dict], *, target: dict) -> dict:
    """把查询行转成前端子图。"""
    nodes = {
        target["canonical_id"]: {
            "id": target["canonical_id"],
            "label": target["name"],
            "type": target["entity_type"],
        }
    }
    edges = []
    for row in rows:
        nodes[row["herb_id"]] = {
            "id": row["herb_id"],
            "label": row["herb"],
            "type": "HERB",
        }
        edges.append(
            {
                "source": row["herb_id"],
                "target": target["canonical_id"],
                "type": row["predicate"],
            }
        )
    return {"nodes": list(nodes.values()), "edges": edges}


def _constraint_rows_to_subgraph(rows: list[dict], constraint_ids: list[str]) -> dict:
    """把交集约束行转成子图。"""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    for row in rows:
        nodes[row["herb_id"]] = {
            "id": row["herb_id"],
            "label": row["herb"],
            "type": "HERB",
        }
        for match in row.get("matches") or []:
            target_id = match["target_id"]
            nodes[target_id] = {
                "id": target_id,
                "label": match["target_name"],
                "type": match["target_type"],
            }
            edges.append(
                {
                    "source": row["herb_id"],
                    "target": target_id,
                    "type": match["predicate"],
                }
            )
    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "constraints": constraint_ids,
    }


def _merge_subgraphs(left: dict, right: dict) -> dict:
    """合并多路子图节点与边。"""
    nodes = {node["id"]: node for node in (left.get("nodes") or []) if node.get("id")}
    edges = list(left.get("edges") or [])
    for node in right.get("nodes") or []:
        if node.get("id"):
            nodes.setdefault(node["id"], node)
    for edge in right.get("edges") or []:
        if edge not in edges:
            edges.append(edge)
    return {"nodes": list(nodes.values()), "edges": edges}


def _assertion_paths_between(
    store: Neo4jStore,
    plan: RetrievalPlan,
    run_id: str,
    limit: int,
) -> list[dict]:
    """1-hop assertion paths that serving projection cannot see (NEGATIVE 十八反)."""
    endpoints = list(plan.linked_entity_ids[:4])
    if len(endpoints) < 2:
        return []
    rows = load_assertion_edges_between(
        store,
        run_id=run_id,
        endpoint_ids=endpoints,
        predicates=_high_risk_predicates(plan) or TASK_HIGH_RISK_PREDICATES,
        limit=limit,
    )
    paths: list[dict] = []
    for row in rows:
        subject_id = row.get("subject_id")
        object_id = row.get("object_id")
        if not subject_id or not object_id:
            continue
        paths.append(
            {
                "nodes": [
                    {
                        "id": subject_id,
                        "name": row.get("subject_name") or subject_id,
                        "type": "HERB",
                    },
                    {
                        "id": object_id,
                        "name": row.get("object_name") or object_id,
                        "type": row.get("object_type") or "HERB",
                    },
                ],
                "predicates": [row["predicate"]],
                "edge_source_ids": [subject_id],
                "edge_target_ids": [object_id],
                "evidence_id_groups": [[row["assertion_id"]]],
                "edge_scores": [float(row.get("final_score") or 0.0)],
            }
        )
    return paths


def _fact_rows_to_subgraph(rows: list[dict], linked: list[dict]) -> dict:
    """把断言事实行转成子图。"""
    nodes = {
        item["canonical_id"]: {
            "id": item["canonical_id"],
            "label": item.get("name") or item.get("preferred_name"),
            "type": item["entity_type"],
        }
        for item in linked
        if item.get("canonical_id")
    }
    edges: list[dict] = []
    for row in rows:
        object_id = row.get("object_id")
        if not object_id:
            continue
        nodes[object_id] = {
            "id": object_id,
            "label": row.get("object_name") or object_id,
            "type": row.get("object_type") or "ENTITY",
        }
        edge = {
            "source": row["subject_id"],
            "target": object_id,
            "type": row["predicate"],
        }
        if edge not in edges:
            edges.append(edge)
    return {"nodes": list(nodes.values()), "edges": edges}


def _path_steps(path: dict) -> list[dict]:
    """把路径拆成逐步关系。"""
    nodes = path.get("nodes") or []
    nodes_by_id = {node["id"]: node for node in nodes}
    predicates = path.get("predicates") or []
    source_ids = path.get("edge_source_ids") or []
    target_ids = path.get("edge_target_ids") or []
    if len(source_ids) == len(predicates) == len(target_ids):
        return [
            {
                "source": nodes_by_id[source_id],
                "predicate": predicate,
                "target": nodes_by_id[target_id],
            }
            for source_id, predicate, target_id in zip(
                source_ids, predicates, target_ids, strict=True
            )
        ]
    return [
        {"source": nodes[index], "predicate": predicate, "target": nodes[index + 1]}
        for index, predicate in enumerate(predicates)
        if index + 1 < len(nodes)
    ]


def _paths_to_subgraph(paths: list[dict]) -> dict:
    """把多条路径画成子图。"""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    for path in paths:
        path_nodes = path.get("nodes") or []
        for node in path_nodes:
            nodes[node["id"]] = {
                "id": node["id"],
                "label": node["name"],
                "type": node["type"],
            }
        for step in _path_steps(path):
            edge = {
                "source": step["source"]["id"],
                "target": step["target"]["id"],
                "type": step["predicate"],
            }
            if edge not in edges:
                edges.append(edge)
    return {"nodes": list(nodes.values()), "edges": edges}
