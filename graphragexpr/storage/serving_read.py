"""Read access over the typed serving projection (SKILL §5.5).

本模块只负责「把 ACTIVE run 的类型化边读成结构化结果」，**不生成自然语言答案、
不做意图路由**——那属于 retrieval / generation 层（SKILL §5.6 / §5.8）。
API 层可以直接编排这些函数来演示 v2 图谱，而不必复制任何检索逻辑。

每个返回项都带 `page_start/page_end` 与 `evidence_ids`，使调用方能一路回溯到
Assertion 层的逐字原句（SKILL §3.1 证据优先）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..domain.ontology import Ontology, load_ontology
from .neo4j_store import Neo4jStore
from .runs import get_active_run

# 属性卡片的槽位顺序：与教材条目的呈现顺序一致，便于人读
CARD_SLOT_ORDER: tuple[str, ...] = (
    "HAS_NATURE_FLAVOR",
    "ENTERS_MERIDIAN",
    "HAS_EFFICACY",
    "TREATS",
    "HAS_DOSAGE",
    "CONTRAINDICATED_IN",
    "INCOMPATIBLE_WITH",
    "COMBINED_WITH",
    "DERIVED_FROM",
    "PROCESSED_AS",
    "CONTAINS_COMPOUND",
    "BELONGS_TO_CATEGORY",
    "RECORDED_IN",
)

_FIND_HERB = """
MATCH (e:Entity {entity_type: 'HERB'})
WHERE (e.preferred_name = $name OR $name IN coalesce(e.aliases, []))
  AND (e.run_id = $run OR EXISTS {
    MATCH (e)-[serving]-()
    WHERE serving.fact_key IS NOT NULL AND serving.active_run_id = $run
  })
RETURN e.canonical_id AS canonical_id, e.preferred_name AS name, e.aliases AS aliases
LIMIT 1
"""

_HERB_CARD = """
MATCH (h:Entity {canonical_id: $herb_id})-[e]->(o:Entity)
WHERE e.fact_key IS NOT NULL AND e.active_run_id = $run
RETURN type(e) AS predicate, o.preferred_name AS value, o.canonical_id AS object_id,
       e.page_start AS page_start, e.page_end AS page_end,
       e.evidence_count AS evidence_count, e.best_support_score AS score,
       e.evidence_ids AS evidence_ids
ORDER BY e.best_support_score DESC
"""

_REVERSE = """
MATCH (target:Entity)
WHERE target.entity_type IN ['EFFICACY', 'INDICATION']
  AND (target.preferred_name = $name OR $name IN coalesce(target.aliases, []))
MATCH (h:Entity {entity_type: 'HERB'})-[e]->(target)
WHERE e.fact_key IS NOT NULL AND e.active_run_id = $run
RETURN h.preferred_name AS herb, h.canonical_id AS herb_id, type(e) AS predicate,
       e.evidence_count AS evidence_count, e.page_start AS page_start,
       e.page_end AS page_end, e.best_support_score AS score,
       e.evidence_ids AS evidence_ids, e.fact_key AS fact_key,
       target.canonical_id AS target_id
ORDER BY e.evidence_count DESC, e.best_support_score DESC
LIMIT $limit
"""

_EVIDENCE = """
MATCH (a:Assertion) WHERE a.assertion_id IN $ids
OPTIONAL MATCH (a)-[:SUBJECT]->(subject:Entity)
OPTIONAL MATCH (a)-[:OBJECT]->(object:Entity)
OPTIONAL MATCH (a)-[:EVIDENCED_BY]->(chunk:Chunk)
WITH a, subject, object, collect(DISTINCT chunk.chunk_id) AS chunk_ids
RETURN a.assertion_id AS assertion_id, a.evidence_quote AS quote,
       a.page_start AS page_start, a.page_end AS page_end,
       a.match_mode AS match_mode, a.predicate AS predicate,
       a.fact_key AS fact_key, subject.canonical_id AS subject_id,
       object.canonical_id AS object_id, chunk_ids,
       a.final_score AS support_score, a.qualifiers_json AS qualifiers_json,
       subject.preferred_name AS subject_name,
       object.preferred_name AS object_name
ORDER BY a.final_score DESC
"""

_INTERSECTION = """
UNWIND $constraint_ids AS constraint_id
MATCH (target:Entity {canonical_id: constraint_id})
MATCH (h:Entity {entity_type: 'HERB'})-[e]->(target)
WHERE e.fact_key IS NOT NULL AND e.active_run_id = $run
  AND type(e) IN ['HAS_EFFICACY', 'TREATS']
WITH h,
     collect(DISTINCT {
       target_id: target.canonical_id, target_name: target.preferred_name,
       target_type: target.entity_type, predicate: type(e)
     }) AS matches,
     collect(DISTINCT e) AS edges
WITH h, matches,
     [match IN matches | match.target_id] AS matched_ids,
     [match IN matches | match.target_name] AS matched_names,
     reduce(ids = [], edge IN edges | ids + coalesce(edge.evidence_ids, [])) AS evidence_ids,
     [edge IN edges WHERE edge.page_start IS NOT NULL | edge.page_start] AS pages,
     reduce(best = 0.0, edge IN edges |
       CASE WHEN coalesce(edge.best_support_score, 0.0) > best
            THEN edge.best_support_score ELSE best END) AS best_score
RETURN h.preferred_name AS herb, h.canonical_id AS herb_id,
       matches, matched_ids, matched_names, evidence_ids, pages, best_score
ORDER BY size(matched_ids) DESC, best_score DESC, herb
LIMIT $limit
"""

_INTERSECTION_BY_NAME = """
UNWIND $names AS name
MATCH (target:Entity)
WHERE target.entity_type IN ['EFFICACY', 'INDICATION']
  AND target.preferred_name IS NOT NULL
  AND size(name) >= 2
  AND (
    target.preferred_name CONTAINS name
    OR (size(target.preferred_name) >= 2 AND name CONTAINS target.preferred_name)
  )
MATCH (h:Entity {entity_type: 'HERB'})-[e]->(target)
WHERE e.fact_key IS NOT NULL AND e.active_run_id = $run
  AND type(e) IN ['HAS_EFFICACY', 'TREATS']
WITH h,
     collect(DISTINCT {
       target_id: target.canonical_id, target_name: target.preferred_name,
       target_type: target.entity_type, predicate: type(e)
     }) AS matches,
     collect(DISTINCT e) AS edges
WITH h, matches,
     [match IN matches | match.target_id] AS matched_ids,
     [match IN matches | match.target_name] AS matched_names,
     reduce(ids = [], edge IN edges | ids + coalesce(edge.evidence_ids, [])) AS evidence_ids,
     [edge IN edges WHERE edge.page_start IS NOT NULL | edge.page_start] AS pages,
     reduce(best = 0.0, edge IN edges |
       CASE WHEN coalesce(edge.best_support_score, 0.0) > best
            THEN edge.best_support_score ELSE best END) AS best_score
RETURN h.preferred_name AS herb, h.canonical_id AS herb_id,
       matches, matched_ids, matched_names, evidence_ids, pages, best_score
ORDER BY size(matched_ids) DESC, best_score DESC, herb
LIMIT $limit
"""

_ENTITY_FACTS = """
UNWIND $entity_ids AS entity_id
MATCH (subject:Entity {canonical_id: entity_id})-[edge]->(object:Entity)
WHERE edge.fact_key IS NOT NULL AND edge.active_run_id = $run
UNWIND coalesce(edge.evidence_ids, []) AS evidence_id
MATCH (a:Assertion {assertion_id: evidence_id})
OPTIONAL MATCH (a)-[:EVIDENCED_BY]->(chunk:Chunk)
WITH subject, edge, object, a, collect(DISTINCT chunk) AS chunks
RETURN subject.canonical_id AS subject_id, subject.preferred_name AS subject_name,
       type(edge) AS predicate, object.canonical_id AS object_id,
       object.preferred_name AS object_name, object.entity_type AS object_type,
       edge.fact_key AS fact_key,
       a.assertion_id AS evidence_id, a.evidence_quote AS quote,
       a.page_start AS page_start, a.page_end AS page_end,
       [chunk IN chunks | chunk.chunk_id] AS chunk_ids,
       a.final_score AS support_score,
       a.qualifiers_json AS qualifiers_json,
       head(chunks).section_path AS section_path
ORDER BY subject_id, predicate, support_score DESC
LIMIT $limit
"""

_ENTITY_SECTIONS = """
UNWIND $entity_ids AS entity_id
MATCH (entity:Entity {canonical_id: entity_id})
MATCH (assertion:Assertion {run_id: $run})-[:SUBJECT]->(entity)
WHERE assertion.validation_status = 'ACCEPTED'
MATCH (assertion)-[:EVIDENCED_BY]->(chunk:Chunk)
RETURN DISTINCT entity.canonical_id AS entity_id,
       entity.preferred_name AS entity_name, chunk.chunk_id AS chunk_id,
       chunk.text AS text, chunk.page_start AS page_start,
       chunk.page_end AS page_end, chunk.section_path AS section_path
LIMIT $limit
"""

_SUBGRAPH = """
MATCH (h:Entity {canonical_id: $herb_id})-[e]->(o:Entity)
WHERE e.fact_key IS NOT NULL AND e.active_run_id = $run
RETURN elementId(h) AS source_id, h.preferred_name AS source_name,
       h.entity_type AS source_type,
       elementId(o) AS target_id, o.preferred_name AS target_name,
       o.entity_type AS target_type, type(e) AS rel_type
LIMIT $limit
"""


@dataclass
class CardSlot:
    """One attribute slot of a herb card."""

    predicate: str
    display: str
    values: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    evidence_count: int = 0


@dataclass
class HerbCard:
    """HerbCard：本文件定义的数据结构或组件。"""
    canonical_id: str
    name: str
    aliases: list[str]
    slots: list[CardSlot]
    subgraph: dict

    def is_empty(self) -> bool:
        """属性卡片是否没有任何槽位。"""
        return not self.slots


def active_run(store: Neo4jStore) -> str | None:
    """Serving 读取时使用的活跃 run_id。"""
    return get_active_run(store)


def find_herb(store: Neo4jStore, name: str, *, run_id: str) -> dict | None:
    """Resolve a herb display name (or alias) to its canonical entity."""
    rows = store.run(_FIND_HERB, name=name.strip(), run=run_id)
    return rows[0] if rows else None


def get_herb_card(
    store: Neo4jStore,
    ontology: Ontology,
    name: str,
    *,
    run_id: str,
    max_values_per_slot: int = 12,
    subgraph_limit: int = 80,
) -> HerbCard | None:
    """Build a typed attribute card for one herb from the serving projection.

    Args:
        store: 连接句柄。
        ontology: 用于取谓词中文展示名。
        name: 药名或别名。
        run_id: 只读该 run 的投影边。
        max_values_per_slot: 每个属性槽保留的取值数（PathRAG 式剪枝，避免噪声淹没）。
        subgraph_limit: 返回给前端高亮的边数上限。

    Returns:
        HerbCard；找不到该中药时返回 None。

    Side effects: 只读。
    """
    herb = find_herb(store, name, run_id=run_id)
    if not herb:
        return None

    rows = store.run(_HERB_CARD, herb_id=herb["canonical_id"], run=run_id)
    grouped: dict[str, CardSlot] = {}
    for row in rows:
        predicate = row["predicate"]
        display = (
            ontology.predicate(predicate).display
            if predicate in ontology.predicates
            else predicate
        )
        slot = grouped.setdefault(predicate, CardSlot(predicate=predicate, display=display))
        if row["value"] and row["value"] not in slot.values:
            slot.values.append(row["value"])
        if row["page_start"] and row["page_start"] not in slot.pages:
            slot.pages.append(row["page_start"])
        slot.evidence_ids.extend(row["evidence_ids"] or [])
        slot.evidence_count += row["evidence_count"] or 0

    ordered: list[CardSlot] = []
    for predicate in CARD_SLOT_ORDER:
        slot = grouped.pop(predicate, None)
        if slot:
            slot.values = slot.values[:max_values_per_slot]
            slot.pages = sorted(slot.pages)[:8]
            slot.evidence_ids = slot.evidence_ids[:16]
            ordered.append(slot)
    ordered.extend(grouped.values())  # 本体新增谓词也不丢

    edges = store.run(_SUBGRAPH, herb_id=herb["canonical_id"], run=run_id, limit=subgraph_limit)
    nodes: dict[str, dict] = {}
    subgraph_edges: list[dict] = []
    for row in edges:
        for key in ("source", "target"):
            nid = row[f"{key}_id"]
            nodes.setdefault(nid, {
                "id": nid, "label": row[f"{key}_name"], "type": row[f"{key}_type"],
            })
        subgraph_edges.append({
            "source": row["source_id"], "target": row["target_id"], "type": row["rel_type"],
        })

    return HerbCard(
        canonical_id=herb["canonical_id"],
        name=herb["name"],
        aliases=herb["aliases"] or [],
        slots=ordered,
        subgraph={"nodes": list(nodes.values()), "edges": subgraph_edges},
    )


# 排序说明：中药优先于功效/主治，再按名称长度降序取最长匹配。
# 若只按长度排，「人参的性味、归经和功效」里的「性味」「归经」「功效」与「人参」
# 同为 2 字，会把真正的中药挤出 LIMIT。
_LINK_IN_TEXT = """
MATCH (e:Entity)
WHERE e.entity_type IN $types
  AND any(surface IN [e.preferred_name] + coalesce(e.aliases, [])
          WHERE surface IS NOT NULL AND size(surface) >= 2 AND $text CONTAINS surface)
  AND (e.run_id = $run OR EXISTS {
    MATCH (e)-[serving]-()
    WHERE serving.fact_key IS NOT NULL AND serving.active_run_id = $run
  })
RETURN e.canonical_id AS canonical_id, e.preferred_name AS name,
       e.entity_type AS entity_type, size(e.preferred_name) AS len,
       CASE e.entity_type WHEN 'HERB' THEN 0 ELSE 1 END AS type_rank
ORDER BY type_rank ASC, len DESC
LIMIT $limit
"""


_LONGER_ENTITY_IN_TEXT = """
MATCH (e:Entity)
WHERE e.preferred_name IS NOT NULL
  AND (e.run_id = $run OR EXISTS {
    MATCH (e)-[serving]-()
    WHERE serving.fact_key IS NOT NULL AND serving.active_run_id = $run
  })
  AND any(surface IN [e.preferred_name] + coalesce(e.aliases, [])
          WHERE surface IS NOT NULL
            AND size(surface) > size($name)
            AND surface CONTAINS $name
            AND $text CONTAINS surface)
RETURN e.canonical_id AS canonical_id, e.entity_type AS entity_type,
       e.preferred_name AS preferred_name
ORDER BY size(e.preferred_name) DESC
LIMIT $limit
"""


def find_longer_entity_mentions(
    store: Neo4jStore, text: str, *, name: str, run_id: str, limit: int = 5,
) -> list[dict]:
    """问题里是否出现了**比 `name` 更长、且包含 `name`** 的已知实体名。

    用途有限且明确：高风险服务层核对「用户指定的对象是否已被**完整**识别」。
    链接器按「问题文本包含实体名」匹配，于是「安宫牛黄丸」会把子串「牛黄」
    当成问题的主语，接着用牛黄的禁忌去回答安宫牛黄丸的问题。

    本函数只报告「存在更长匹配」这一事实，**不重做消歧、不改动链接结果、
    不返回替代主语**——是否拒绝使用该证据由服务层决定。
    未收录的复方名（如本体里没有的「龙胆泻肝丸」）查不到更长匹配，
    这类对象错配由适用性核验的**条件**检查兜住。

    Side effects: 只读。
    """
    return store.run(
        _LONGER_ENTITY_IN_TEXT, text=text, name=name, run=run_id, limit=limit
    )


def link_entities_in_text(
    store: Neo4jStore, text: str, *, run_id: str,
    types: tuple[str, ...] = ("HERB", "EFFICACY", "INDICATION"), limit: int = 8,
) -> list[dict]:
    """Longest-match entity linking against the ACTIVE run's canonical entities.

    仅用于 API 层的演示编排；正式的实体链接与意图路由属 retrieval 层（SKILL §5.6）。
    按名称长度降序返回，调用方取第一个即为最长匹配。
    """
    return store.run(
        _LINK_IN_TEXT, text=text, run=run_id, types=list(types), limit=limit
    )


def find_herbs_by_effect(
    store: Neo4jStore, name: str, *, run_id: str, limit: int = 25
) -> list[dict]:
    """Reverse lookup: which herbs have this efficacy / treat this indication."""
    return store.run(_REVERSE, name=name.strip(), run=run_id, limit=limit)


def _name_covers_constraint(target_name: str, constraint_name: str) -> bool:
    """判断实体名是否覆盖交集约束词。"""
    if not target_name or not constraint_name:
        return False
    return constraint_name in target_name or target_name in constraint_name


def _remap_compound_intersection(
    rows: list[dict],
    constraint_ids: list[str],
    constraint_names: list[str],
    limit: int,
) -> list[dict]:
    """Map compound efficacy nodes (补气利水) back onto asked atomic constraints."""
    remapped: list[dict] = []
    for row in rows:
        matched_ids: list[str] = []
        matched_names: list[str] = []
        matches: list[dict] = []
        for match in row.get("matches") or []:
            target_name = match.get("target_name") or ""
            for constraint_id, constraint_name in zip(
                constraint_ids, constraint_names, strict=False
            ):
                if constraint_id in matched_ids:
                    continue
                if not _name_covers_constraint(target_name, constraint_name):
                    continue
                matched_ids.append(constraint_id)
                matched_names.append(constraint_name)
                matches.append(dict(match))
        if not matched_ids:
            continue
        item = dict(row)
        item["matched_ids"] = matched_ids
        item["matched_names"] = matched_names
        item["matches"] = matches
        item["coverage"] = len(matched_ids) / len(constraint_ids)
        item["exact_match"] = len(matched_ids) == len(constraint_ids)
        remapped.append(item)
    remapped.sort(
        key=lambda item: (
            -float(item.get("coverage") or 0.0),
            -float(item.get("best_score") or 0.0),
            item.get("herb") or "",
        )
    )
    return remapped[:limit]


def find_herbs_by_constraints(
    store: Neo4jStore,
    constraint_ids: list[str],
    *,
    run_id: str,
    limit: int = 25,
    constraint_names: list[str] | None = None,
) -> list[dict]:
    """Find herbs matching all constraints first, followed by partial coverage.

    Exact canonical_id intersection is preferred. If that yields no full match
    and display names are provided, a second pass treats compound efficacy
    nodes (e.g. 补气利水) as covering the asked atomic names (补气, 利水).
    Partial rows stay marked and are never promoted to asserted facts.
    """
    unique_ids = list(dict.fromkeys(item for item in constraint_ids if item))
    if len(unique_ids) < 2:
        raise ValueError("at least two different constraint ids are required")
    rows = store.run(
        _INTERSECTION,
        constraint_ids=unique_ids,
        constraint_count=len(unique_ids),
        run=run_id,
        limit=limit,
    )
    for row in rows:
        matched = len(set(row.get("matched_ids") or []))
        row["coverage"] = matched / len(unique_ids)
        row["exact_match"] = matched == len(unique_ids)
    names = [name for name in (constraint_names or []) if name]
    if names and not any(row.get("exact_match") for row in rows):
        name_rows = store.run(
            _INTERSECTION_BY_NAME,
            names=names,
            run=run_id,
            limit=limit,
        )
        remapped = _remap_compound_intersection(
            name_rows, unique_ids, names, limit
        )
        if remapped:
            return remapped
    return rows


def load_entity_facts(
    store: Neo4jStore,
    entity_ids: list[str],
    *,
    run_id: str,
    limit: int = 120,
) -> list[dict]:
    """Load evidence-backed outgoing facts for comparison map/reduce."""
    unique_ids = list(dict.fromkeys(item for item in entity_ids if item))
    if not unique_ids:
        return []
    return _decode_qualifiers(
        store.run(_ENTITY_FACTS, entity_ids=unique_ids, run=run_id, limit=limit)
    )


def load_entity_sections(
    store: Neo4jStore,
    entity_ids: list[str],
    *,
    run_id: str,
    limit: int = 80,
) -> list[dict]:
    """Load provenance chunks for evidence-backed, extractive comparison summaries."""
    unique_ids = list(dict.fromkeys(item for item in entity_ids if item))
    if not unique_ids:
        return []
    return store.run(
        _ENTITY_SECTIONS, entity_ids=unique_ids, run=run_id, limit=limit
    )


def find_supported_paths(
    store: Neo4jStore,
    start_id: str,
    end_id: str,
    *,
    run_id: str,
    allowed_predicates: list[str] | None = None,
    max_hops: int = 3,
    limit: int = 20,
) -> list[dict]:
    """Return short evidence-backed paths over domain serving edges only."""
    if start_id == end_id:
        raise ValueError("path endpoints must be different")
    schema_predicates = tuple(load_ontology().predicates)
    requested = tuple(allowed_predicates or schema_predicates)
    allowed = [item for item in requested if item in schema_predicates]
    if not allowed:
        raise ValueError("allowed_predicates must contain a domain predicate")
    hops = max(1, min(int(max_hops), 3))
    query = f"""
MATCH path = (start:Entity {{canonical_id: $start_id}})
             -[rels*1..{hops}]-(end:Entity {{canonical_id: $end_id}})
WHERE all(node IN nodes(path) WHERE node:Entity)
  AND all(node IN nodes(path)
          WHERE single(candidate IN nodes(path) WHERE candidate = node))
  AND all(rel IN relationships(path) WHERE type(rel) IN $allowed_predicates
          AND rel.fact_key IS NOT NULL
          AND rel.active_run_id = $run
          AND size(coalesce(rel.evidence_ids, [])) > 0)
RETURN [node IN nodes(path) | {{id: node.canonical_id, name: node.preferred_name,
                                type: node.entity_type}}] AS nodes,
       [rel IN relationships(path) | type(rel)] AS predicates,
       [rel IN relationships(path) | startNode(rel).canonical_id] AS edge_source_ids,
       [rel IN relationships(path) | endNode(rel).canonical_id] AS edge_target_ids,
       [rel IN relationships(path) | rel.fact_key] AS fact_keys,
       [rel IN relationships(path) | rel.evidence_ids] AS evidence_id_groups,
       [rel IN relationships(path) | rel.page_start] AS pages,
       [rel IN relationships(path) | coalesce(rel.best_support_score, 0.0)] AS edge_scores
ORDER BY reduce(total = 0.0, rel IN relationships(path) |
                  total + coalesce(rel.best_support_score, 0.0)) / length(path) DESC,
         length(path) ASC
LIMIT $limit
"""
    return store.run(
        query,
        start_id=start_id,
        end_id=end_id,
        run=run_id,
        allowed_predicates=allowed,
        limit=limit,
    )


def load_evidence(store: Neo4jStore, assertion_ids: list[str]) -> list[dict]:
    """Fetch verbatim evidence quotes for a set of assertions."""
    if not assertion_ids:
        return []
    return _decode_qualifiers(store.run(_EVIDENCE, ids=assertion_ids))


def _decode_qualifiers(rows: list[dict]) -> list[dict]:
    """Decode the JSON property used by the persisted Assertion schema."""
    decoded: list[dict] = []
    for row in rows:
        item = dict(row)
        raw = item.pop("qualifiers_json", None)
        if raw is not None:
            try:
                value = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, json.JSONDecodeError):
                value = {}
            item["qualifiers"] = value if isinstance(value, dict) else {}
        else:
            item.setdefault("qualifiers", {})
        decoded.append(item)
    return decoded


def card_to_context(card: HerbCard, evidence: list[dict]) -> tuple[str, list[dict]]:
    """Serialise a card into prompt context plus a source list.

    确定性序列化：列名、页码、证据都由程序填充，不交给模型决定
    （对应设计 §2.3.5「引用页码由程序填充」）。
    """
    lines = [f"【{card.name}】"]
    if card.aliases:
        lines.append(f"  别名：{'、'.join(card.aliases)}")
    for slot in card.slots:
        pages = "、".join(f"第{p}页" for p in slot.pages) or "页码缺失"
        lines.append(f"  {slot.display}：{'、'.join(slot.values)}（{pages}）")

    if evidence:
        lines.append("\n【教材原句】")
        for item in evidence:
            quote = (item["quote"] or "").replace("\n", "")
            lines.append(f"  第{item['page_start']}页「{quote}」")

    sources = [
        {
            "page_start": item["page_start"],
            "page_end": item["page_start"],
            "score": 1.0,
            "matched_terms": [item["predicate"]],
            "evidence_quote": (item["quote"] or "").replace("\n", ""),
            "match_mode": item["match_mode"],
        }
        for item in evidence
    ]
    return "\n".join(lines), sources
