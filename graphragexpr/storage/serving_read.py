"""Read access over the typed serving projection (SKILL §5.5).

本模块只负责「把 ACTIVE run 的类型化边读成结构化结果」，**不生成自然语言答案、
不做意图路由**——那属于 retrieval / generation 层（SKILL §5.6 / §5.8）。
API 层可以直接编排这些函数来演示 v2 图谱，而不必复制任何检索逻辑。

每个返回项都带 `page_start/page_end` 与 `evidence_ids`，使调用方能一路回溯到
Assertion 层的逐字原句（SKILL §3.1 证据优先）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.ontology import Ontology
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
WHERE e.run_id = $run AND (e.preferred_name = $name OR $name IN e.aliases)
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
WHERE target.run_id = $run AND target.entity_type IN ['EFFICACY', 'INDICATION']
  AND (target.preferred_name = $name OR $name IN target.aliases)
MATCH (h:Entity {entity_type: 'HERB'})-[e]->(target)
WHERE e.fact_key IS NOT NULL AND e.active_run_id = $run
RETURN h.preferred_name AS herb, h.canonical_id AS herb_id, type(e) AS predicate,
       e.evidence_count AS evidence_count, e.page_start AS page_start,
       e.best_support_score AS score
ORDER BY e.evidence_count DESC, e.best_support_score DESC
LIMIT $limit
"""

_EVIDENCE = """
MATCH (a:Assertion) WHERE a.assertion_id IN $ids
RETURN a.assertion_id AS assertion_id, a.evidence_quote AS quote,
       a.page_start AS page_start, a.match_mode AS match_mode,
       a.predicate AS predicate
ORDER BY a.final_score DESC
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
    canonical_id: str
    name: str
    aliases: list[str]
    slots: list[CardSlot]
    subgraph: dict

    def is_empty(self) -> bool:
        return not self.slots


def active_run(store: Neo4jStore) -> str | None:
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
            slot.evidence_ids = slot.evidence_ids[:6]
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
WHERE e.run_id = $run AND e.entity_type IN $types AND size(e.preferred_name) >= 2
  AND $text CONTAINS e.preferred_name
RETURN e.canonical_id AS canonical_id, e.preferred_name AS name,
       e.entity_type AS entity_type, size(e.preferred_name) AS len,
       CASE e.entity_type WHEN 'HERB' THEN 0 ELSE 1 END AS type_rank
ORDER BY type_rank ASC, len DESC
LIMIT $limit
"""


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


def load_evidence(store: Neo4jStore, assertion_ids: list[str]) -> list[dict]:
    """Fetch verbatim evidence quotes for a set of assertions."""
    if not assertion_ids:
        return []
    return store.run(_EVIDENCE, ids=assertion_ids)


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
