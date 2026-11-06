"""Verify the v2 graph against the defects measured on the legacy graph.

对照维度来自旧图实测缺陷：
同名跨类型身份、笛卡尔积冗余边、关系类型爆炸、证据缺失、静默降级。

只读，不修改任何数据。
    python scripts/verify_kg_v2.py
    python scripts/verify_kg_v2.py --herb 人参
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphragexpr.domain.ontology import load_ontology  # noqa: E402
from graphragexpr.storage.neo4j_store import Neo4jStore  # noqa: E402
from graphragexpr.storage.runs import get_active_run  # noqa: E402

LEGACY_FILTER = (
    "NOT type(r) IN ['FROM_CHUNK','PART_OF','SUBJECT','OBJECT','EVIDENCED_BY','IN_RUN'] "
    "AND r.fact_key IS NULL"
)


def compare(store: Neo4jStore, run_id: str) -> None:
    """对比新旧图谱在示例问题上的表现。"""
    print("=" * 74)
    print(f"新旧对照（ACTIVE run = {run_id}）")
    print("=" * 74)
    print(f"{'指标':<28}{'旧图 (KG0)':>18}{'新图 (v2)':>18}")
    print("-" * 74)

    legacy_names = store.run(
        f"""
        MATCH (e) WHERE NOT e:Chunk AND NOT e:Document AND NOT e:Entity
          AND NOT e:Assertion AND NOT e:IngestRun AND e.name IS NOT NULL
        WITH e.name AS nm, collect(DISTINCT labels(e)[0]) AS t WHERE size(t) > 1
        RETURN count(*) AS value
        """
    )[0]["value"]
    # 新图身份是 canonical_id，同名不同类型必然是不同实体，故错连数恒为 0
    # 必须按 active_run_id 过滤：库里同时存在历史 run 的投影边，
    # 不过滤会把旧 run 的问题算到当前 run 头上。
    v2_cross = store.run(
        """
        MATCH (a:Entity)-[e]->(b:Entity)
        WHERE e.fact_key IS NOT NULL AND e.active_run_id = $run
          AND a.canonical_id = b.canonical_id
        RETURN count(e) AS value
        """,
        run=run_id,
    )[0]["value"]
    print(f"{'同名跨类型导致的错连':<28}{legacy_names:>18}{v2_cross:>18}")

    legacy = store.run(
        f"""
        MATCH (a)-[r]->(b) WHERE {LEGACY_FILTER}
          AND a.name IS NOT NULL AND b.name IS NOT NULL
        WITH a.name + '|' + type(r) + '|' + b.name AS k, count(*) AS c
        RETURN count(*) AS logical, sum(c) AS actual
        """
    )[0]
    v2 = store.run(
        """
        MATCH (a:Entity)-[e]->(b:Entity) WHERE e.fact_key IS NOT NULL AND e.active_run_id = $run
        WITH a.canonical_id + '|' + type(e) + '|' + b.canonical_id AS k, count(*) AS c
        RETURN count(*) AS logical, sum(c) AS actual
        """,
        run=run_id,
    )[0]
    legacy_dup = (legacy["actual"] or 0) - (legacy["logical"] or 0)
    v2_dup = (v2["actual"] or 0) - (v2["logical"] or 0)
    legacy_pct = 100.0 * legacy_dup / legacy["actual"] if legacy["actual"] else 0
    v2_pct = 100.0 * v2_dup / v2["actual"] if v2["actual"] else 0
    print(f"{'冗余重复边':<28}{f'{legacy_dup} ({legacy_pct:.1f}%)':>18}"
          f"{f'{v2_dup} ({v2_pct:.1f}%)':>18}")

    legacy_types = store.run(
        f"MATCH ()-[r]->() WHERE {LEGACY_FILTER} RETURN count(DISTINCT type(r)) AS value"
    )[0]["value"]
    v2_types = store.run(
        "MATCH ()-[e]->() WHERE e.fact_key IS NOT NULL AND e.active_run_id = $run "
        "RETURN count(DISTINCT type(e)) AS value", run=run_id,
    )[0]["value"]
    ontology_size = len(load_ontology().predicates)
    print(f"{'关系类型数':<28}{legacy_types:>18}{f'{v2_types} / 本体 {ontology_size}':>18}")

    legacy_ev = store.run(
        f"MATCH ()-[r]->() WHERE {LEGACY_FILTER} RETURN count(r.page_start) AS value"
    )[0]["value"]
    legacy_total = store.run(
        f"MATCH ()-[r]->() WHERE {LEGACY_FILTER} RETURN count(r) AS value"
    )[0]["value"]
    v2_ev = store.run(
        "MATCH ()-[e]->() WHERE e.fact_key IS NOT NULL AND e.active_run_id = $run "
        "RETURN count(e.page_start) AS ev, count(e) AS total", run=run_id,
    )[0]
    v2_ev_text = f"{v2_ev['ev']}/{v2_ev['total']}"
    print(f"{'带页码 provenance 的边':<28}{f'{legacy_ev}/{legacy_total}':>18}{v2_ev_text:>18}")

    legacy_concept = store.run("MATCH (e:概念) RETURN count(e) AS value")[0]["value"]
    v2_concept = store.run(
        "MATCH (e:Entity {entity_type: 'CONCEPT'}) "
        "WHERE e.run_id = $run OR EXISTS { MATCH (e)-[:IN_RUN]->(:IngestRun {run_id: $run}) } "
        "RETURN count(e) AS value",
        run=run_id,
    )[0]["value"]
    print(f"{'CONCEPT 兜底节点':<28}{legacy_concept:>18}{v2_concept:>18}")

    coverage = store.run(
        """
        MATCH (a:Assertion {run_id: $run, validation_status: 'ACCEPTED'})
        RETURN count(a) AS accepted,
               sum(CASE WHEN a.evidence_quote IS NOT NULL AND a.match_mode <> 'not_found'
                        THEN 1 ELSE 0 END) AS with_evidence
        """,
        run=run_id,
    )[0]
    accepted = coverage["accepted"] or 0
    pct = 100.0 * (coverage["with_evidence"] or 0) / accepted if accepted else 0
    print(f"{'ACCEPTED 断言证据覆盖率':<28}{'不适用（无断言层）':>18}{f'{pct:.1f}%':>18}")


def show_herb(store: Neo4jStore, run_id: str, herb: str) -> None:
    """属性聚合演示：旧检索链路答不出的那道题。"""
    print()
    print("=" * 74)
    print(f"属性聚合：{herb}（新图类型化边，跨块，带页码）")
    print("=" * 74)
    rows = store.run(
        """
        MATCH (h:Entity {entity_type: 'HERB'}) WHERE h.preferred_name = $herb
        MATCH (h)-[e]->(o:Entity) WHERE e.fact_key IS NOT NULL AND e.active_run_id = $run
        RETURN type(e) AS predicate, o.preferred_name AS value,
               e.page_start AS page_start, e.page_end AS page_end,
               e.evidence_count AS evidence_count, e.best_support_score AS score
        ORDER BY predicate, score DESC
        """,
        herb=herb, run=run_id,
    )
    if not rows:
        print(f"  未找到 {herb} 的类型化属性边")
        return

    ontology = load_ontology()
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["predicate"], []).append(row)
    for predicate, items in grouped.items():
        display = ontology.predicate(predicate).display if predicate in ontology.predicates else predicate
        values = "、".join(dict.fromkeys(i["value"] for i in items))[:110]
        pages = sorted({i["page_start"] for i in items if i["page_start"]})
        print(f"  {display}（{predicate}）：{values}")
        print(f"      页码 {pages[:6]}｜证据条数 {sum(i['evidence_count'] or 0 for i in items)}")

    # 断言层可回溯逐字证据
    evidence = store.run(
        """
        MATCH (h:Entity {entity_type: 'HERB'}) WHERE h.preferred_name = $herb
        MATCH (a:Assertion {run_id: $run})-[:SUBJECT]->(h)
        RETURN a.predicate AS p, a.evidence_quote AS q, a.page_start AS page,
               a.match_mode AS mode
        ORDER BY a.final_score DESC LIMIT 5
        """,
        herb=herb, run=run_id,
    )
    if evidence:
        print("\n  逐字证据（断言层，可回到教材原句）：")
        for row in evidence:
            quote = (row["q"] or "").replace("\n", " ")[:56]
            print(f"    [{row['p']}] 第{row['page']}页 ({row['mode']})「{quote}」")


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Verify the v2 KG against legacy defects")
    parser.add_argument("--herb", default="人参", help="用于属性聚合演示的药名")
    parser.add_argument("--run", default=None, help="指定 run_id，默认取 ACTIVE")
    args = parser.parse_args()

    store = Neo4jStore()
    try:
        store.verify()
        run_id = args.run or get_active_run(store)
        if not run_id:
            print("没有 ACTIVE 的 IngestRun；请先执行 build_kg_v2.py --activate")
            return 1
        compare(store, run_id)
        show_herb(store, run_id, args.herb)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
