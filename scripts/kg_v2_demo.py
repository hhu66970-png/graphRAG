"""Audit the legacy graph and run the v2 extractor on real textbook chunks.

用途：
  1. 量化旧图（KG0）的结构性缺陷，产出可复现的对照数字；
  2. 从 Neo4j 取真实教材文本块，跑 v2 抽取管线，展示带证据跨度的断言。

只读旧图，不修改任何现网数据；v2 抽取结果默认只打印、不入库。
    python scripts/kg_v2_demo.py --audit
    python scripts/kg_v2_demo.py --extract 人参 --limit 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphragexpr.config.settings import KGSettings  # noqa: E402
from graphragexpr.domain.ids import chunk_id as make_chunk_id  # noqa: E402
from graphragexpr.domain.ids import normalize_text  # noqa: E402
from graphragexpr.domain.models import ChunkRecord  # noqa: E402
from graphragexpr.domain.ontology import load_ontology  # noqa: E402
from graphragexpr.extract.extractor import ChunkExtractor  # noqa: E402
from graphragexpr.extract.llm_client import OpenAICompatibleClient  # noqa: E402
from graphragexpr.linking.entity_resolver import AliasStore, EntityResolver  # noqa: E402
from graphragexpr.storage.neo4j_store import Neo4jStore  # noqa: E402

AUDIT_QUERIES: dict[str, str] = {
    "同名跨类型的名字数": """
        MATCH (e) WHERE NOT e:Chunk AND NOT e:Document AND NOT e:Entity
          AND NOT e:Assertion AND NOT e:IngestRun AND e.name IS NOT NULL
        WITH e.name AS nm, collect(DISTINCT labels(e)[0]) AS types WHERE size(types) > 1
        RETURN count(*) AS value
    """,
    "不同关系类型数": """
        MATCH ()-[r]->() WHERE NOT type(r) IN ['FROM_CHUNK','PART_OF','SUBJECT','OBJECT',
                                               'EVIDENCED_BY','IN_RUN']
          AND r.fact_key IS NULL
        RETURN count(DISTINCT type(r)) AS value
    """,
    "领域关系总数": """
        MATCH ()-[r]->() WHERE NOT type(r) IN ['FROM_CHUNK','PART_OF','SUBJECT','OBJECT',
                                               'EVIDENCED_BY','IN_RUN']
          AND r.fact_key IS NULL
        RETURN count(r) AS value
    """,
    "带页码证据的关系数": """
        MATCH ()-[r]->() WHERE NOT type(r) IN ['FROM_CHUNK','PART_OF','SUBJECT','OBJECT',
                                               'EVIDENCED_BY','IN_RUN']
          AND r.fact_key IS NULL
        RETURN count(r.page_start) AS value
    """,
    "降级为概念的节点数": "MATCH (e:概念) RETURN count(e) AS value",
}

REDUNDANCY_QUERY = """
MATCH (a)-[r]->(b)
WHERE NOT type(r) IN ['FROM_CHUNK','PART_OF','SUBJECT','OBJECT','EVIDENCED_BY','IN_RUN']
  AND r.fact_key IS NULL AND a.name IS NOT NULL AND b.name IS NOT NULL
WITH a.name + '|' + type(r) + '|' + b.name AS k, count(*) AS c
RETURN count(*) AS logical, sum(c) AS actual, sum(c) - count(*) AS redundant
"""

FIND_CHUNK = """
MATCH (c:Chunk) WHERE c.text CONTAINS $keyword
RETURN c.index AS index, c.text AS text, c.page_start AS page_start,
       c.page_end AS page_end, coalesce(c.section, '') AS section
ORDER BY c.index LIMIT $limit
"""


def audit(store: Neo4jStore) -> None:
    print("=" * 68)
    print("旧图（KG0）结构性缺陷审计")
    print("=" * 68)
    for label, query in AUDIT_QUERIES.items():
        rows = store.run(query)
        print(f"  {label:<24}: {rows[0]['value'] if rows else 0}")

    rows = store.run(REDUNDANCY_QUERY)
    if rows and rows[0]["actual"]:
        row = rows[0]
        ratio = 100.0 * row["redundant"] / row["actual"]
        print(f"  {'逻辑三元组数':<24}: {row['logical']}")
        print(f"  {'实际边数':<24}: {row['actual']}")
        print(f"  {'笛卡尔积冗余边':<24}: {row['redundant']}  ({ratio:.1f}%)")


def extract_demo(store: Neo4jStore, keyword: str, limit: int) -> None:
    settings = KGSettings()
    ontology = load_ontology()
    resolver = EntityResolver(AliasStore(), settings)
    llm = OpenAICompatibleClient(settings)
    extractor = ChunkExtractor(
        ontology=ontology, settings=settings, resolver=resolver, llm=llm, run_id="demo-run"
    )

    rows = store.run(FIND_CHUNK, keyword=keyword, limit=limit)
    if not rows:
        print(f"未找到包含 {keyword!r} 的文本块")
        return

    for row in rows:
        section_path = [row["section"]] if row["section"] else []
        text = row["text"]
        chunk = ChunkRecord(
            chunk_id=make_chunk_id("legacy-zyx", section_path, normalize_text(text)),
            document_id="legacy-zyx",
            entry_id=None,
            section_path=section_path,
            order_in_entry=row["index"],
            text=text,
            normalized_text=normalize_text(text),
            page_start=row["page_start"],
            page_end=row["page_end"],
            content_hash=make_chunk_id("legacy-zyx", section_path, normalize_text(text)),
        )
        print("=" * 68)
        print(f"PDF 第 {chunk.page_start}-{chunk.page_end} 页｜章节：{row['section'] or '未识别'}")
        print("-" * 68)
        print(text[:220].replace("\n", " ") + ("..." if len(text) > 220 else ""))
        print("-" * 68)

        outcome = extractor.extract(chunk)
        print(f"统计：{outcome.counts()}")
        print(f"LLM 调用：{llm.usage.calls} 次，token 合计 {llm.usage.total_tokens}")
        print()
        for record in outcome.assertions:
            marker = {"ACCEPTED": "✓", "REVIEW": "?", "REJECTED": "✗"}[record.validation_status]
            obj = record.object_id or (record.object_value or {}).get("raw")
            print(f"  {marker} [{record.validation_status}] {record.predicate}")
            print(f"      主语 {record.subject_id}")
            print(f"      宾语 {obj}")
            if record.object_value:
                value = record.object_value
                print(f"      剂量值对象 min={value['value_min']} max={value['value_max']} "
                      f"unit={value['unit']} route={value['route']}")
            print(f"      证据「{record.evidence_quote}」 span=({record.evidence_start},"
                  f"{record.evidence_end}) mode={record.match_mode} 页码={record.page_start}")
            print(f"      rule_score={record.rule_score} final_score={record.final_score}")
        if outcome.rejected:
            print("\n  被拒候选（失败清单，不隐藏）：")
            for item in outcome.rejected[:5]:
                print(f"    - {item['reason']}: {item.get('flags') or item['raw'].get('predicate')}")


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Legacy KG audit + v2 extraction demo")
    parser.add_argument("--audit", action="store_true", help="审计旧图缺陷")
    parser.add_argument("--extract", metavar="KEYWORD", help="对含该关键词的文本块跑 v2 抽取")
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    store = Neo4jStore()
    try:
        store.verify()
        if args.audit:
            audit(store)
        if args.extract:
            extract_demo(store, args.extract, args.limit)
        if not args.audit and not args.extract:
            parser.print_help()
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
