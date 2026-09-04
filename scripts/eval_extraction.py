"""Score the KG against the textbook's own structured fields (设计 §9.2).

    python scripts/eval_extraction.py                 # 评测当前 ACTIVE run
    python scripts/eval_extraction.py --run <run_id>  # 评测指定 run
    python scripts/eval_extraction.py --json out.json # 同时落盘

指标口径见 graphragexpr/evaluation/reference.py 的模块说明：
Recall@field 是主指标；Precision@field 是**下界**，不得当作总体准确率。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphragexpr.domain.ids import normalize_text  # noqa: E402
from graphragexpr.domain.models import ChunkRecord  # noqa: E402
from graphragexpr.evaluation.extraction_eval import evaluate  # noqa: E402
from graphragexpr.evaluation.reference import (  # noqa: E402
    build_reference, coverage_summary,
)
from graphragexpr.storage.neo4j_store import Neo4jStore  # noqa: E402
from graphragexpr.storage.runs import get_active_run  # noqa: E402

LOAD = """
MATCH (c:Chunk) WHERE c.text IS NOT NULL
RETURN c.index AS i, c.text AS t, c.page_start AS ps, c.page_end AS pe,
       coalesce(c.section, '') AS sec
ORDER BY c.index
"""


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Extraction eval vs structured fields")
    parser.add_argument("--run", default=None)
    parser.add_argument("--json", default=None, help="把完整结果写入该文件")
    args = parser.parse_args()

    store = Neo4jStore()
    try:
        store.verify()
        run_id = args.run or get_active_run(store)
        if not run_id:
            print("没有 ACTIVE 的 IngestRun")
            return 1

        rows = store.run(LOAD)
        chunks = [
            ChunkRecord(
                chunk_id=f"c{r['i']}", document_id="zhongyaoxue-5e",
                section_path=[r["sec"]] if r["sec"] else [], order_in_entry=r["i"],
                text=r["t"], normalized_text=normalize_text(r["t"]),
                page_start=r["ps"] or 0, page_end=r["pe"] or r["ps"] or 0,
                content_hash=f"h{r['i']}",
            )
            for r in rows
        ]
        references = build_reference(chunks)
        coverage = coverage_summary(references)

        print("=" * 74)
        print("教材结构化字段参照集（纯规则解析，未调用 LLM）")
        print("=" * 74)
        for key, value in coverage.items():
            print(f"  {key:<24}{value}")

        result = evaluate(store, references, run_id=run_id)

        print()
        print("=" * 74)
        print(f"抽取质量评测（run = {run_id}）")
        print("=" * 74)
        print(f"{'字段':<14}{'金标取值':>9}{'图谱取值':>9}{'命中':>7}"
              f"{'Recall@field':>14}{'Prec@field↓':>13}")
        print("-" * 74)
        label = {"nature_flavor": "性味", "meridians": "归经", "efficacies": "功效"}
        for key, item in result["per_predicate"].items():
            print(f"{label.get(key, key):<14}{item['gold_values']:>9}"
                  f"{item['predicted_values']:>9}{item['matched']:>7}"
                  f"{item['recall_at_field']:>14.1%}"
                  f"{item['precision_at_field_lower_bound']:>13.1%}")
        overall = result["overall"]
        print("-" * 74)
        print(f"{'合计':<14}{overall['gold_values']:>9}{overall['predicted_values']:>9}"
              f"{overall['matched']:>7}{overall['recall_at_field']:>14.1%}"
              f"{overall['precision_at_field_lower_bound']:>13.1%}")
        print(f"\n参照集共 {overall['reference_herbs']} 味药，"
              f"其中 {overall['herbs_absent_from_graph']} 味在图谱中无任何对应属性边")

        print("\n完全召回率（该药字段取值一条不漏）：")
        for key, item in result["per_predicate"].items():
            n, total = item["herbs_fully_recalled"], item["herbs_evaluated"]
            ratio = n / total if total else 0
            print(f"  {label.get(key, key):<8}{n}/{total}  ({ratio:.1%})")

        print("\n漏检样例（失败分析）：")
        for key, item in result["examples"].items():
            missed = item["missed"][:5]
            if missed:
                print(f"  {label.get(key, key)}: " +
                      "；".join(f"{h}→{v}" for h, v in missed))

        print("\n⚠ 口径提醒：Prec@field 标注了 ↓ 表示它是**下界**——图谱合理包含")
        print("  【应用】段落等字段之外的正确事实，它们不在参照集里但并不算错。")
        print("  本参照集也不是设计 §9.1 要求的双人标注金标准。")

        if args.json:
            Path(args.json).write_text(
                json.dumps({"coverage": coverage, **result}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\n完整结果已写入 {args.json}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
