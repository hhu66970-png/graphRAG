"""Threshold calibration curve (设计 §2.1.2-E).

阈值必须通过开发集标定后再冻结，不能把初始值直接写进报告结论。
断言已带 final_score，本脚本离线扫描接受阈值，无需重跑抽取、不调用 LLM。

    python scripts/calibrate_thresholds.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphragexpr.config.settings import KGSettings  # noqa: E402
from graphragexpr.domain.ids import normalize_text  # noqa: E402
from graphragexpr.domain.models import ChunkRecord  # noqa: E402
from graphragexpr.evaluation.extraction_eval import (  # noqa: E402
    load_scored_values, sweep_thresholds,
)
from graphragexpr.evaluation.reference import build_reference  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Calibrate the acceptance threshold")
    parser.add_argument("--run", default=None)
    parser.add_argument("--json", default=None)
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
        scored = load_scored_values(store, run_id=run_id)

        thresholds = [round(0.0 + 0.05 * i, 2) for i in range(0, 21)]
        curve = sweep_thresholds(references, scored, thresholds)

        current = KGSettings().final_accept_threshold
        print("=" * 70)
        print(f"接受阈值标定曲线（run = {run_id}）")
        print(f"参照集 {len(references)} 味药；断言取值 {len(scored)} 条")
        print("=" * 70)
        print(f"{'阈值':>6}{'保留取值':>10}{'命中':>8}{'Recall':>10}{'Prec↓':>9}{'F1↓':>9}")
        print("-" * 70)
        best = max(curve, key=lambda r: r["f1_lower_bound"])
        for row in curve:
            mark = ""
            if row is best:
                mark = "  ← F1↓ 最高"
            if abs(row["threshold"] - current) < 1e-9:
                mark += f"  （当前配置 {current}）"
            print(f"{row['threshold']:>6.2f}{row['predicted_values']:>10}"
                  f"{row['matched']:>8}{row['recall_at_field']:>10.1%}"
                  f"{row['precision_at_field_lower_bound']:>9.1%}"
                  f"{row['f1_lower_bound']:>9.1%}{mark}")

        print()
        print("读法：Prec 是下界（图谱含字段外的正确事实），因此曲线只能用于")
        print("**相对比较**——看阈值变化带来的增减，不能把绝对值当准确率。")
        print("建图取向是 Precision 优先（设计 §2.1.1-2），若要提高精确率，")
        print("应选 Recall 开始明显下滑之前的最大阈值。")

        if args.json:
            Path(args.json).write_text(
                json.dumps({"run_id": run_id, "curve": curve}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            print(f"\n曲线已写入 {args.json}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
