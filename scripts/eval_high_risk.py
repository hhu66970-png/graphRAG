"""高风险链路的三组对比评测（任务书 §9 / §11）。

默认只跑 C 组（条件证据单元 + 输出校验，即生产链路），因为 A/B 两组要调用
模型。想跑 A/B 必须显式给 `--allow-llm`，否则如实报「未运行」。

语义指标（正确性、完整性、条件遗漏、拒答是否恰当）需要人工标注，
本脚本**不会**给出这类分数；见 `eval/high_risk/README.md`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graphragexpr.evaluation.high_risk_eval import (  # noqa: E402
    DEFAULT_DATASET,
    ArmRunner,
    compare,
    load_dataset,
)


def main() -> int:
    # 连接串只从 .env（已被忽略，不入库）或既有环境变量读取，脚本不回显任何密钥。
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("dev", "test"), default=None)
    parser.add_argument(
        "--arms", default="conditional_units_verified",
        help="逗号分隔；可选 " + ",".join(
            ("raw_text_prompt", "raw_text_verified", "conditional_units_verified")
        ),
    )
    parser.add_argument(
        "--allow-llm", action="store_true",
        help="允许调用模型跑 A/B 两组；未给出时 A/B 记为未运行",
    )
    parser.add_argument("--run-id", default=None, help="限定数据批次；默认取 ACTIVE")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if dataset.validation_errors:
        print(json.dumps(
            {"dataset": dataset.summary, "validation_errors": list(dataset.validation_errors)},
            ensure_ascii=False, indent=2,
        ))
        return 1
    if args.validate_only:
        print(json.dumps({"dataset": dataset.summary}, ensure_ascii=False, indent=2))
        return 0

    from graphragexpr.storage import serving_read
    from graphragexpr.storage.neo4j_store import Neo4jStore

    store = Neo4jStore()
    try:
        run_id = args.run_id or serving_read.active_run(store)
        if not run_id:
            print(json.dumps(
                {"error": "没有 ACTIVE 的 IngestRun，无法评测"},
                ensure_ascii=False, indent=2,
            ))
            return 1

        llm = None
        if args.allow_llm:
            from graphragexpr.generation.llm import OpenAICompatibleClient
            from graphragexpr.config import KGSettings

            llm = OpenAICompatibleClient(KGSettings())

        runner = ArmRunner(llm=llm, store=store, run_id=run_id)
        arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
        result = compare(
            dataset,
            runners={arm: runner.runner(arm) for arm in arms},
            split=args.split,
        )
    finally:
        store.close()

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"已写入 {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
