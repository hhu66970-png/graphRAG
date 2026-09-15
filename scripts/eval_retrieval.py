"""Validate the retrieval set and optionally score JSONL predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graphragexpr.evaluation.retrieval_eval import (
    DEFAULT_DATASET,
    evaluate_predictions,
    load_dataset,
    load_predictions,
)


def main() -> int:
    """命令行入口：解析参数并执行本脚本的主流程。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if dataset.validation_errors:
        print(
            json.dumps(
                {
                    "dataset": dataset.summary,
                    "validation_errors": dataset.validation_errors,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    output: dict = {"dataset": dataset.summary, "validation_errors": []}
    if args.predictions and not args.validate_only:
        output["metrics"] = evaluate_predictions(
            dataset.rows, load_predictions(args.predictions)
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
