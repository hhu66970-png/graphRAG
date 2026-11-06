"""高风险组件评测：默认保留无模型的生产参考 C；实验臂必须显式启用。

校验和仲裁模板导出不连接模型或数据库。实际评测只读 ACTIVE 批次。
语义正确性等仍为 not_measured，不能由待测系统自行生成金标。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graphragexpr.evaluation.high_risk_eval import (  # noqa: E402
    ARMS, DEFAULT_DATASET, HARNESS_VERSION, ArmRunner, adjudication_template,
    compare, load_dataset,
)


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


class _BoundedCompletions:
    """Evaluation-only transport adapter; no changes to the production client."""
    def __init__(self, delegate, max_output_tokens: int):
        self.delegate, self.max_output_tokens = delegate, max_output_tokens

    def create(self, **kwargs):
        kwargs.update(temperature=0.0, max_tokens=self.max_output_tokens)
        return self.delegate.create(**kwargs)


def _evaluation_llm(max_output_tokens: int):
    from graphragexpr.config.settings import KGSettings
    from graphragexpr.extract.llm_client import OpenAICompatibleClient
    from openai import OpenAI

    token, endpoint, model = (os.getenv(name) for name in ("LLM_TOKEN", "LLM_ENDPOINT", "LLM_MODEL"))
    if not all(value and value.strip() for value in (token, endpoint, model)):
        raise ValueError("--allow-llm 需要明确配置 LLM_TOKEN、LLM_ENDPOINT 和 LLM_MODEL，不回显其值")
    settings = KGSettings()
    transport = OpenAI(api_key=token, base_url=endpoint, timeout=settings.llm_timeout_s)
    wrapped = SimpleNamespace(
        base_url=transport.base_url,
        chat=SimpleNamespace(completions=_BoundedCompletions(
            transport.chat.completions, max_output_tokens)),
    )
    return OpenAICompatibleClient(settings, client=wrapped)


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("dev", "test"), default=None)
    parser.add_argument("--arms", default="conditional_units_verified",
                        help="逗号分隔；可选 " + ",".join(ARMS))
    parser.add_argument("--allow-llm", action="store_true",
                        help="允许显式实验臂调用已配置的兼容接口；默认不调用")
    parser.add_argument("--scope", choices=("question", "component"), default="question",
                        help="仅控制实验臂；component 使用金标主语，question 按问句链接，均非完整E2E")
    parser.add_argument("--evidence-char-budget", type=_positive, default=6000)
    parser.add_argument("--max-output-tokens", type=_positive, default=512)
    parser.add_argument("--assertion-limit", type=_positive, default=200)
    parser.add_argument("--run-id", default=None, help="限定数据批次；默认取 ACTIVE")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--export-adjudication", type=Path,
                        help="仅导出待审仲裁与空成对题模板；不连接数据库/模型")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--predictions-out", type=Path, default=None,
                        help="另导出逐题JSONL；完整预测也总是保留在结果JSON中")
    args = parser.parse_args(argv)
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        parser.error("--arms 需要不重复的已知评测臂")
    dataset = load_dataset(args.dataset)
    if dataset.validation_errors:
        print(json.dumps({"dataset": dataset.summary,
                          "validation_errors": list(dataset.validation_errors)}, ensure_ascii=False, indent=2))
        return 1
    if args.export_adjudication:
        _write_json(args.export_adjudication, adjudication_template(dataset))
        print(f"已导出待人工审核模板 {args.export_adjudication}；未改原题集或标注状态")
        return 0
    if args.validate_only:
        print(json.dumps({"dataset": dataset.summary}, ensure_ascii=False, indent=2))
        return 0

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    llm = None
    if args.allow_llm and any(arm != "conditional_units_verified" for arm in arms):
        try:
            llm = _evaluation_llm(args.max_output_tokens)
        except ValueError as exc:
            parser.error(str(exc))
    from graphragexpr.storage import serving_read
    from graphragexpr.storage.neo4j_store import Neo4jStore

    store = Neo4jStore()
    try:
        run_id = args.run_id or serving_read.active_run(store)
        if not run_id:
            print(json.dumps({"error": "没有 ACTIVE 的 IngestRun，无法评测"}, ensure_ascii=False))
            return 1
        runner = ArmRunner(llm=llm, store=store, run_id=run_id, scope=args.scope,
                           evidence_char_budget=args.evidence_char_budget,
                           assertion_limit=args.assertion_limit)
        working_tree = _git_value("status", "--porcelain")
        metadata = {
            "evaluation_id": str(uuid.uuid4()), "harness_version": HARNESS_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_sha": _git_value("rev-parse", "HEAD"),
            "evaluation_code_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__), ROOT / "graphragexpr/evaluation/high_risk_eval.py")
            },
            "branch": _git_value("branch", "--show-current"),
            "working_tree_dirty": bool(working_tree) if working_tree is not None else None,
            "dataset_file_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "ingest_run_id": run_id, "arms": arms, "scope_experimental_arms": args.scope,
            "model": os.getenv("LLM_MODEL", "deepseek-chat") if llm else None,
            "temperature": 0.0 if llm else None,
            "max_output_tokens": args.max_output_tokens if llm else None,
            "evidence_char_budget": args.evidence_char_budget,
            "assertion_limit": args.assertion_limit,
            "generation_pair_policy": "reuse_exact_candidates_within_each_representation",
            "representation_comparison": "same_source_rows_not_equal_provider_token_counts",
            "production_reference": "unchanged_gold_subject_deterministic_service",
        }
        result = compare(dataset, runners={arm: runner.runner(arm) for arm in arms},
                         split=args.split, run_metadata=metadata)
    finally:
        store.close()
    if args.predictions_out:
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_out.open("w", encoding="utf-8") as stream:
            for arm, predictions in result["predictions"].items():
                for ident, payload in predictions.items():
                    stream.write(json.dumps({"evaluation_id": metadata["evaluation_id"],
                                             "arm": arm, "id": ident, **payload}, ensure_ascii=False) + "\n")
    if args.out:
        _write_json(args.out, result)
        print(f"已写入 {args.out}（含逐题证据、候选、拒因与运行元数据）")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
