"""Validation and transparent metrics for the frozen retrieval question set."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DATASET = Path(__file__).parents[2] / "eval" / "retrieval" / "questions.jsonl"
EXPECTED_COUNTS = {
    "attr": 25,
    "reverse": 20,
    "intersect": 20,
    "compare": 20,
    "path": 20,
    "safety": 15,
}
REQUIRED_FIELDS = {
    "id",
    "split",
    "type",
    "question",
    "expected_route",
    "gold_entities",
    "gold_predicates",
    "gold_answer_points",
    "gold_pages",
    "gold_chunk_ids",
    "required_facts",
    "forbidden_unsupported_claims",
    "should_abstain",
    "annotation_status",
}


@dataclass(frozen=True)
class RetrievalDataset:
    rows: list[dict]
    summary: dict[str, Any]
    validation_errors: list[str]


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def load_dataset(path: str | Path = DEFAULT_DATASET) -> RetrievalDataset:
    """Load and validate distribution, identity and required annotation fields."""
    source = Path(path)
    rows = _read_jsonl(source)
    errors: list[str] = []
    ids = [str(row.get("id") or "") for row in rows]
    counts = Counter(str(row.get("type")) for row in rows)
    if dict(counts) != EXPECTED_COUNTS:
        errors.append(f"unexpected type distribution: {dict(counts)}")
    duplicate_ids = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        errors.append(f"duplicate ids: {duplicate_ids}")
    for row in rows:
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            errors.append(f"{row.get('id', '<missing-id>')}: missing {sorted(missing)}")
        if row.get("split") not in {"dev", "test"}:
            errors.append(f"{row.get('id')}: invalid split")
        if row.get("type") in {"reverse", "intersect"} and not row.get(
            "gold_result_entities"
        ):
            errors.append(f"{row.get('id')}: missing gold_result_entities")
    adjudicated = sum(
        bool(row.get("gold_pages") and row.get("gold_chunk_ids")) for row in rows
    )
    return RetrievalDataset(
        rows=rows,
        summary={
            "path": str(source),
            "total": len(rows),
            "by_type": dict(counts),
            "dev": sum(row.get("split") == "dev" for row in rows),
            "test": sum(row.get("split") == "test" for row in rows),
            "evidence_adjudicated": adjudicated,
        },
        validation_errors=errors,
    )


def _recall(gold: list[Any], predicted: list[Any]) -> float:
    expected = set(gold)
    if not expected:
        return 1.0
    return len(expected & set(predicted)) / len(expected)


def _entity_names(values: list[Any]) -> list[str]:
    names: list[str] = []
    for value in values:
        if isinstance(value, dict):
            name = value.get("name") or value.get("preferred_name")
            if name:
                names.append(str(name))
        elif value is not None:
            names.append(str(value))
    return names


def evaluate_predictions(
    rows: list[dict], predictions: dict[str, dict]
) -> dict[str, Any]:
    """Compute only metrics supported by the available gold annotations."""
    evaluated = [row for row in rows if row["id"] in predictions]
    if not evaluated:
        raise ValueError("predictions do not contain any dataset ids")
    route_scores: list[float] = []
    entity_scores: list[float] = []
    predicate_scores: list[float] = []
    abstain_scores: list[float] = []
    page_scores: list[float] = []
    page_gold_count = 0
    by_type: dict[str, dict[str, float]] = {}
    for row in evaluated:
        prediction = predictions[row["id"]]
        route = float(prediction.get("intent") == row["expected_route"])
        entity = _recall(
            row["gold_entities"],
            _entity_names(prediction.get("linked_entities") or []),
        )
        predicate = _recall(row["gold_predicates"], prediction.get("predicates") or [])
        abstain = float(bool(prediction.get("abstained")) == row["should_abstain"])
        route_scores.append(route)
        entity_scores.append(entity)
        predicate_scores.append(predicate)
        abstain_scores.append(abstain)
        if row["gold_pages"]:
            page_gold_count += 1
            page_scores.append(
                _recall(row["gold_pages"], prediction.get("pages") or [])
            )
        bucket = by_type.setdefault(
            row["type"], {"count": 0.0, "route": 0.0, "entity": 0.0, "predicate": 0.0}
        )
        bucket["count"] += 1
        bucket["route"] += route
        bucket["entity"] += entity
        bucket["predicate"] += predicate
    for bucket in by_type.values():
        count = bucket["count"]
        for key in ("route", "entity", "predicate"):
            bucket[key] = bucket[key] / count
    total = len(evaluated)
    return {
        "evaluated": total,
        "prediction_coverage": total / len(rows),
        "route_accuracy": sum(route_scores) / total,
        "entity_recall": sum(entity_scores) / total,
        "predicate_recall": sum(predicate_scores) / total,
        "abstain_accuracy": sum(abstain_scores) / total,
        "page_gold_coverage": page_gold_count / total,
        "page_recall": (sum(page_scores) / len(page_scores)) if page_scores else None,
        "by_type": by_type,
    }


def load_predictions(path: str | Path) -> dict[str, dict]:
    rows = _read_jsonl(Path(path))
    ids = [str(row["id"]) for row in rows]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate prediction ids: {sorted(duplicates)}")
    return {str(row["id"]): row for row in rows}
