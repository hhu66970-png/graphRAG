from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DATASET = Path(__file__).parents[2] / "eval" / "retrieval" / "questions.jsonl"


def _load_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_retrieval_eval_set_has_required_distribution():
    rows = _load_rows()

    assert len(rows) >= 120
    assert Counter(row["type"] for row in rows) == {
        "attr": 25,
        "reverse": 20,
        "intersect": 20,
        "compare": 20,
        "path": 20,
        "safety": 15,
    }
    assert len({row["id"] for row in rows}) == len(rows)


def test_retrieval_eval_rows_keep_gold_and_safety_contracts():
    required = {
        "id",
        "split",
        "type",
        "question",
        "gold_entities",
        "gold_predicates",
        "gold_answer_points",
        "gold_pages",
        "gold_chunk_ids",
        "required_facts",
        "forbidden_unsupported_claims",
        "should_abstain",
    }

    for row in _load_rows():
        assert required <= row.keys(), row["id"]
        assert row["split"] in {"dev", "test"}
        assert row["question"].strip()
        assert isinstance(row["should_abstain"], bool)
        if row["type"] == "safety":
            assert row["forbidden_unsupported_claims"]


def test_dataset_readme_documents_annotation_status():
    readme = DATASET.with_name("README.md").read_text(encoding="utf-8")

    assert "人工仲裁" in readme
    assert "不可" in readme and "当前图谱输出" in readme
