from __future__ import annotations

from graphragexpr.evaluation.retrieval_eval import evaluate_predictions, load_dataset


def test_dataset_validator_reports_120_frozen_questions():
    dataset = load_dataset()

    assert dataset.summary["total"] == 120
    assert dataset.summary["evidence_adjudicated"] == 118
    assert dataset.validation_errors == []


def test_evaluator_does_not_invent_page_metric_without_gold_pages():
    dataset = load_dataset()
    sample = next(row for row in dataset.rows if not row["gold_pages"])
    predictions = {
        sample["id"]: {
            "intent": sample["expected_route"],
            "linked_entities": [{"name": name} for name in sample["gold_entities"]],
            "predicates": sample["gold_predicates"],
            "pages": [999],
            "abstained": sample["should_abstain"],
        }
    }

    report = evaluate_predictions([sample], predictions)

    assert report["route_accuracy"] == 1.0
    assert report["entity_recall"] == 1.0
    assert report["predicate_recall"] == 1.0
    assert report["page_gold_coverage"] == 0.0
    assert report["page_recall"] is None
