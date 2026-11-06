"""Evaluation controls only: fake model/store, never a live API or database."""
import json
from types import SimpleNamespace

import pytest

from graphragexpr.evaluation.high_risk_eval import (
    ArmRunner, HighRiskDataset, adjudication_template, compare, load_dataset,
    mechanical_metrics, semantic_metrics,
)
from graphragexpr.highrisk.claims import CandidateClaim, ClaimBatch, ClaimKind
from .conftest import FakeHighRiskStore, dosage_row


def question(**changes):
    item = {
        "id": "synthetic-1", "split": "test", "risk_category": "dosage",
        "question": "测试甲的用量是什么？", "expected_subject": "测试甲",
        "required_predicates": ["HAS_DOSAGE"], "gold_answer_points": ["测试用提议，不是医学金标"],
        "forbidden_unsupported_claims": [], "should_abstain": False,
        "annotation_status": "draft_from_textbook_pending_human_adjudication",
    }
    item.update(changes)
    return item


@pytest.fixture
def store_and_links(monkeypatch):
    from graphragexpr.storage import serving_read
    store = FakeHighRiskStore([dosage_row(
        "source-1", "煎服，1～2g。", subject_id="H:测试甲", subject_name="测试甲")])
    linked_inputs = []

    def link(_store, text, **kwargs):
        linked_inputs.append(text)
        if "测试甲" not in text:
            return []
        return [{"canonical_id": "H:测试甲", "entity_type": "HERB", "name": "测试甲"}]

    monkeypatch.setattr(serving_read, "link_entities_in_text", link)
    return store, linked_inputs


class Model:
    def __init__(self, *, bad_value=False, include_missing=False, fail=False):
        self.calls = []
        self.bad_value, self.include_missing, self.fail = bad_value, include_missing, fail

    def complete_json(self, *, system, user, schema, purpose):
        self.calls.append({"system": system, "user": user, "purpose": purpose})
        if self.fail:
            raise RuntimeError("authorization=SECRET_CANARY raw hidden reasoning")
        context = user.split("证据：\n", 1)[1].split("\n问题：", 1)[0]
        evidence = json.loads(context)
        n = 9 if self.bad_value else 2
        claim = CandidateClaim(
            claim_id="candidate-1", kind=ClaimKind.DOSAGE, subject_name="测试甲",
            statement=f"煎服，1～{n}g。", asserted_numbers=[1, n], asserted_unit="g",
            asserted_route="煎服", cited_unit_ids=[evidence[0]["unit_id"]],
        )
        claims = [claim]
        if self.include_missing:
            claims.append(claim.model_copy(update={"claim_id": "missing", "cited_unit_ids": []}))
        return schema(claims=claims, notes=["SECRET_CANARY internal reasoning"])


def run_pair(runner, *arms):
    return compare(HighRiskDataset((question(),)),
                   runners={arm: runner.runner(arm) for arm in arms})


def test_raw_reference_baseline_can_accept_and_reject(store_and_links):
    store, _ = store_and_links
    model = Model(include_missing=True)
    runner = ArmRunner(llm=model, store=store, run_id="run-fake")
    result = run_pair(runner, "raw_text_prompt", "raw_text_verified")
    a = result["predictions"]["raw_text_prompt"]["synthetic-1"]
    b = result["predictions"]["raw_text_verified"]["synthetic-1"]
    assert len(a["claims"]) == 2 and len(b["claims"]) == 1
    assert b["rejections"][0]["hard_failures"] == ["missing_citation"]
    assert b["answer_status"] != "abstain"
    assert len(model.calls) == 1
    assert result["controlled_pairs"][0]["same_sources_prompt_candidates"]
    assert "SECRET_CANARY" not in json.dumps(result)


def test_conditional_pair_reuses_exact_candidates_and_only_checks_second(store_and_links):
    store, _ = store_and_links
    model = Model(bad_value=True)
    runner = ArmRunner(llm=model, store=store, run_id="run-fake")
    result = run_pair(runner, "conditional_units_prompt", "conditional_units_claims_verified")
    unverified = result["predictions"]["conditional_units_prompt"]["synthetic-1"]
    verified = result["predictions"]["conditional_units_claims_verified"]["synthetic-1"]
    assert unverified["claims"] and not verified["claims"]
    assert verified["rejections"]
    assert unverified["_audit"]["candidate_hash"] == verified["_audit"]["candidate_hash"]
    assert len(model.calls) == 1
    assert result["controlled_pairs"][0]["same_sources_prompt_candidates"]


def test_both_representations_freeze_same_sources_under_bounded_context(store_and_links):
    store, _ = store_and_links
    runner = ArmRunner(llm=Model(), store=store, evidence_char_budget=6000)
    result = run_pair(runner, "raw_text_prompt", "conditional_units_prompt")
    raw, units = [result["predictions"][arm]["synthetic-1"]["_audit"]
                  for arm in ("raw_text_prompt", "conditional_units_prompt")]
    assert raw["source_hash"] == units["source_hash"]
    assert raw["source_ids"] == units["source_ids"] == ["source-1"]
    assert raw["context_chars"] <= 6000 and units["context_chars"] <= 6000
    assert raw["context_chars"] != units["context_chars"]  # no equal-token pretense


def test_question_scope_uses_no_expected_subject(store_and_links):
    store, inputs = store_and_links
    runner = ArmRunner(llm=Model(), store=store)
    item = question(expected_subject="DO_NOT_LINK_THIS_GOLD")
    payload = runner.runner("raw_text_prompt")(item)
    assert inputs == [item["question"]]
    assert not payload["_audit"]["uses_gold_subject"]


def test_component_scope_declares_gold_subject(store_and_links):
    store, inputs = store_and_links
    runner = ArmRunner(llm=Model(), store=store, scope="component")
    payload = runner.runner("raw_text_prompt")(question())
    assert inputs == ["测试甲"]
    assert payload["_audit"]["uses_gold_subject"]


def test_production_reference_keeps_old_component_subject_and_no_model(store_and_links):
    store, inputs = store_and_links
    model = Model()
    payload = ArmRunner(llm=model, store=store).runner("conditional_units_verified")(question())
    assert inputs == ["测试甲"]
    assert not model.calls
    assert payload["_audit"]["scope"] == "legacy_component_gold_subject"
    assert payload["_audit"]["generator"] == "production_deterministic_rendering_no_llm"


def test_unknown_subject_is_not_a_correct_abstention(store_and_links):
    store, _ = store_and_links
    payload = ArmRunner(llm=Model(), store=store).runner("raw_text_prompt")(
        question(question="未知复方的用量？"))
    metrics = mechanical_metrics({"x": payload})
    assert payload["answer_status"] == "unresolved_subject"
    assert metrics["failed"] == 1 and metrics["abstained"] == metrics["answered"] == 0


def test_not_run_does_not_query_store_or_count_as_answer():
    class NeverStore:
        def run(self, *args, **kwargs):
            raise AssertionError("should not query")
    result = run_pair(ArmRunner(store=NeverStore()), "raw_text_prompt", "raw_text_verified")
    metrics = result["arms"]["raw_text_prompt"]["mechanical"]
    assert metrics["not_run"] == 1 and metrics["answered"] == 0
    assert result["delta"] is None
    assert not result["controlled_pairs"][0]["same_sources_prompt_candidates"]


def test_provider_failure_is_shared_and_does_not_export_raw_exception(store_and_links):
    store, _ = store_and_links
    model = Model(fail=True)
    result = run_pair(ArmRunner(llm=model, store=store), "raw_text_prompt", "raw_text_verified")
    assert len(model.calls) == 1
    assert result["arms"]["raw_text_prompt"]["mechanical"]["failed"] == 1
    assert "RuntimeError" in json.dumps(result)
    assert "SECRET_CANARY" not in json.dumps(result)


def test_oversized_evidence_is_not_truncated_or_sent_to_model(store_and_links):
    store, _ = store_and_links
    model = Model()
    payload = ArmRunner(llm=model, store=store, evidence_char_budget=1).runner("raw_text_prompt")(question())
    assert not model.calls and payload["abstained"]
    assert payload["_audit"]["skipped_source_ids"] == ["source-1"]


def test_adjudication_export_preserves_drafts_and_blank_pair_questions():
    dataset = load_dataset()
    before = json.dumps(dataset.items, ensure_ascii=False, sort_keys=True)
    template = adjudication_template(dataset)
    assert len(template["items"]) == 14
    assert not template["semantic_scoring_allowed"]
    for item in template["items"]:
        assert item["annotation_status"] == "pending_human_review"
        assert item["adjudication"]["decision"] == "pending"
        assert len(item["reviews"]) == 2
        pair = item["paired_variant_draft"]
        assert pair["question"] == "" and pair["should_abstain"] is None
        assert pair["split"] == item["split"]
        assert not pair["gold_answer_points"]
    assert before == json.dumps(dataset.items, ensure_ascii=False, sort_keys=True)
    assert not dataset.summary["semantic_scoring_allowed"]
    assert semantic_metrics(dataset)["status"] == "not_measured"


def test_export_allowlist_drops_nested_reasoning_notes_and_credentials():
    payload = {"answer_status": "abstain", "claims": [{"notes": "SECRET", "claim_id": "x"}],
               "analysis": "SECRET", "_audit": {"analysis": "SECRET", "evidence": [
                   {"unit_id": "x", "reasoning": "SECRET", "api_key": "SECRET"}]}}
    result = compare(HighRiskDataset((question(),)), runners={"fixture": lambda item: payload},
                     run_metadata={"api_key": "SECRET", "model": "fake"})
    assert "SECRET" not in json.dumps(result)
    assert result["metadata"] == {"model": "fake"}


def test_cli_validation_and_template_do_not_construct_database_or_model(monkeypatch, tmp_path):
    from scripts import eval_high_risk as cli
    from graphragexpr.storage import neo4j_store
    monkeypatch.setattr(neo4j_store, "Neo4jStore", lambda: pytest.fail("no database"))
    monkeypatch.setattr(cli, "_evaluation_llm", lambda cap: pytest.fail("no model"))
    assert cli.main(["--validate-only", "--allow-llm"]) == 0
    output = tmp_path / "review.json"
    assert cli.main(["--export-adjudication", str(output)]) == 0
    assert not json.loads(output.read_text(encoding="utf-8"))["semantic_scoring_allowed"]


@pytest.mark.parametrize("missing", ["LLM_TOKEN", "LLM_ENDPOINT", "LLM_MODEL"])
def test_live_adapter_requires_explicit_provider_settings(monkeypatch, missing):
    from scripts import eval_high_risk as cli
    for key, value in {"LLM_TOKEN": "fake-secret", "LLM_ENDPOINT": "https://example.invalid/v1",
                       "LLM_MODEL": "fixture"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match="LLM_MODEL"):
        cli._evaluation_llm(200)


def test_actual_client_adapter_uses_complete_json_and_bounded_call(monkeypatch):
    from scripts import eval_high_risk as cli
    import openai
    seen = []

    def create(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({"claims": [], "notes": ["ignored"]})))])

    transport = SimpleNamespace(base_url="https://example.invalid/v1",
                                chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: transport)
    for key, value in {"LLM_TOKEN": "fake", "LLM_ENDPOINT": "https://example.invalid/v1",
                       "LLM_MODEL": "deepseek-fixture"}.items():
        monkeypatch.setenv(key, value)
    client = cli._evaluation_llm(250)
    result = client.complete_json(system="s", user="u", schema=ClaimBatch, purpose="fixture")
    assert isinstance(result, ClaimBatch)
    assert seen[0]["max_tokens"] == 250 and seen[0]["temperature"] == 0.0
    assert seen[0]["model"] == "deepseek-fixture"


def test_unknown_or_repeated_arms_fail_before_database_access():
    from scripts import eval_high_risk as cli
    for arms in ("unknown", "raw_text_prompt,raw_text_prompt", ""):
        with pytest.raises(SystemExit):
            cli.main(["--arms", arms])


def test_duplicate_candidate_ids_do_not_release_rejected_claims(store_and_links):
    store, _ = store_and_links

    class DuplicateModel(Model):
        def complete_json(self, **kwargs):
            batch = super().complete_json(**kwargs)
            valid = batch.claims[0]
            invalid = valid.model_copy(update={"asserted_numbers": [99], "statement": "99g。"})
            return ClaimBatch(claims=[valid, invalid])  # deliberately same claim_id

    payload = ArmRunner(llm=DuplicateModel(), store=store).runner(
        "conditional_units_claims_verified")(question())
    assert len(payload["claims"]) == 1
    assert "99" not in payload["answer_text"]
    assert payload["rejections"]


@pytest.mark.parametrize("refs,citable,failure", [
    (["missing"], True, "unknown_evidence"),
    (["known"], False, "evidence_not_locatable"),
])
def test_raw_baseline_rejects_unknown_and_unlocatable_refs(refs, citable, failure):
    from graphragexpr.evaluation.high_risk_eval import _verify_raw_citations
    claim = CandidateClaim(claim_id="x", kind=ClaimKind.DOSAGE, statement="测试结论",
                           cited_unit_ids=refs)
    accepted, rejections = _verify_raw_citations([claim], [{"unit_id": "known", "is_citable": citable}])
    assert not accepted and failure in rejections[0]["hard_failures"]


def test_cli_default_keeps_production_reference_and_exports_reproducible_record(
        monkeypatch, store_and_links, tmp_path):
    from scripts import eval_high_risk as cli
    from graphragexpr.storage import neo4j_store, serving_read
    store, _ = store_and_links
    monkeypatch.setattr(neo4j_store, "Neo4jStore", lambda: store)
    monkeypatch.setattr(serving_read, "active_run", lambda store: "fake-active")
    monkeypatch.setattr(cli, "_evaluation_llm", lambda cap: pytest.fail("default must not use model"))
    monkeypatch.setattr(cli, "_git_value", lambda *args: "" if args[0] == "status" else "fixture-sha")
    monkeypatch.setenv("LLM_TOKEN", "SECRET_CANARY")
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps(question(), ensure_ascii=False) + "\n", encoding="utf-8")
    output, predictions = tmp_path / "result.json", tmp_path / "predictions.jsonl"
    assert cli.main(["--dataset", str(data), "--out", str(output),
                     "--predictions-out", str(predictions), "--allow-llm"]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert list(result["arms"]) == ["conditional_units_verified"]
    assert result["metadata"]["ingest_run_id"] == "fake-active"
    assert result["metadata"]["code_sha"] == "fixture-sha"
    assert result["metadata"]["model"] is None
    assert len(result["metadata"]["evaluation_code_sha256"]) == 2
    row = json.loads(predictions.read_text(encoding="utf-8"))
    assert row["id"] == "synthetic-1" and row["_audit"]["uses_gold_subject"]
    assert "SECRET_CANARY" not in output.read_text(encoding="utf-8")
    assert result["semantic"]["status"] == "not_measured"
