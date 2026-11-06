"""Offline checks and fake HTTP contracts; none of these tests calls a model."""

from __future__ import annotations

import json

import pytest

from scripts import check_ppt_demo as check


@pytest.fixture
def manifest():
    return json.loads(check.DEFAULT_MANIFEST.read_text(encoding="utf-8"))


def response(case, method):
    result = {"answer": "synthetic response", "method": method, "sources": [],
              "subgraph": {"nodes": [], "edges": []}, "warnings": []}
    if case["kind"] == "high_risk_design":
        status = case["expected_answer_status"]
        result.update(answer_status=status, high_risk={"is_high_risk": True},
                      evidence_units=[], claims=[], abstained=status != "grounded")
        if status in {"grounded", "partial_grounded"}:
            result.update(evidence_units=[{"unit_id": "synthetic-evidence"}],
                          claims=[{"unit_ids": ["synthetic-evidence"]}])
    if case["kind"] == "normal_route":
        result["intent"] = case["expected_intent"]
    return result


def fake_server(manifest, *, change=None):
    by_question = {case["question"]: case for case in manifest["cases"]}
    calls = []

    def send(base, path, payload, timeout):
        calls.append((path, payload))
        if path == "/api/health":
            return 200, {"status": "ok"}, None
        case = by_question[payload["question"]]
        data = response(case, payload["method"])
        if change:
            change(case, data)
        return 200, data, None

    return send, calls


def test_default_is_manifest_only_without_network(manifest, monkeypatch, capsys):
    monkeypatch.setattr(check, "request_json", lambda *a: pytest.fail("offline mode made a request"))
    assert check.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["requests_run"] == 0
    assert report["readiness"] == "not_run"
    assert check.validate_manifest(manifest) == []
    assert sum(len(c["methods"]) for c in manifest["cases"]) == 9


def test_explicit_validation_is_offline_and_cannot_be_combined_with_run(monkeypatch):
    monkeypatch.setattr(check, "request_json", lambda *a: pytest.fail("validation made a request"))
    assert check.main(["--validate-only"]) == 0
    with pytest.raises(SystemExit) as exc:
        check.main(["--validate-only", "--run", "--base-url", "http://localhost:5000"])
    assert exc.value.code == 2


def test_missing_manual_review_is_a_nonzero_readiness_exit(manifest, monkeypatch, capsys):
    send, _ = fake_server(manifest)
    monkeypatch.setattr(check, "request_json", send)
    assert check.main(["--run", "--base-url", "http://localhost:5000"]) == 2
    assert json.loads(capsys.readouterr().out)["readiness"] == "needs_manual_review"


def test_grounded_status_without_claims_or_evidence_is_not_a_pass(manifest):
    case = next(c for c in manifest["cases"] if c.get("expected_answer_status") == "grounded")
    data = response(case, "auto")
    data.update(claims=[], evidence_units=[])
    assert "grounded_without_evidence" in check.check_response(case, "auto", 200, data)["failures"]


def test_traceable_public_manifest_has_no_private_path(manifest):
    assert manifest["source"]["filename"] == "最终版（1）(1).pptx"
    assert manifest["source"]["sha256"] == "1c532558b8c3761f12c2fb6f46354685a548d67b2616d0a7b605d5d8ced98251"
    text = json.dumps(manifest, ensure_ascii=False).lower()
    for private in ("wechat", "wxid_", "c:\\", "d:\\", "users/seven"):
        assert private not in text
    assert all(c["medical_gold"] is False for c in manifest["cases"])


def test_manifest_rejects_automatic_gold_for_manual_pair(manifest):
    case = next(c for c in manifest["cases"] if c["kind"] == "manual_pair")
    case["expected_answer"] = "可以搭配使用"
    assert "invalid_manual_pair" in check.validate_manifest(manifest)


@pytest.mark.parametrize("mutation", [
    lambda m: m["cases"][0].pop("physical_slide"),
    lambda m: m["cases"][0].update(expected_answer_status=[]),
    lambda m: m["source"].update(filename="D:\\private\\demo.pptx"),
])
def test_malformed_manifest_is_a_validation_failure(manifest, mutation):
    mutation(manifest)
    assert check.validate_manifest(manifest)


@pytest.mark.parametrize("data", [[], {"schema_version": 1, "cases": None}])
def test_invalid_manifest_root_is_a_clean_cli_failure(data, tmp_path, capsys):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert check.main(["--manifest", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["readiness"] == "failed"


@pytest.mark.parametrize("args", [["--run"], ["--base-url", "http://localhost:5000"],
                                  ["--reviewed-case", "renshen_wugong_pair"],
                                  ["--run", "--base-url", "http://user:secret@localhost:5000"]])
def test_live_execution_requires_explicit_safe_arguments(args, monkeypatch):
    monkeypatch.setattr(check, "request_json", lambda *a: pytest.fail("invalid arguments made a request"))
    with pytest.raises(SystemExit) as exc:
        check.main(args)
    assert exc.value.code == 2


def test_nine_live_contracts_still_need_manual_review(manifest, monkeypatch):
    send, calls = fake_server(manifest)
    monkeypatch.setattr(check, "request_json", send)
    result = check.run_checks(manifest, "http://localhost:5000")
    assert len(calls) == 10  # health + nine actual API calls
    assert result["requests_run"] == 9
    assert all(r["passed"] for r in result["results"])
    assert result["readiness"] == "needs_manual_review"
    assert result["pending_manual_cases"] == ["renshen_wugong_pair"]
    assert result["medical_correctness_verified"] is False
    assert all(r["elapsed_seconds"] >= 0 for r in result["results"])


def test_operator_review_is_recorded_as_declaration_only(manifest, monkeypatch):
    send, _ = fake_server(manifest)
    monkeypatch.setattr(check, "request_json", send)
    result = check.run_checks(manifest, "http://localhost:5000", reviewed_cases=["renshen_wugong_pair"])
    assert result["readiness"] == "checks_complete_with_operator_review"
    assert result["review_provenance"] == "operator_declaration_only"
    assert result["medical_correctness_verified"] is False


def test_health_failure_stops_queries_and_hides_error_body(manifest, monkeypatch):
    calls = []

    def unhealthy(*args):
        calls.append(args)
        return 500, {"message": "secret credential details"}, None

    monkeypatch.setattr(check, "request_json", unhealthy)
    result = check.run_checks(manifest, "http://localhost:5000")
    assert len(calls) == 1 and result["requests_run"] == 0
    assert result["readiness"] == "failed"
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("kind,field,value,failure", [
    ("high_risk_design", "answer_status", "verification_unavailable", "answer_status_mismatch"),
    ("normal_route", "intent", "path", "intent_mismatch"),
    ("manual_pair", "sources", "bad shape", "invalid_sources"),
])
def test_contract_or_expected_behavior_failure_is_reported(manifest, kind, field, value, failure):
    case = next(c for c in manifest["cases"] if c["kind"] == kind)
    data = response(case, case["methods"][0])
    data[field] = value
    result = check.check_response(case, case["methods"][0], 200, data)
    assert not result["passed"] and failure in result["failures"]


def test_failed_response_does_not_echo_server_details(manifest):
    case = manifest["cases"][0]
    result = check.check_response(case, "auto", 500, {"error": "token=secret", "keys": ["secret"]})
    assert not result["passed"]
    assert "secret" not in json.dumps(result)


def test_malformed_status_fields_do_not_crash_or_echo_payload(manifest):
    case = manifest["cases"][0]
    data = response(case, "auto")
    data.update(answer_status={"secret": "sensitive"}, intent=["sensitive"])
    result = check.check_response(case, "auto", 200, data)
    assert not result["passed"]
    assert result["visible"]["answer_status"] is None
    assert "sensitive" not in json.dumps(result)


def test_behavior_failure_overrides_review_declaration(manifest, monkeypatch):
    send, _ = fake_server(manifest, change=lambda case, data: data.update(method="wrong"))
    monkeypatch.setattr(check, "request_json", send)
    result = check.run_checks(manifest, "http://localhost:5000", reviewed_cases=["renshen_wugong_pair"])
    assert result["readiness"] == "failed"


def test_http_error_body_is_not_read(monkeypatch):
    from urllib.error import HTTPError
    from io import BytesIO

    def fail(*args, **kwargs):
        raise HTTPError("http://localhost", 500, "secret", {}, BytesIO(b'{"error":"secret"}'))

    monkeypatch.setattr(check.request, "urlopen", fail)
    assert check.request_json("http://localhost", "/api/health", None, 1) == (500, None, "http_error")
