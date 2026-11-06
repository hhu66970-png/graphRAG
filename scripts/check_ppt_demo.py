"""Check the PPT demo manifest; call a running backend only with --run --base-url."""

from __future__ import annotations

import argparse
from collections import Counter
from http.client import HTTPException
import json
import math
from pathlib import Path
import re
import time
from urllib import error, parse, request

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "eval/demo/ppt_demo_cases.json"
STATUSES = {"grounded", "partial_grounded", "source_only", "abstain"}
INTENTS = {"attr", "compare", "fallback"}
METHODS = {"auto", "vector", "keyword_graph"}
ALL_STATUSES = STATUSES | {"verification_unavailable", "not_applicable"}
ALL_INTENTS = INTENTS | {"reverse", "intersect", "path"}


def validate_manifest(data):
    """Validate this small, fixed demo set without importing application code."""
    failures = []
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        return ["invalid_manifest_schema"]
    source = data.get("source", {})
    if (not isinstance(source, dict)
            or not isinstance(source.get("filename"), str)
            or not source["filename"].endswith(".pptx")
            or any(c in source["filename"] for c in "/\\:")
            or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", "")))):
        failures.append("invalid_source_provenance")
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != 8:
        return failures + ["expected_eight_cases"]
    ids, states, intents, pairs, requests = [], [], [], 0, 0
    for case in cases:
        if not isinstance(case, dict):
            failures.append("invalid_case")
            continue
        ids.append(case.get("id"))
        if (not isinstance(case.get("id"), str)
                or not re.fullmatch(r"[a-z][a-z0-9_]*", case["id"])
                or not isinstance(case.get("question"), str)
                or not case["question"].strip()
                or type(case.get("physical_slide")) is not int
                or case["physical_slide"] not in {8, 9, 10}
                or case.get("medical_gold") is not False):
            failures.append("invalid_case_metadata")
            continue
        methods = case.get("methods")
        if (not isinstance(methods, list) or not methods
                or any(not isinstance(m, str) or m not in METHODS for m in methods)
                or len(set(methods)) != len(methods)):
            failures.append("invalid_case_methods")
            continue
        requests += len(methods)
        kind = case.get("kind")
        if kind == "high_risk_design":
            state = case.get("expected_answer_status")
            if not isinstance(state, str) or state not in STATUSES:
                failures.append("invalid_design_status")
            else:
                states.append(state)
            if methods != ["auto"] or case["physical_slide"] != 8:
                failures.append("invalid_design_case")
        elif kind == "normal_route":
            intent = case.get("expected_intent")
            if not isinstance(intent, str) or intent not in INTENTS:
                failures.append("invalid_route_intent")
            else:
                intents.append(intent)
            if methods != ["auto"] or case["physical_slide"] != 10:
                failures.append("invalid_route_case")
        elif kind == "manual_pair":
            pairs += 1
            if (methods != ["vector", "keyword_graph"]
                    or case.get("manual_review_required") is not True
                    or case["physical_slide"] != 9
                    or "expected_answer_status" in case
                    or "expected_answer" in case):
                failures.append("invalid_manual_pair")
        else:
            failures.append("invalid_case_kind")
    if len(set(str(value) for value in ids)) != 8:
        failures.append("duplicate_case_id")
    if Counter(states) != Counter(STATUSES):
        failures.append("expected_four_design_statuses")
    if Counter(intents) != Counter(INTENTS):
        failures.append("expected_three_normal_intents")
    if pairs != 1 or requests != 9:
        failures.append("expected_one_pair_nine_requests")
    return failures


def request_json(base_url, path, payload, timeout):
    """Never return a server error body, exception text, headers or credentials."""
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(base_url + path, data=body, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as response:
            code = response.status
            raw = response.read(2_000_001)
    except error.HTTPError as exc:
        return exc.code, None, "http_error"
    except (OSError, ValueError, HTTPException):
        return None, None, "connection_failed"
    if len(raw) > 2_000_000:
        return code, None, "response_too_large"
    try:
        return code, json.loads(raw), None
    except (ValueError, UnicodeError):
        return code, None, "invalid_json"


def check_response(case, method, code, data, transport_failure=None):
    """Check contracts only; summaries deliberately exclude raw answers/errors."""
    failures, visible = [], {}
    if code != 200:
        failures.append("http_status")
    if transport_failure:
        failures.append(transport_failure)
    if not isinstance(data, dict):
        failures.append("response_not_object")
    elif code == 200:
        if "error" in data:
            failures.append("error_response")
        for key, expected_type in (("answer", str), ("sources", list), ("subgraph", dict), ("warnings", list)):
            if not isinstance(data.get(key), expected_type):
                failures.append("invalid_" + key)
        if not isinstance(data.get("answer"), str) or not data["answer"].strip():
            failures.append("empty_answer")
        graph = data.get("subgraph")
        if isinstance(graph, dict) and any(not isinstance(graph.get(key), list) for key in ("nodes", "edges")):
            failures.append("invalid_subgraph_lists")
        if data.get("method") != method:
            failures.append("method_mismatch")
        if case["kind"] == "high_risk_design":
            if data.get("answer_status") != case["expected_answer_status"]:
                failures.append("answer_status_mismatch")
            if not isinstance(data.get("high_risk"), dict) or data["high_risk"].get("is_high_risk") is not True:
                failures.append("high_risk_flag")
            for key in ("claims", "evidence_units"):
                if not isinstance(data.get(key), list):
                    failures.append("invalid_" + key)
            expected_abstained = case["expected_answer_status"] != "grounded"
            if data.get("abstained") is not expected_abstained:
                failures.append("abstained_mismatch")
            if case["expected_answer_status"] == "grounded" and (
                    not data.get("claims") or not data.get("evidence_units")):
                failures.append("grounded_without_evidence")
            if case["expected_answer_status"] in {"source_only", "abstain"}:
                if data.get("claims") != [] or data.get("abstained") is not True:
                    failures.append("unjustified_conclusion")
        if case["kind"] == "normal_route" and data.get("intent") != case["expected_intent"]:
            failures.append("intent_mismatch")
        state, intent = data.get("answer_status"), data.get("intent")
        visible = {
            "answer_present": isinstance(data.get("answer"), str) and bool(data["answer"].strip()),
            "answer_status": state if isinstance(state, str) and state in ALL_STATUSES else None,
            "intent": intent if isinstance(intent, str) and intent in ALL_INTENTS else None,
            "sources_count": len(data["sources"]) if isinstance(data.get("sources"), list) else None,
            "warnings_count": len(data["warnings"]) if isinstance(data.get("warnings"), list) else None,
        }
    return {"passed": not failures, "failures": failures, "visible": visible}


def run_checks(manifest, base_url, *, timeout=120, reviewed_cases=()):
    """Run nine live requests, stopping before queries when health fails."""
    started = time.perf_counter()
    code, health, failure = request_json(base_url, "/api/health", None, timeout)
    health_ok = code == 200 and isinstance(health, dict) and health.get("status") == "ok" and not failure
    report = {
        "mode": "live_api",
        "source": manifest["source"],
        "medical_correctness_verified": False,
        "health": {"passed": health_ok, "http_status": code,
                   "elapsed_seconds": round(time.perf_counter() - started, 3)},
        "results": [],
    }
    if not health_ok:
        report.update(readiness="failed", failures=["health_failed"], requests_run=0)
        return report
    for case in manifest["cases"]:
        for method in case["methods"]:
            started = time.perf_counter()
            code, data, failure = request_json(base_url, "/api/graphrag", {"question": case["question"], "method": method}, timeout)
            result = check_response(case, method, code, data, failure)
            result.update(case_id=case["id"], method=method, http_status=code,
                          elapsed_seconds=round(time.perf_counter() - started, 3),
                          manual_review_required=case.get("manual_review_required", False))
            report["results"].append(result)
    manual = {case["id"] for case in manifest["cases"] if case.get("manual_review_required")}
    missing = sorted(manual - set(reviewed_cases))
    failed = any(not result["passed"] for result in report["results"])
    report.update(
        requests_run=len(report["results"]),
        pending_manual_cases=missing,
        operator_declared_reviewed_cases=sorted(set(reviewed_cases)),
        review_provenance="operator_declaration_only",
        readiness="failed" if failed else ("needs_manual_review" if missing else "checks_complete_with_operator_review"),
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Explicitly call the running backend; may invoke configured models")
    mode.add_argument("--validate-only", action="store_true", help="Only validate the manifest (also the default); never make requests")
    parser.add_argument("--base-url", help="Backend origin, e.g. http://127.0.0.1:5000")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--reviewed-case", action="append", default=[], metavar="CASE_ID", help="Operator declaration after reviewing both actual pair answers and citations")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.run != bool(args.base_url) or (args.reviewed_case and not args.run):
        parser.error("live checks require both --run and --base-url; review declarations require a live check")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.base_url:
        try:
            url = parse.urlsplit(args.base_url)
            valid = (url.scheme in {"http", "https"} and url.hostname and not url.username
                     and not url.password and not url.query and not url.fragment and url.path in {"", "/"})
            _ = url.port
        except ValueError:
            valid = False
        if not valid:
            parser.error("base URL must be an HTTP(S) origin without credentials or query parameters")
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        failures = validate_manifest(manifest)
    except (OSError, ValueError, TypeError):
        manifest, failures = {}, ["manifest_unreadable"]
    manual_ids = ({case["id"] for case in manifest["cases"] if case.get("manual_review_required")}
                  if not failures else set())
    if set(args.reviewed_case) - manual_ids:
        failures.append("unknown_reviewed_case")
    if failures:
        report = {"mode": "manifest_only", "readiness": "failed", "failures": failures}
    elif args.run:
        report = run_checks(manifest, args.base_url.rstrip("/"), timeout=args.timeout, reviewed_cases=args.reviewed_case)
    else:
        report = {"mode": "manifest_only", "source": manifest["source"], "manifest_valid": True, "case_count": 8,
                  "planned_requests": 9, "requests_run": 0, "readiness": "not_run",
                  "medical_correctness_verified": False}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        try:
            args.out.write_text(rendered + "\n", encoding="utf-8")
        except OSError:
            print(json.dumps({"readiness": "failed", "failures": ["output_write_failed"]}))
            return 1
    print(rendered)
    if report["readiness"] == "failed":
        return 1
    return 2 if report["readiness"] == "needs_manual_review" else 0


if __name__ == "__main__":
    raise SystemExit(main())
