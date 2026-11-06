"""PPT behavior regressions with synthetic evidence, not textbook/medical gold.

Storage, retrieval and the ordinary LLM remain test doubles. Production high-risk
verification and route_query execute; no live database, model or demo is tested.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphragexpr.retrieval.router import route_query
from tests.high_risk.conftest import assertion_row
from .conftest import FREE_TEXT, NormalLLM, ask, linked_herb, linked_herb_rows

MANIFEST = json.loads((Path(__file__).resolve().parents[2] / "eval/demo/ppt_demo_cases.json").read_text(encoding="utf-8"))
CASES = {case["id"]: case for case in MANIFEST["cases"]}


def synthetic_contra(name, severity, suffix):
    """Artificial restriction row; page 1 is a fixture value, not a PPT citation."""
    quote = f"孕妇{severity}。"
    return assertion_row(
        assertion_id="synthetic-" + suffix, fact_key="synthetic-fact-" + suffix,
        subject_id="H:" + name, subject_name=name,
        predicate="CONTRAINDICATED_IN", polarity="NEGATIVE",
        object_id="P:孕妇", object_name="孕妇",
        evidence_quote=quote, chunk_texts=[quote], chunk_page_starts=[1], chunk_page_ends=[1],
    )


@pytest.mark.parametrize("case_id,name,severities", [
    ("fuzi_pregnancy", "附子", ["慎用"]),
    ("chuanwu_pregnancy", "川乌", ["慎用", "禁用"]),
    ("fuzi_lactation", "附子", ["慎用"]),
    ("liuwei_dihuang_pregnancy", "地黄", ["慎用"]),
])
def test_ppt_high_risk_states_with_synthetic_evidence(client, wire, case_id, name, severities):
    case = CASES[case_id]
    wiring = wire(
        rows=[synthetic_contra(name, severity, str(i)) for i, severity in enumerate(severities)],
        auto_linked=linked_herb(name, "H:" + name),
        linked_rows=linked_herb_rows(name, "H:" + name),
        auto_items=[], llm=NormalLLM(error=RuntimeError("ordinary LLM must not run")),
    )
    code, data = ask(client, case["question"], "auto")
    assert code == 200
    assert data["answer_status"] == case["expected_answer_status"]
    assert data["high_risk"]["is_high_risk"] is True
    assert data["evidence_method"] == "high_risk_assertion"
    assert wiring.llm.constructed == wiring.llm.calls == 0
    assert FREE_TEXT not in data["answer"]
    if case_id == "chuanwu_pregnancy":
        assert data["decision"] == "conflict"
        assert "慎用" in data["decision_text"] and "禁用" in data["decision_text"]
    if case_id in {"fuzi_lactation", "liuwei_dihuang_pregnancy"}:
        assert data["claims"] == []
        assert data["abstained"] is True
        assert data["sources"] == []
        assert data["subgraph"] == {"nodes": [], "edges": []}
    if case_id == "fuzi_lactation":
        assert "哺乳期" in data["decision_text"]
        assert data["decision"] == "unknown"
    if case_id == "liuwei_dihuang_pregnancy":
        # Production object detection names the longer suffix 地黄丸 in its
        # explanation. The required behavior is rejecting that compound as 地黄.
        assert "地黄丸" in data["answer"]
        assert "无法确认为同一对象" in data["answer"]


@pytest.mark.parametrize("case_id,names", [
    ("renshen_efficacy", ["人参"]),
    ("renshen_xiyangshen_compare", ["人参", "西洋参"]),
    ("tcm_history", []),
])
def test_ppt_normal_routes_use_real_router(client, wire, api_module, monkeypatch, case_id, names):
    case = CASES[case_id]
    entities = [{"canonical_id": "H:" + name, "name": name, "entity_type": "HERB"} for name in names]
    wiring = wire(llm=NormalLLM(text="synthetic ordinary response"))
    fake_retrieve = api_module._retrieve_auto

    def retrieve_with_real_router(question):
        packet, graph, sources = fake_retrieve(question)
        plan = route_query(question, entities)
        assert plan.intent == case["expected_intent"]
        packet.intent = plan.intent
        packet.trace["route"] = {"intent": plan.intent, "is_high_risk": plan.is_high_risk}
        return packet, graph, sources

    monkeypatch.setattr(api_module, "_retrieve_auto", retrieve_with_real_router)
    code, data = ask(client, case["question"], "auto")
    assert code == 200
    assert data["intent"] == case["expected_intent"]
    assert data["high_risk"]["is_high_risk"] is False
    assert wiring.llm.calls == 1
    assert data["answer"] == "synthetic ordinary response"
