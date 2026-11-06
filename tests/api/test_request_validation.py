"""Reject malformed requests before invoking retrieval or generation."""

import json

import pytest


@pytest.fixture
def no_backend_calls(api_module, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid input reached a backend dependency")

    for name in (
        "_is_high_risk_question", "_high_risk_response", "_retrieve_auto",
        "_retrieve_keyword_graph", "_configured_embedder", "OpenAI",
    ):
        monkeypatch.setattr(api_module, name, unexpected)


@pytest.mark.parametrize("payload", [
    None, [], ["text"], "text", 1, True,
    {}, {"question": None}, {"question": 1}, {"question": True},
    {"question": []}, {"question": {}}, {"question": ""},
    {"question": " \t\n\u3000"},
    {"question": "麻黄的功效", "method": None},
    {"question": "麻黄的功效", "method": []},
    {"question": "麻黄的功效", "method": {}},
    {"question": "麻黄的功效", "method": 1},
    {"question": "麻黄的功效", "method": "unknown"},
])
def test_invalid_json_payloads_are_400_json(client, no_backend_calls, payload):
    response = client.post(
        "/api/graphrag", data=json.dumps(payload), content_type="application/json"
    )
    assert response.status_code == 400
    assert response.is_json
    assert isinstance(response.get_json()["error"], str)


@pytest.mark.parametrize("body,content_type", [
    ('{"question":', "application/json"),
    ('{"question":"麻黄"}', "text/plain"),
    ("question=麻黄", "application/x-www-form-urlencoded"),
    ("", "application/json"),
])
def test_malformed_body_is_400_json(client, no_backend_calls, body, content_type):
    response = client.post("/api/graphrag", data=body, content_type=content_type)
    assert response.status_code == 400
    assert response.is_json
    assert "error" in response.get_json()


def test_oversized_body_is_413_json(client, api_module, monkeypatch, no_backend_calls):
    monkeypatch.setitem(api_module.app.config, "MAX_CONTENT_LENGTH", 128)
    response = client.post("/api/graphrag", json={"question": "a" * 129})
    assert response.status_code == 413
    assert response.get_json() == {"error": "Request body is too large"}


def test_valid_question_is_trimmed_and_uses_default_method(client, wire):
    wiring = wire(rows=[])
    response = client.post("/api/graphrag", json={"question": "  麻黄的功效是什么？ \n"})
    assert response.status_code == 200
    assert response.get_json()["method"] == "auto"
    assert wiring.auto_calls == [{"question": "麻黄的功效是什么？"}]
    assert wiring.llm.calls == 1
