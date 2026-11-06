"""Local-only defaults, explicit CORS origins and launcher health identity."""

import os
from pathlib import Path
from types import SimpleNamespace

from flask import Flask
import pytest
import yaml


def test_server_defaults_disable_debugging_and_external_listening(api_module, monkeypatch):
    for key in ("BACKEND_HOST", "BACKEND_PORT", "BACKEND_DEBUG"):
        monkeypatch.delenv(key, raising=False)
    assert api_module._server_options() == {
        "host": "127.0.0.1", "port": 5000, "debug": False, "use_reloader": False,
    }


def test_server_settings_can_be_overridden_explicitly(api_module, monkeypatch):
    monkeypatch.setenv("BACKEND_HOST", "0.0.0.0")
    monkeypatch.setenv("BACKEND_PORT", "5100")
    monkeypatch.setenv("BACKEND_DEBUG", "true")
    assert api_module._server_options() == {
        "host": "0.0.0.0", "port": 5100, "debug": True, "use_reloader": False,
    }


@pytest.mark.parametrize("port", ["0", "-1", "65536", "not-a-port"])
def test_invalid_port_fails_before_startup(api_module, monkeypatch, port):
    monkeypatch.setenv("BACKEND_PORT", port)
    with pytest.raises(ValueError):
        api_module._server_options()


@pytest.mark.parametrize("limit", ["0", "-1", "invalid"])
def test_invalid_request_limit_is_rejected(api_module, monkeypatch, limit):
    monkeypatch.setenv("BACKEND_MAX_REQUEST_BYTES", limit)
    with pytest.raises(ValueError):
        api_module._configure_http(Flask(__name__))


@pytest.mark.parametrize("origins,origin,allowed", [
    ("", "https://unrelated.example", False),
    ("http://localhost:3000", "http://localhost:3000", True),
    ("http://localhost:3000", "https://unrelated.example", False),
    ("http://localhost:3000, https://ui.example", "https://ui.example", True),
])
def test_cors_is_opt_in_and_matches_configured_origins(
    api_module, monkeypatch, origins, origin, allowed
):
    monkeypatch.setenv("CORS_ORIGINS", origins)
    monkeypatch.delenv("BACKEND_MAX_REQUEST_BYTES", raising=False)
    application = Flask(__name__)
    api_module._configure_http(application)
    application.add_url_rule("/api/test", view_func=lambda: {"status": "ok"})
    assert application.config["MAX_CONTENT_LENGTH"] == 65536
    response = application.test_client().get("/api/test", headers={"Origin": origin})
    assert response.headers.get("Access-Control-Allow-Origin") == (origin if allowed else None)


def test_health_identifies_the_process_only_after_database_check(client, api_module, monkeypatch):
    calls = []
    monkeypatch.setattr(api_module, "driver", SimpleNamespace(
        verify_connectivity=lambda: calls.append("verify")
    ))
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["pid"] == os.getpid()
    assert calls == ["verify"]


def test_unavailable_database_is_not_reported_as_healthy(client, api_module, monkeypatch):
    def unavailable():
        raise ConnectionError("unavailable")
    monkeypatch.setattr(api_module, "driver", SimpleNamespace(verify_connectivity=unavailable))
    response = client.get("/api/health")
    assert response.status_code == 500
    assert response.get_json()["status"] == "error"


def test_compose_requires_password_and_publishes_only_on_loopback():
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "docker-compose.yml").read_text())
    service = config["services"]["neo4j"]
    assert all(port.startswith("127.0.0.1:") for port in service["ports"])
    assert "${NEO4J_PASSWORD:?" in service["environment"]["NEO4J_AUTH"]
    example = dict(
        line.split("=", 1) for line in (root / ".env.example").read_text().splitlines()
        if line and not line.startswith("#")
    )
    assert example["NEO4J_PASSWORD"] == ""
