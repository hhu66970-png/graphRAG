"""Configuration gates run before any integration driver is created."""

import pytest

from tests.conftest import _integration_settings


@pytest.fixture
def configured_test_database(monkeypatch):
    for key, value in {
        "URL": "bolt://127.0.0.1:17687", "USER": "test-user",
        "PASSWORD": "test-password", "DB": "disposable",
    }.items():
        monkeypatch.setenv("NEO4J_TEST_" + key, value)
    monkeypatch.setenv("NEO4J_URL", "bolt://business.example:7687")
    monkeypatch.setenv("NEO4J_USER", "business-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "business-password")
    monkeypatch.setenv("NEO4J_DB", "business")
    monkeypatch.setenv("ALLOW_DESTRUCTIVE_TESTS", "true")


@pytest.mark.parametrize("permission", [None, "", "false", "0"])
def test_integration_requires_explicit_permission(
    monkeypatch, configured_test_database, permission
):
    if permission is None:
        monkeypatch.delenv("ALLOW_DESTRUCTIVE_TESTS", raising=False)
    else:
        monkeypatch.setenv("ALLOW_DESTRUCTIVE_TESTS", permission)
    with pytest.raises(pytest.skip.Exception, match="ALLOW_DESTRUCTIVE_TESTS"):
        _integration_settings()


@pytest.mark.parametrize("missing", ["URL", "USER", "PASSWORD", "DB"])
def test_business_configuration_cannot_fill_missing_test_settings(
    monkeypatch, configured_test_database, missing
):
    monkeypatch.delenv("NEO4J_TEST_" + missing)
    with pytest.raises(pytest.skip.Exception, match="NEO4J_TEST"):
        _integration_settings()


def test_explicit_business_database_is_still_rejected(monkeypatch, configured_test_database):
    monkeypatch.setenv("NEO4J_TEST_URL", "bolt://business.example:7687")
    monkeypatch.setenv("NEO4J_TEST_DB", "business")
    with pytest.raises(pytest.skip.Exception, match="指向业务库"):
        _integration_settings()


def test_test_configuration_is_used_without_business_defaults(configured_test_database):
    assert _integration_settings() == {
        "url": "bolt://127.0.0.1:17687", "user": "test-user",
        "password": "test-password", "database": "disposable",
    }


def test_separate_test_database_on_same_server_is_allowed(monkeypatch, configured_test_database):
    monkeypatch.setenv("NEO4J_TEST_URL", "bolt://business.example:7687")
    assert _integration_settings()["database"] == "disposable"
