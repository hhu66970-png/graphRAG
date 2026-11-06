# tests/conftest.py
"""
Pytest configuration and fixtures for GraphRAG tests
"""
import pytest
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# Driver creation parses the URI during API import; it does not connect here.
os.environ.setdefault("NEO4J_URL", "bolt://localhost:7687")


def _integration_settings():
    """Require explicit credentials and permission for the disposable test database."""
    if (os.getenv("ALLOW_DESTRUCTIVE_TESTS") or "").strip().lower() not in {"true", "1", "yes"}:
        pytest.skip("集成测试需显式设置 ALLOW_DESTRUCTIVE_TESTS=true")
    values = {key: os.getenv("NEO4J_TEST_" + key) for key in ("URL", "USER", "PASSWORD", "DB")}
    if not all(value and value.strip() for value in values.values()):
        pytest.skip("请配置独立 NEO4J_TEST_URL/USER/PASSWORD/DB，不读取业务库配置")
    if (values["URL"] == os.getenv("NEO4J_URL")
            and values["DB"] == (os.getenv("NEO4J_DB") or "neo4j")):
        pytest.skip("NEO4J_TEST_* 指向业务库；请使用独立测试实例或数据库")
    return {
        "url": values["URL"], "user": values["USER"],
        "password": values["PASSWORD"], "database": values["DB"],
    }


@pytest.fixture(scope="session")
def neo4j_test_config():
    return _integration_settings()


@pytest.fixture(scope="session")
def neo4j_driver(neo4j_test_config):
    driver = GraphDatabase.driver(
        neo4j_test_config["url"],
        auth=(neo4j_test_config["user"], neo4j_test_config["password"]),
    )
    try:
        yield driver
    finally:
        driver.close()


@pytest.fixture(scope="session")
def neo4j_connection_test(neo4j_driver):
    """Verify Neo4j connection before running tests"""
    try:
        neo4j_driver.verify_connectivity()
        return True
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")


@pytest.fixture
def clean_database(neo4j_driver, neo4j_test_config):
    """Clear only the explicitly authorized test database before and after a test."""
    try:
        neo4j_driver.verify_connectivity()
    except Exception as exc:
        pytest.skip(f"Neo4j not available: {exc}")

    database = neo4j_test_config["database"]
    with neo4j_driver.session(database=database) as session:
        session.run("MATCH (n) DETACH DELETE n").consume()
    try:
        yield
    finally:
        with neo4j_driver.session(database=database) as session:
            session.run("MATCH (n) DETACH DELETE n").consume()


@pytest.fixture(scope="session")
def sample_text():
    """Provide sample text for testing"""
    return """
人参味甘、微苦，性微温，具有大补元气等功效。
"""
