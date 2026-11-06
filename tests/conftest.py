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


@pytest.fixture(scope="session")
def neo4j_driver():
    """Provide Neo4j driver for tests"""
    if not all(os.getenv(name) for name in ("NEO4J_URL", "NEO4J_USER", "NEO4J_PASSWORD")):
        pytest.skip("Neo4j integration environment is not configured")
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URL"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
    )
    yield driver
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
def clean_database(neo4j_driver):
    """Provide an empty database for tests that need one.

    安全护栏：
    本 fixture 会清空整个数据库。若目标库里已有数据，默认**跳过测试**而不是删除——
    否则任何人执行一次 `pytest` 都会毁掉现网图谱。

    需要在真实非空库上跑破坏性测试时，显式设置：
        ALLOW_DESTRUCTIVE_TESTS=true
    更推荐的做法是把 NEO4J_URL 指向一个专用的空实例。
    """
    try:
        neo4j_driver.verify_connectivity()
    except Exception as exc:
        pytest.skip(f"Neo4j not available: {exc}")

    database = os.getenv("NEO4J_DB") or None
    with neo4j_driver.session(database=database) as session:
        existing = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]

    allow_destructive = (os.getenv("ALLOW_DESTRUCTIVE_TESTS") or "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if existing and not allow_destructive:
        pytest.skip(
            f"目标库非空（{existing} 个节点），已跳过以避免清空现网数据。"
            "请将 NEO4J_URL 指向空实例，或设置 ALLOW_DESTRUCTIVE_TESTS=true。"
        )

    with neo4j_driver.session(database=database) as session:
        session.run("MATCH (n) DETACH DELETE n")
    yield
    with neo4j_driver.session(database=database) as session:
        session.run("MATCH (n) DETACH DELETE n")


@pytest.fixture(scope="session")
def sample_text():
    """Provide sample text for testing"""
    return """
人参味甘、微苦，性微温，具有大补元气等功效。
"""
