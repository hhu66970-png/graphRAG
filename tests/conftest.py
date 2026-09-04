# tests/conftest.py
"""
Pytest configuration and fixtures for GraphRAG tests
"""
import pytest
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


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
    """Clean database before each test"""
    try:
        neo4j_driver.verify_connectivity()
    except Exception as exc:
        pytest.skip(f"Neo4j not available: {exc}")
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    yield
    # Cleanup after test
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


@pytest.fixture(scope="session")
def sample_text():
    """Provide sample text for testing"""
    return """
人参味甘、微苦，性微温，具有大补元气等功效。
"""
