"""测试后端不再依赖 neo4j_graphrag，前端默认自动路由。"""

from pathlib import Path


def test_backend_api_does_not_depend_on_neo4j_graphrag():
    """测试：后端API不依赖当Neo4jGraphRAG。"""
    api_source = (Path(__file__).parents[2] / "backend" / "api.py").read_text(
        encoding="utf-8"
    )

    assert "neo4j_graphrag" not in api_source


def test_frontend_defaults_to_automatic_routing():
    """测试：前端默认到自动路由。"""
    frontend = (Path(__file__).parents[2] / "frontend" / "index.html").read_text(
        encoding="utf-8"
    )

    assert '<option value="auto" selected>' in frontend
