"""Unit tests for domain normalization and LLM JSON parsing."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "graphragexpr" / "extract"))

from build_kg_dyn import (
    normalize_entity_type,
    normalize_relationship_type,
    parse_json_response,
)


def test_parse_json_response_accepts_markdown_fence():
    payload = {
        "entities": [{"name": "人参", "type": "中药"}],
        "relationships": [],
    }
    result = parse_json_response(f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```")
    assert result["entities"][0]["name"] == "人参"


def test_parse_json_response_finds_surrounding_object():
    result = parse_json_response('说明如下：{"entities": [], "relationships": []}。')
    assert result == {"entities": [], "relationships": []}


def test_entity_type_alias_and_allowlist():
    assert normalize_entity_type("药材") == "中药"
    assert normalize_entity_type("公司") == "概念"


def test_relationship_type_is_safe_for_cypher():
    assert normalize_relationship_type("具有 功效`) MATCH (n)") == "具有_功效_MATCH_n"
    assert normalize_relationship_type("") == "相关"
