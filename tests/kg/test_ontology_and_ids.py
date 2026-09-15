"""受控本体与稳定 ID 的契约测试（SKILL §14.2）。"""

from __future__ import annotations

import pytest

from graphragexpr.domain.errors import (
    DomainRangeViolation,
    UnknownEntityTypeError,
    UnknownPredicateError,
)
from graphragexpr.domain.ids import (
    canonical_id,
    chunk_id,
    fact_key,
    normalize_text,
    new_run_id,
)
from graphragexpr.domain.ontology import load_ontology


@pytest.fixture(scope="module")
def ontology():
    """测试夹具：加载受控本体。"""
    return load_ontology()


class TestOntology:
    """测试类：覆盖「Ontology」相关行为。"""
    def test_loads_and_self_validates(self, ontology):
        """测试：加载且自我自校验。"""
        assert ontology.schema_version
        assert "HERB" in ontology.entity_types
        assert "HAS_EFFICACY" in ontology.predicates

    def test_unknown_entity_type_is_rejected_not_downgraded(self, ontology):
        """SKILL §3.4：未知类型必须拒绝，不得统一降级为「概念」。

        这是旧实现最严重的静默失真：normalize_entity_type() 把一切未知类型
        变成「概念」，造成 982 个概念节点。
        """
        with pytest.raises(UnknownEntityTypeError):
            ontology.resolve_entity_type("经络腧穴")
        # 且不得返回 CONCEPT
        assert ontology.resolve_entity_type("中药") == "HERB"

    def test_unknown_predicate_is_rejected(self, ontology):
        """测试：未知谓词为被拒绝。"""
        with pytest.raises(UnknownPredicateError):
            ontology.resolve_predicate("主编")

    def test_legacy_aliases_map_to_stable_ids(self, ontology):
        """测试：旧别名映射到稳定ID。"""
        assert ontology.resolve_predicate("具有功效") == "HAS_EFFICACY"
        assert ontology.resolve_predicate("功效为") == "HAS_EFFICACY"
        assert ontology.resolve_predicate("相反") == "INCOMPATIBLE_WITH"
        assert ontology.resolve_entity_type("药材") == "HERB"

    def test_domain_range_enforced(self, ontology):
        """测试：定义域区间被强制。"""
        ontology.check_domain_range("HAS_EFFICACY", "HERB", "EFFICACY")
        with pytest.raises(DomainRangeViolation):
            ontology.check_domain_range("HAS_EFFICACY", "EFFICACY", "HERB")
        with pytest.raises(DomainRangeViolation):
            ontology.check_domain_range("ENTERS_MERIDIAN", "HERB", "EFFICACY")

    def test_high_risk_predicates_declared(self, ontology):
        """测试：高风险predicates已声明。"""
        high_risk = set(ontology.high_risk_predicates())
        assert {"HAS_DOSAGE", "CONTRAINDICATED_IN", "INCOMPATIBLE_WITH"} <= high_risk
        assert not ontology.is_high_risk("HAS_EFFICACY")

    def test_prompt_block_is_rendered_from_ontology(self, ontology):
        """测试：prompt块为渲染来自ontology。"""
        block = ontology.prompt_schema_block()
        assert "HAS_DOSAGE" in block and "HERB" in block

    def test_empty_input_rejected(self, ontology):
        """测试：空输入被拒绝。"""
        with pytest.raises(UnknownEntityTypeError):
            ontology.resolve_entity_type("")


class TestStableIds:
    """测试类：覆盖「StableIds」相关行为。"""
    def test_normalize_strips_whitespace_and_fullwidth(self):
        """测试：normalize去掉空白且全角。"""
        assert normalize_text("归脾、肺 经") == normalize_text("归脾、肺经")
        assert normalize_text("9～30g") == normalize_text("9~30g")

    def test_canonical_id_is_type_scoped(self):
        """同名不同类型必须得到不同身份 —— 旧图 506 个同名跨类型实体的正解。"""
        herb = canonical_id("HERB", "呕吐")
        indication = canonical_id("INDICATION", "呕吐")
        assert herb != indication
        assert herb.startswith("HERB:") and indication.startswith("INDICATION:")

    def test_canonical_id_stable_across_surface_variants(self):
        """测试：canonicalID稳定跨表面形式变体。"""
        assert canonical_id("HERB", "人 参") == canonical_id("HERB", "人参")

    def test_chunk_id_changes_with_content(self):
        """测试：文本块ID随…变化带内容。"""
        a = chunk_id("doc1", ["总论"], "内容甲")
        b = chunk_id("doc1", ["总论"], "内容乙")
        assert a != b
        assert a == chunk_id("doc1", ["总论"], "内容甲")

    def test_fact_key_is_order_insensitive_for_qualifiers(self):
        """测试：fact键为顺序无关用于限定符。"""
        left = fact_key("HERB:x", "HAS_DOSAGE", "9-30g", {"route": "煎服", "population": "成人"})
        right = fact_key("HERB:x", "HAS_DOSAGE", "9-30g", {"population": "成人", "route": "煎服"})
        assert left == right

    def test_fact_key_distinguishes_polarity(self):
        """测试：fact键区分极性。"""
        positive = fact_key("HERB:x", "COMBINED_WITH", "HERB:y", {}, "POSITIVE")
        negative = fact_key("HERB:x", "COMBINED_WITH", "HERB:y", {}, "NEGATIVE")
        assert positive != negative

    def test_run_id_is_monotonic_and_unique(self):
        """测试：runID为单调递增且唯一。"""
        first, second = new_run_id(), new_run_id()
        assert first != second
        assert first.split("-")[0] <= second.split("-")[0]
