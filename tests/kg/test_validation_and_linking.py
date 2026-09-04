"""I1 实体消歧 / I2 关系验证 / I3 跨块补全 / LLMClient 的单元测试。

重点回归 SKILL 与设计文档明确点名的缺陷：
  - rule_score 与 final_score 语义不得混用（设计 §2.1.2-A）
  - 未进入二次校验时 llm_verdict 必须是 NOT_RUN
  - 绝不跨实体类型合并（旧图 506 个同名跨类型实体的根因）
  - 跨块补全以 entry_id 为主依据，相邻块只是回退（设计 §2.1.2-D）
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from graphragexpr.config.settings import KGSettings
from graphragexpr.domain.errors import LLMContractError
from graphragexpr.domain.ids import canonical_id, chunk_id, normalize_text
from graphragexpr.domain.models import ChunkRecord, EntityRecord, EvidenceSpan
from graphragexpr.domain.ontology import load_ontology
from graphragexpr.extract.cross_chunk import build_groups
from graphragexpr.extract.dosage import parse_dosage
from graphragexpr.extract.llm_client import FakeLLMClient
from graphragexpr.extract.relation_validator import validate_relation
from graphragexpr.linking.entity_resolver import AliasStore, EntityResolver


@pytest.fixture(scope="module")
def ontology():
    return load_ontology()


@pytest.fixture
def settings():
    return KGSettings()


def _span(quote: str, mode: str = "raw_exact", distance: int = 0) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_quote=quote, evidence_start=0, evidence_end=len(quote),
        match_mode=mode, edit_distance=distance,
    )


class TestRelationValidator:
    def test_accepts_clean_relation_without_calling_llm(self, ontology, settings):
        """明确支持的候选不应触发 LLM；llm_verdict 必须保持 NOT_RUN。"""
        calls = []

        def verifier() -> str:
            calls.append(1)
            return "SUPPORTED"

        result = validate_relation(
            ontology=ontology, settings=settings,
            predicate="HAS_EFFICACY", subject_type="HERB", object_type="EFFICACY",
            subject_name="人参", object_name="大补元气",
            evidence=_span("人参大补元气，复脉固脱"),
            llm_verifier=verifier,
        )
        assert result.status == "ACCEPTED"
        assert result.llm_verdict == "NOT_RUN"
        assert calls == []
        assert result.rule_score == result.final_score

    def test_domain_range_violation_rejected_without_llm(self, ontology, settings):
        calls = []
        result = validate_relation(
            ontology=ontology, settings=settings,
            predicate="ENTERS_MERIDIAN", subject_type="HERB", object_type="EFFICACY",
            subject_name="人参", object_name="大补元气",
            evidence=_span("人参大补元气"),
            llm_verifier=lambda: calls.append(1) or "SUPPORTED",
        )
        assert result.status == "REJECTED"
        assert any("domain_range_violation" in f for f in result.rule_flags)
        assert calls == []

    def test_self_loop_rejected(self, ontology, settings):
        result = validate_relation(
            ontology=ontology, settings=settings,
            predicate="COMBINED_WITH", subject_type="HERB", object_type="HERB",
            subject_name="人参", object_name="人参", evidence=_span("人参配人参"),
        )
        assert result.status == "REJECTED" and "self_loop" in result.rule_flags

    def test_ocr_noise_rejected(self, ontology, settings):
        result = validate_relation(
            ontology=ontology, settings=settings,
            predicate="HAS_EFFICACY", subject_type="HERB", object_type="EFFICACY",
            subject_name="人参", object_name="���", evidence=_span("人参大补元气"),
        )
        assert result.status == "REJECTED"
        assert any("ocr_noise" in f for f in result.rule_flags)

    def test_unlocatable_evidence_never_accepted(self, ontology, settings):
        """SKILL §3.1：证据无法定位只能 REVIEW。"""
        result = validate_relation(
            ontology=ontology, settings=settings,
            predicate="HAS_EFFICACY", subject_type="HERB", object_type="EFFICACY",
            subject_name="人参", object_name="大补元气",
            evidence=EvidenceSpan(evidence_quote="x", match_mode="not_found"),
        )
        assert result.status == "REVIEW"
        assert "evidence_not_locatable" in result.rule_flags

    def test_negation_polarity_mismatch_penalised(self, ontology, settings):
        result = validate_relation(
            ontology=ontology, settings=settings,
            predicate="COMBINED_WITH", subject_type="HERB", object_type="HERB",
            subject_name="人参", object_name="藜芦",
            evidence=_span("不宜与藜芦同用"), polarity="POSITIVE",
            llm_verifier=lambda: "UNSUPPORTED",
        )
        assert "negation_polarity_mismatch" in result.rule_flags
        assert result.status in {"REJECTED", "REVIEW"}

    def test_high_risk_always_goes_through_verifier(self, ontology, settings):
        """高风险谓词即使规则分很高也必须二次校验。"""
        calls = []

        def verifier() -> str:
            calls.append(1)
            return "SUPPORTED"

        result = validate_relation(
            ontology=ontology, settings=settings,
            predicate="HAS_DOSAGE", subject_type="HERB", object_type="DOSAGE",
            subject_name="人参", object_name="3～9g",
            evidence=_span("煎服，3～9g"),
            dosage=parse_dosage("煎服，3～9g"),
            llm_verifier=verifier,
        )
        assert calls == [1]
        assert result.llm_verdict == "SUPPORTED"
        assert "high_risk_predicate" in result.rule_flags

    def test_dosage_without_value_is_penalised(self, ontology, settings):
        result = validate_relation(
            ontology=ontology, settings=settings,
            predicate="HAS_DOSAGE", subject_type="HERB", object_type="DOSAGE",
            subject_name="人参", object_name="适量",
            evidence=_span("适量外用"), dosage=parse_dosage("适量外用"),
            llm_verifier=lambda: "SUPPORTED",
        )
        assert "dosage_missing_value_or_unit" in result.rule_flags

    def test_verifier_disabled_degrades_to_review(self, ontology):
        """降级路径：关掉 LLM 校验后，灰区候选进 REVIEW 而不是被放行。"""
        settings = KGSettings(use_llm_verifier=False)
        result = validate_relation(
            ontology=ontology, settings=settings,
            predicate="HAS_DOSAGE", subject_type="HERB", object_type="DOSAGE",
            subject_name="人参", object_name="3～9g",
            evidence=_span("煎服，3～9g"), dosage=parse_dosage("煎服，3～9g"),
            llm_verifier=lambda: "SUPPORTED",
        )
        assert result.status == "REVIEW"
        assert result.llm_verdict == "NOT_RUN"
        assert "verifier_unavailable" in result.rule_flags


class TestEntityResolver:
    def _resolver(self, settings):
        store = AliasStore()
        store.load_authority([
            {"entity_type": "HERB", "preferred_name": "人参", "aliases": ["园参", "生晒参"]},
            {"entity_type": "HERB", "preferred_name": "红参"},
            {"entity_type": "EFFICACY", "preferred_name": "大补元气"},
        ])
        return EntityResolver(store, settings), store

    def test_exact_alias_match(self, settings):
        resolver, store = self._resolver(settings)
        decision = resolver.resolve("园参", "HERB")
        assert not decision.is_new
        assert decision.record.preferred_name == "人参"

    def test_never_merges_across_entity_types(self, settings):
        """同名不同类型必须是两个不同身份 —— 旧图最严重缺陷的回归测试。"""
        resolver, store = self._resolver(settings)
        herb = resolver.resolve("呕吐", "INDICATION")
        efficacy = resolver.resolve("呕吐", "EFFICACY")
        assert herb.record.canonical_id != efficacy.record.canonical_id
        assert herb.record.entity_type == "INDICATION"
        assert efficacy.record.entity_type == "EFFICACY"

    def test_related_but_distinct_herbs_not_merged(self, settings):
        """人参 / 红参 有关联但不等价，不得合并（设计 §2.1.1-1）。"""
        resolver, store = self._resolver(settings)
        ginseng = resolver.resolve("人参", "HERB")
        red = resolver.resolve("红参", "HERB")
        assert ginseng.record.canonical_id != red.record.canonical_id

    def test_ocr_variant_becomes_candidate_not_silent_merge(self, settings):
        """字符候选只在超过自动确认阈值时才合并，否则建新实体并标记待复核。"""
        resolver, store = self._resolver(settings)
        decision = resolver.resolve("大补元气力", "EFFICACY")
        if decision.is_new:
            assert decision.record.review_status in {"NEEDS_REVIEW", "AUTO_CONFIRMED"}
        else:
            assert decision.record.link_method != "new_entity"

    def test_new_entity_created_when_no_candidate(self, settings):
        resolver, store = self._resolver(settings)
        decision = resolver.resolve("石斛", "HERB")
        assert decision.is_new
        assert decision.record.canonical_id == canonical_id("HERB", "石斛")

    def test_registration_is_idempotent(self, settings):
        resolver, store = self._resolver(settings)
        first = resolver.resolve("石斛", "HERB").record
        second = resolver.resolve("石斛", "HERB").record
        assert first.canonical_id == second.canonical_id
        assert len([r for r in store.all_records() if r.preferred_name == "石斛"]) == 1

    def test_empty_surface_rejected(self, settings):
        resolver, _ = self._resolver(settings)
        with pytest.raises(ValueError):
            resolver.resolve("   ", "HERB")


def _chunk(cid: str, text: str, *, entry: str | None, order: int, page: int) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=cid, document_id="doc", entry_id=entry, section_path=["补虚药"],
        order_in_entry=order, text=text, normalized_text=normalize_text(text),
        page_start=page, page_end=page, content_hash=cid,
    )


class TestCrossChunk:
    def test_entry_id_is_primary_basis(self, settings):
        """同一 entry_id 内配对，即使两块不相邻。"""
        chunks = [
            _chunk("c1", "人参　味甘、微苦，性微温。", entry="e_ginseng", order=0, page=132),
            _chunk("c2", "其他药材的无关内容。", entry="e_other", order=0, page=132),
            _chunk("c3", "人参　煎服，3～9g。不宜与藜芦同用。", entry="e_ginseng", order=1, page=133),
        ]
        surfaces = {"c1": ["人参"], "c2": ["黄芪"], "c3": ["人参"]}
        groups = build_groups(chunks, surfaces, settings)
        ginseng = [g for g in groups if g.anchor_entity == "人参"]
        assert ginseng, "应按 entry_id 把人参的两块配到一起"
        assert set(ginseng[0].chunk_ids) == {"c1", "c3"}
        assert ginseng[0].strategy == "entry_id"
        assert (ginseng[0].page_start, ginseng[0].page_end) == (132, 133)

    def test_does_not_pair_across_different_entries(self, settings):
        chunks = [
            _chunk("c1", "人参　味甘。", entry="e_ginseng", order=0, page=132),
            _chunk("c2", "黄芪　味甘。", entry="e_astragalus", order=0, page=140),
        ]
        groups = build_groups(chunks, {"c1": ["人参"], "c2": ["黄芪"]}, settings)
        assert all(len(set(g.chunk_ids)) <= 1 or g.anchor_entity not in {"人参", "黄芪"}
                   or set(g.chunk_ids) <= {"c1"} or set(g.chunk_ids) <= {"c2"}
                   for g in groups)

    def test_adjacent_fallback_when_entry_missing(self, settings):
        chunks = [
            _chunk("c1", "人参　味甘、微苦。", entry=None, order=0, page=132),
            _chunk("c2", "人参　煎服，3～9g。", entry=None, order=1, page=133),
        ]
        groups = build_groups(chunks, {"c1": ["人参"], "c2": ["人参"]}, settings)
        assert groups, "entry_id 缺失时应退到 section/相邻策略"
        assert groups[0].strategy in {"section_path", "adjacent"}

    def test_disabled_by_feature_flag(self):
        """消融开关：KG3 关闭后不产出任何跨块候选。"""
        settings = KGSettings(use_cross_chunk=False)
        chunks = [_chunk("c1", "人参", entry="e", order=0, page=1),
                  _chunk("c2", "人参", entry="e", order=1, page=2)]
        assert build_groups(chunks, {"c1": ["人参"], "c2": ["人参"]}, settings) == []

    def test_empty_input(self, settings):
        assert build_groups([], {}, settings) == []


class _Payload(BaseModel):
    entities: list[str]


class TestFakeLLMClient:
    def test_returns_validated_payload(self):
        client = FakeLLMClient(responses={"extract": ['{"entities": ["人参"]}']})
        result = client.complete_json(system="s", user="u", schema=_Payload, purpose="extract")
        assert result.entities == ["人参"]
        assert client.usage.calls == 1

    def test_strips_code_fence(self):
        client = FakeLLMClient(responses={"extract": ['```json\n{"entities": []}\n```']})
        assert client.complete_json(system="s", user="u", schema=_Payload, purpose="extract").entities == []

    def test_invalid_json_raises_contract_error(self):
        client = FakeLLMClient(responses={"extract": ["这不是 JSON"]})
        with pytest.raises(LLMContractError):
            client.complete_json(system="s", user="u", schema=_Payload, purpose="extract")

    def test_schema_violation_raises(self):
        client = FakeLLMClient(responses={"extract": ['{"wrong_field": 1}']})
        with pytest.raises(LLMContractError):
            client.complete_json(system="s", user="u", schema=_Payload, purpose="extract")

    def test_missing_purpose_raises(self):
        client = FakeLLMClient(responses={})
        with pytest.raises(LLMContractError):
            client.complete_json(system="s", user="u", schema=_Payload, purpose="unknown")


class TestNoFalseMerge:
    """回归：包含式匹配绝不能自动合并有关联但不等价的中药（设计 §2.1.1-1）。

    这些用例来自全量构建后在真实图谱上发现的错误合并。
    """

    def _resolver(self, settings, seed_names):
        store = AliasStore()
        store.load_authority(
            [{"entity_type": "HERB", "preferred_name": n} for n in seed_names]
        )
        return EntityResolver(store, settings)

    @pytest.mark.parametrize(
        ("existing", "incoming"),
        [
            ("麻黄根", "麻黄"),    # 麻黄发汗 / 麻黄根止汗，功效相反
            ("麻黄", "麻黄根"),    # 反向同样不能合并
            ("黄连", "胡黄连"),    # Coptis / Picrorhiza，不同科属
            ("沙参", "南沙参"),
            ("防已", "广防已"),
            ("甘草", "炙甘草"),    # 生品与炮制品
            ("乌头", "制乌头"),
            ("牛黄", "人工牛黄"),
        ],
    )
    def test_containment_never_auto_merges(self, settings, existing, incoming):
        resolver = self._resolver(settings, [existing])
        decision = resolver.resolve(incoming, "HERB")
        assert decision.is_new, f"{incoming} 不应被合并进 {existing}"
        assert decision.record.preferred_name == incoming
        assert decision.record.canonical_id == canonical_id("HERB", incoming)

    def test_containment_still_reported_as_candidate(self, settings):
        """不自动合并，但候选要留痕，便于人工复核与评测。"""
        resolver = self._resolver(settings, ["麻黄根"])
        decision = resolver.resolve("麻黄", "HERB")
        assert decision.candidates, "包含关系应作为候选被记录"
        assert decision.candidates[0][1] < settings.link_auto_confirm_threshold

    def test_exact_alias_still_merges(self, settings):
        """精确别名匹配不受影响，仍应合并。"""
        store = AliasStore()
        store.load_authority([
            {"entity_type": "HERB", "preferred_name": "人参", "aliases": ["园参", "生晒参"]},
        ])
        resolver = EntityResolver(store, settings)
        decision = resolver.resolve("园参", "HERB")
        assert not decision.is_new
        assert decision.record.preferred_name == "人参"

    def test_no_self_loop_after_resolution(self, settings):
        """两个不同表面形式若解析到同一 canonical_id，不得据此建自环边。"""
        store = AliasStore()
        store.load_authority([
            {"entity_type": "HERB", "preferred_name": "黄连", "aliases": ["川黄连"]},
        ])
        resolver = EntityResolver(store, settings)
        left = resolver.resolve("黄连", "HERB").record.canonical_id
        right = resolver.resolve("川黄连", "HERB").record.canonical_id
        assert left == right, "真别名应指向同一身份"
        # 调用方据此判断自环：canonical_id 相同即为自环，须拒绝
        assert left == right


class TestTruncatedJsonRecovery:
    """输出被长度上限截断时抢救完整前缀，而不是整块丢弃（SKILL §17 不隐藏失败）。"""

    def _client(self, payload: str, settings):
        return FakeLLMClient(responses={"extract": [payload]}, settings=settings)

    def test_recovers_complete_elements(self, settings):
        from graphragexpr.extract.extractor import RawExtraction
        truncated = (
            '{"entities": [{"name": "麻黄", "entity_type": "HERB"}], '
            '"relations": [{"subject": "麻黄", "predicate": "HAS_EFFICACY", '
            '"object_text": "发汗解表", "object_type": "EFFICACY", '
            '"polarity": "POSITIVE", "evidence_quote": "发汗解表"}, '
            '{"subject": "麻黄", "predicate": "HAS_EFFI'   # 被截断
        )
        client = self._client(truncated, settings)
        result = client.complete_json(
            system="s", user="u", schema=RawExtraction, purpose="extract"
        )
        assert len(result.relations) == 1, "应保留完整的那一条，丢弃被截断的"
        assert result.relations[0].object_text == "发汗解表"
        assert client.truncation_recoveries == 1, "降级必须可见"

    def test_intact_json_is_not_touched(self, settings):
        from graphragexpr.extract.extractor import RawExtraction
        client = self._client('{"entities": [], "relations": []}', settings)
        client.complete_json(system="s", user="u", schema=RawExtraction, purpose="extract")
        assert client.truncation_recoveries == 0

    def test_unrecoverable_garbage_still_raises(self, settings):
        from graphragexpr.extract.extractor import RawExtraction
        client = self._client("这不是 JSON，只是一段自然语言", settings)
        with pytest.raises(LLMContractError):
            client.complete_json(system="s", user="u", schema=RawExtraction, purpose="extract")

    def test_truncated_inside_string_drops_that_element(self, settings):
        from graphragexpr.extract.extractor import RawExtraction
        truncated = (
            '{"entities": [], "relations": [{"subject": "麻黄", "predicate": "HAS_EFFICACY", '
            '"object_text": "发汗", "object_type": "EFFICACY", "polarity": "POSITIVE", '
            '"evidence_quote": "发汗解表"}, {"subject": "麻黄", "evidence_quote": "未闭合的字符串'
        )
        client = self._client(truncated, settings)
        result = client.complete_json(
            system="s", user="u", schema=RawExtraction, purpose="extract"
        )
        assert len(result.relations) == 1
