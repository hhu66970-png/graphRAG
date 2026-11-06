"""抽取管线端到端测试（FakeLLM，不依赖在线 API）。

覆盖 2.1.1–2.1.3 的整合行为：严格 Schema -> I1 链接 -> 证据定位 -> I2 验证 -> Assertion。
"""

from __future__ import annotations

import json

import pytest

from graphragexpr.config.settings import KGSettings
from graphragexpr.domain.ids import canonical_id, normalize_text
from graphragexpr.domain.models import ChunkRecord
from graphragexpr.domain.ontology import load_ontology
from graphragexpr.extract.extractor import ChunkExtractor
from graphragexpr.extract.llm_client import FakeLLMClient
from graphragexpr.linking.entity_resolver import AliasStore, EntityResolver

CHUNK_TEXT = (
    "人参　味甘、微苦，性微温。归脾、肺、心、肾经。\n"
    "大补元气，复脉固脱，补脾益肺。煎服，3～9g。不宜与藜芦同用。"
)


@pytest.fixture(scope="module")
def ontology():
    """测试夹具：加载受控本体。"""
    return load_ontology()


@pytest.fixture
def chunk():
    """测试夹具：单个 Chunk。"""
    return ChunkRecord(
        chunk_id="c-ginseng-1", document_id="doc-zyx", entry_id="entry-ginseng",
        section_path=["补虚药", "补气药"], order_in_entry=0,
        text=CHUNK_TEXT, normalized_text=normalize_text(CHUNK_TEXT),
        page_start=132, page_end=132, content_hash="hash-1",
    )


def _extraction(relations: list[dict], entities: list[dict] | None = None) -> str:
    """测试用：构造 LLM 抽取 JSON。"""
    return json.dumps(
        {
            "entities": entities if entities is not None else [
                {"name": "人参", "entity_type": "HERB"},
                {"name": "大补元气", "entity_type": "EFFICACY"},
                {"name": "脾经", "entity_type": "MERIDIAN"},
                {"name": "藜芦", "entity_type": "HERB"},
            ],
            "relations": relations,
        },
        ensure_ascii=False,
    )


def _build(ontology, settings, extraction_json: str, verdicts: list[str] | None = None):
    """测试用：跑抽取管线并返回结果。"""
    responses = {"extract": [extraction_json]}
    if verdicts:
        responses["verify"] = [json.dumps({"verdict": v}) for v in verdicts]
    llm = FakeLLMClient(responses=responses, settings=settings)
    resolver = EntityResolver(AliasStore(), settings)
    extractor = ChunkExtractor(
        ontology=ontology, settings=settings, resolver=resolver, llm=llm, run_id="run-test"
    )
    return extractor, llm


class TestExtractorPipeline:
    def test_clean_relation_becomes_accepted_assertion(self, ontology, chunk):
        settings = KGSettings()
        extractor, llm = _build(ontology, settings, _extraction([
            {"subject": "人参", "predicate": "HAS_EFFICACY", "object_text": "大补元气",
             "object_type": "EFFICACY", "polarity": "POSITIVE",
             "evidence_quote": "大补元气，复脉固脱"},
        ]))
        outcome = extractor.extract(chunk)

        assert len(outcome.assertions) == 1
        record = outcome.assertions[0]
        assert record.validation_status == "ACCEPTED"
        assert record.predicate == "HAS_EFFICACY"
        assert record.subject_id == canonical_id("HERB", "人参")
        assert record.object_id == canonical_id("EFFICACY", "大补元气")
        assert record.page_start == 132 and record.chunk_ids == ["c-ginseng-1"]
        # 证据必须能回到原文跨度
        assert chunk.text[record.evidence_start:record.evidence_end] == record.evidence_quote
        assert record.match_mode in {"raw_exact", "raw_span", "normalized_exact"}
        record.assert_publishable()

    def test_unknown_predicate_is_recorded_and_rejected(self, ontology, chunk):
        """未知谓词必须记录并拒绝，不得静默塞进图里。

        旧实现会把「主编」「习称」这类词直接变成关系类型，造成 182 种关系。
        """
        settings = KGSettings()
        extractor, _ = _build(ontology, settings, _extraction([
            {"subject": "人参", "predicate": "主编", "object_text": "大补元气",
             "object_type": "EFFICACY", "polarity": "POSITIVE", "evidence_quote": "大补元气"},
        ]))
        outcome = extractor.extract(chunk)
        assert outcome.assertions == []
        assert "主编" in outcome.unknown_predicates
        assert outcome.rejected[0]["reason"] == "unknown_predicate"

    def test_unknown_entity_type_not_downgraded_to_concept(self, ontology, chunk):
        settings = KGSettings()
        extractor, _ = _build(
            ontology, settings,
            _extraction(
                relations=[],
                entities=[{"name": "手太阴肺经", "entity_type": "经络"}],
            ),
        )
        outcome = extractor.extract(chunk)
        assert "经络" in outcome.unknown_types
        # 不得产生一个 CONCEPT 实体来兜底
        assert outcome.assertions == []

    def test_domain_range_violation_rejected(self, ontology, chunk):
        settings = KGSettings()
        extractor, _ = _build(ontology, settings, _extraction([
            {"subject": "大补元气", "predicate": "ENTERS_MERIDIAN", "object_text": "人参",
             "object_type": "HERB", "polarity": "POSITIVE", "evidence_quote": "归脾、肺、心、肾经"},
        ]))
        outcome = extractor.extract(chunk)
        assert outcome.assertions == []
        assert outcome.rejected[0]["reason"] == "validation_rejected"

    def test_fabricated_evidence_cannot_be_accepted(self, ontology, chunk):
        """模型编造证据时，四级定位失败 -> 只能 REVIEW，绝不入正式事实图。"""
        settings = KGSettings()
        extractor, _ = _build(ontology, settings, _extraction([
            {"subject": "人参", "predicate": "HAS_EFFICACY", "object_text": "大补元气",
             "object_type": "EFFICACY", "polarity": "POSITIVE",
             "evidence_quote": "人参能治疗一切疑难杂症并延年益寿"},
        ]))
        outcome = extractor.extract(chunk)
        assert len(outcome.assertions) == 1
        record = outcome.assertions[0]
        assert record.validation_status == "REVIEW"
        assert record.match_mode == "not_found"
        record.assert_publishable()  # REVIEW 不触发门槛

    def test_dosage_becomes_value_object_with_high_risk_verification(self, ontology, chunk):
        settings = KGSettings()
        extractor, llm = _build(ontology, settings, _extraction([
            {"subject": "人参", "predicate": "HAS_DOSAGE", "object_text": "煎服，3～9g",
             "object_type": "DOSAGE", "polarity": "POSITIVE", "evidence_quote": "煎服，3～9g"},
        ]), verdicts=["SUPPORTED"])
        outcome = extractor.extract(chunk)

        record = outcome.assertions[0]
        assert record.object_id is None, "剂量不建实体身份"
        assert record.object_value["value_min"] == 3.0
        assert record.object_value["value_max"] == 9.0
        assert record.object_value["unit"] == "g"
        assert record.object_value["route"] == "煎服"
        # 高风险谓词必须走过二次校验
        assert any(purpose == "verify" for purpose, _ in llm.seen)

    def test_negation_produces_negative_polarity_not_positive_edge(self, ontology, chunk):
        """「不宜与藜芦同用」不能变成一条肯定的配伍边。"""
        settings = KGSettings()
        extractor, _ = _build(ontology, settings, _extraction([
            {"subject": "人参", "predicate": "INCOMPATIBLE_WITH", "object_text": "藜芦",
             "object_type": "HERB", "polarity": "NEGATIVE",
             "evidence_quote": "不宜与藜芦同用"},
        ]), verdicts=["SUPPORTED"])
        outcome = extractor.extract(chunk)
        record = outcome.assertions[0]
        assert record.polarity == "NEGATIVE"
        assert record.predicate == "INCOMPATIBLE_WITH"

    def test_same_name_different_type_gets_distinct_identity(self, ontology, chunk):
        settings = KGSettings()
        extractor, _ = _build(ontology, settings, _extraction(
            relations=[
                {"subject": "人参", "predicate": "TREATS", "object_text": "呕吐",
                 "object_type": "INDICATION", "polarity": "POSITIVE",
                 "evidence_quote": "大补元气，复脉固脱"},
            ],
            entities=[
                {"name": "人参", "entity_type": "HERB"},
                {"name": "呕吐", "entity_type": "INDICATION"},
                {"name": "呕吐", "entity_type": "EFFICACY"},
            ],
        ))
        outcome = extractor.extract(chunk)
        record = outcome.assertions[0]
        assert record.object_id == canonical_id("INDICATION", "呕吐")
        assert record.object_id != canonical_id("EFFICACY", "呕吐")

    def test_ablation_flag_disables_validation(self, ontology, chunk):
        """KG0/KG1 消融：关掉 I2 后不得出现 ACCEPTED。"""
        settings = KGSettings(use_relation_validation=False)
        extractor, _ = _build(ontology, settings, _extraction([
            {"subject": "人参", "predicate": "HAS_EFFICACY", "object_text": "大补元气",
             "object_type": "EFFICACY", "polarity": "POSITIVE",
             "evidence_quote": "大补元气，复脉固脱"},
        ]))
        outcome = extractor.extract(chunk)
        assert outcome.assertions[0].validation_status == "REVIEW"

    def test_counts_expose_failure_ledger(self, ontology, chunk):
        settings = KGSettings()
        extractor, _ = _build(ontology, settings, _extraction([
            {"subject": "人参", "predicate": "HAS_EFFICACY", "object_text": "大补元气",
             "object_type": "EFFICACY", "polarity": "POSITIVE",
             "evidence_quote": "大补元气，复脉固脱"},
            {"subject": "人参", "predicate": "习称", "object_text": "神草",
             "object_type": "HERB", "polarity": "POSITIVE", "evidence_quote": "大补元气"},
        ]))
        counts = extractor.extract(chunk).counts()
        assert counts["assertions"] == 1
        assert counts["unknown_predicates"] == 1
        assert counts["rejected"] == 1


class TestEntryContinuation:
    """I3 跨块补全：把省略主语的句子归属到条目锚点药。"""

    CONT_TEXT = (
        "本品味辛性凉，有发表散邪、解肌退热之功。煎服，9～15g。\n"
        "《伤寒论》载其配伍应用。脾胃虚寒者慎用。"
    )

    def _cont_chunk(self):
        """测试用：条目续块。"""
        return ChunkRecord(
            chunk_id="c-ge-2", document_id="doc-zyx", entry_id="entry-gegen",
            section_path=["解表药"], order_in_entry=1,
            text=self.CONT_TEXT, normalized_text=normalize_text(self.CONT_TEXT),
            page_start=105, page_end=105, content_hash="hash-2",
        )

    def _payload(self, relations):
        """测试用：LLM JSON 载荷。"""
        return json.dumps({"entities": [], "relations": relations}, ensure_ascii=False)

    def test_implicit_subject_attributed_to_anchor(self, ontology):
        """「本品味辛性凉」在逐块抽取里没有主语，补全后应归属葛根。"""
        settings = KGSettings()
        llm = FakeLLMClient(
            responses={"complete_entry": [self._payload([
                {"subject": "葛根", "predicate": "HAS_EFFICACY", "object_text": "解肌退热",
                 "object_type": "EFFICACY", "polarity": "POSITIVE",
                 "evidence_quote": "本品味辛性凉，有发表散邪、解肌退热之功"},
            ])]},
            settings=settings,
        )
        extractor = ChunkExtractor(
            ontology=ontology, settings=settings,
            resolver=EntityResolver(AliasStore(), settings), llm=llm, run_id="run-i3",
        )
        outcome = extractor.extract_entry_continuation(
            self._cont_chunk(), anchor_name="葛根",
            group_chunk_ids=["c-ge-1", "c-ge-2"], page_range=(104, 105),
        )
        record = outcome.assertions[0]
        assert record.subject_id == canonical_id("HERB", "葛根")
        assert record.is_cross_chunk is True
        assert record.chunk_ids == ["c-ge-1", "c-ge-2"], "断言应记录整个条目的来源块"
        assert (record.page_start, record.page_end) == (104, 105)
        assert record.validation_status == "ACCEPTED"

    def test_non_anchor_subject_is_rejected(self, ontology):
        """补全只允许锚点药作主语，防止把邻近药材的属性错接过来。"""
        settings = KGSettings()
        llm = FakeLLMClient(
            responses={"complete_entry": [self._payload([
                {"subject": "麻黄", "predicate": "HAS_EFFICACY", "object_text": "发汗解表",
                 "object_type": "EFFICACY", "polarity": "POSITIVE",
                 "evidence_quote": "本品味辛性凉，有发表散邪、解肌退热之功"},
            ])]},
            settings=settings,
        )
        extractor = ChunkExtractor(
            ontology=ontology, settings=settings,
            resolver=EntityResolver(AliasStore(), settings), llm=llm, run_id="run-i3",
        )
        outcome = extractor.extract_entry_continuation(
            self._cont_chunk(), anchor_name="葛根",
            group_chunk_ids=["c-ge-1", "c-ge-2"], page_range=(104, 105),
        )
        assert outcome.assertions == []
        assert outcome.rejected[0]["reason"] == "subject_not_anchor"

    def test_completion_still_goes_through_validation(self, ontology):
        """补全候选不因来自 I3 而放宽 I2：编造证据同样只能 REVIEW。"""
        settings = KGSettings()
        llm = FakeLLMClient(
            responses={"complete_entry": [self._payload([
                {"subject": "葛根", "predicate": "HAS_EFFICACY", "object_text": "解肌退热",
                 "object_type": "EFFICACY", "polarity": "POSITIVE",
                 "evidence_quote": "葛根可治百病，为万能之药"},
            ])]},
            settings=settings,
        )
        extractor = ChunkExtractor(
            ontology=ontology, settings=settings,
            resolver=EntityResolver(AliasStore(), settings), llm=llm, run_id="run-i3",
        )
        outcome = extractor.extract_entry_continuation(
            self._cont_chunk(), anchor_name="葛根",
            group_chunk_ids=["c-ge-1", "c-ge-2"], page_range=(104, 105),
        )
        assert outcome.assertions[0].validation_status == "REVIEW"
        assert outcome.assertions[0].match_mode == "not_found"

    def test_dosage_in_continuation_parsed_as_value_object(self, ontology):
        settings = KGSettings()
        llm = FakeLLMClient(
            responses={
                "complete_entry": [self._payload([
                    {"subject": "葛根", "predicate": "HAS_DOSAGE", "object_text": "煎服，9～15g",
                     "object_type": "DOSAGE", "polarity": "POSITIVE",
                     "evidence_quote": "煎服，9～15g"},
                ])],
                "verify": ['{"verdict": "SUPPORTED"}'],
            },
            settings=settings,
        )
        extractor = ChunkExtractor(
            ontology=ontology, settings=settings,
            resolver=EntityResolver(AliasStore(), settings), llm=llm, run_id="run-i3",
        )
        outcome = extractor.extract_entry_continuation(
            self._cont_chunk(), anchor_name="葛根",
            group_chunk_ids=["c-ge-1", "c-ge-2"], page_range=(104, 105),
        )
        record = outcome.assertions[0]
        assert record.object_value["value_min"] == 9.0
        assert record.object_value["value_max"] == 15.0
        assert record.is_cross_chunk is True
