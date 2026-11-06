"""Strict-schema extraction producing evidence-bearing assertions.

把 2.1.1–2.1.3 的各个环节串成一条可复现管线：

    严格 Schema 抽取（LLM 只能输出本体内的稳定 ID，且必须自带逐字证据）
      -> I1 实体链接与消歧（类型内解析，跨类型绝不合并）
      -> 证据定位（四级匹配，拿到原文字符跨度）
      -> I2 规则验证 + 灰区二次校验（rule_score / final_score 语义分离）
      -> AssertionRecord（带 chunk_ids、页码、evidence_quote、版本）

本模块**不决定发布哪个 Run**，只输出结构化候选与验证结果。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ..config.settings import KGSettings
from ..domain.errors import OntologyError
from ..domain.ids import assertion_id, fact_key
from ..domain.models import AssertionRecord, ChunkRecord
from ..domain.ontology import Ontology
from ..linking.entity_resolver import EntityResolver
from .dosage import parse_dosage
from .evidence import locate_evidence
from .llm_client import LLMClient
from .relation_validator import validate_relation

logger = logging.getLogger(__name__)

_SYSTEM = "你是中药学知识图谱抽取专家。只输出 JSON，严禁编造教材片段之外的内容。"

_USER_TEMPLATE = """请从下面的《中药学》教材片段中抽取实体与关系。

{schema_block}

硬性要求：
1. entity_type 与 predicate 只能使用上面列出的**英文 ID**，不得自创。
2. 每条关系必须给出 evidence_quote：**逐字复制**片段中支持该关系的最短原句。
3. 无法在片段中找到逐字依据的关系，不要输出。
4. polarity：句中出现「不宜/忌/禁/勿」等否定时填 NEGATIVE，否则 POSITIVE。
5. 剂量类关系（HAS_DOSAGE）的 object_text 填完整剂量表述，如「煎服，3～9g」。
6. 本片段最多输出 40 条关系；超出时只保留证据最明确的，宁缺毋滥。

来源：PDF 第 {page_start}-{page_end} 页；章节：{section}

片段原文：
\"\"\"
{text}
\"\"\"

只返回如下 JSON：
{{
  "entities": [{{"name": "实体名", "entity_type": "英文类型ID"}}],
  "relations": [{{
    "subject": "主语实体名", "predicate": "英文关系ID", "object_text": "宾语实体名或剂量表述",
    "object_type": "英文类型ID", "polarity": "POSITIVE|NEGATIVE",
    "evidence_quote": "逐字原句"
  }}]
}}"""


_ENTRY_USER_TEMPLATE = """本片段出自《中药学》中【{anchor}】的条目（PDF 第 {page_start}-{page_end} 页）。
片段里的「本品」「其」「该药」等指代**一律指 {anchor}**。

{schema_block}

硬性要求：
1. 只抽取**主语是 {anchor}** 的关系；主语不是它的一律不要输出。
2. predicate 与 object_type 只能使用上面的英文 ID。
3. 每条关系必须给出 evidence_quote：**逐字复制**本片段中的最短原句。
4. 片段中找不到逐字依据的，不要输出。
5. 出现「不宜/忌/禁/勿」等否定时 polarity 填 NEGATIVE。
6. 最多输出 30 条关系。

片段原文：
\"\"\"
{text}
\"\"\"

只返回如下 JSON：
{{
  "entities": [],
  "relations": [{{
    "subject": "{anchor}", "predicate": "英文关系ID", "object_text": "宾语实体名或剂量表述",
    "object_type": "英文类型ID", "polarity": "POSITIVE|NEGATIVE",
    "evidence_quote": "逐字原句"
  }}]
}}"""


class _RawEntity(BaseModel):
    name: str
    entity_type: str


class _RawRelation(BaseModel):
    subject: str
    predicate: str
    object_text: str
    object_type: str
    polarity: str = "POSITIVE"
    evidence_quote: str = ""


class RawExtraction(BaseModel):
    """LLM 输出契约。"""

    entities: list[_RawEntity] = Field(default_factory=list)
    relations: list[_RawRelation] = Field(default_factory=list)


@dataclass
class ExtractionOutcome:
    """One chunk's extraction result plus a rejection ledger."""

    assertions: list[AssertionRecord] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    unknown_types: list[str] = field(default_factory=list)
    unknown_predicates: list[str] = field(default_factory=list)

    def counts(self) -> dict:
        """汇总本块抽取的接受/拒绝/未知类型计数。"""
        by_status: dict[str, int] = {}
        for a in self.assertions:
            by_status[a.validation_status] = by_status.get(a.validation_status, 0) + 1
        return {
            "assertions": len(self.assertions),
            "by_status": by_status,
            "rejected": len(self.rejected),
            "unknown_types": len(self.unknown_types),
            "unknown_predicates": len(self.unknown_predicates),
        }


class ChunkExtractor:
    """Extract validated assertions from a single chunk."""

    def __init__(
        self,
        *,
        ontology: Ontology,
        settings: KGSettings,
        resolver: EntityResolver,
        llm: LLMClient,
        run_id: str,
    ) -> None:
        self._ontology = ontology
        self._settings = settings
        self._resolver = resolver
        self._llm = llm
        self._run_id = run_id

    def extract(self, chunk: ChunkRecord) -> ExtractionOutcome:
        """Run extraction + linking + validation for one chunk.

        Args:
            chunk: 带原文与页码的抽取单元。

        Returns:
            ExtractionOutcome；被拒候选进入 `rejected`，未知类型/谓词分别登记，
            供 health 接口与评测统计。

        Side effects: 调用一次抽取 LLM；灰区关系可能各触发一次校验 LLM。
        """
        outcome = ExtractionOutcome()

        prompt = _USER_TEMPLATE.format(
            schema_block=self._ontology.prompt_schema_block(),
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            section=" / ".join(chunk.section_path) or "未识别",
            text=chunk.text,
        )
        raw = self._llm.complete_json(
            system=_SYSTEM, user=prompt, schema=RawExtraction, purpose="extract"
        )
        assert isinstance(raw, RawExtraction)

        # ---- 实体：类型必须在本体内，未知类型记录并丢弃，不降级为「概念」----
        resolved: dict[str, tuple[str, str]] = {}  # surface -> (canonical_id, entity_type)
        for entity in raw.entities:
            try:
                entity_type = self._ontology.resolve_entity_type(entity.entity_type)
            except OntologyError:
                outcome.unknown_types.append(entity.entity_type)
                continue
            if self._ontology.is_value_object(entity_type):
                continue  # 值对象不建实体身份
            if not self._settings.use_entity_resolution:
                continue
            decision = self._resolver.resolve(entity.name, entity_type, context=chunk.text)
            resolved[entity.name] = (decision.record.canonical_id, entity_type)

        # ---- 关系 -----------------------------------------------------------
        for relation in raw.relations:
            record = self._build_assertion(chunk, relation, resolved, outcome)
            if record is not None:
                outcome.assertions.append(record)

        return outcome

    def extract_entry_continuation(
        self,
        chunk: ChunkRecord,
        *,
        anchor_name: str,
        group_chunk_ids: list[str],
        page_range: tuple[int, int],
    ) -> ExtractionOutcome:
        """I3: re-attribute implicit-subject facts to the entry's anchor herb.

        教材条目里大量句子省略主语（「本品味辛性凉，有发表散邪之功」）。
        逐块抽取时这些句子要么被丢弃，要么主语被错认。本方法把条目锚点药名
        显式注入 Prompt，只抽取主语为该药的关系，并把整个条目的 chunk_ids
        一并记入断言，使事实可回溯到条目内的全部来源块。

        产出的候选**同样要过 I2 验证**，不因来自补全而放宽。

        Args:
            chunk: 条目内的某一块（通常是不含药名的续写块）。
            anchor_name: 该条目的药名。
            group_chunk_ids: 整个条目涉及的块 ID，写入 Assertion.chunk_ids。
            page_range: 条目的页码范围。

        Returns:
            ExtractionOutcome，其中断言的 is_cross_chunk 为 True。

        Side effects: 调用一次补全 LLM；灰区/高风险关系可能再触发校验 LLM。
        """
        outcome = ExtractionOutcome()
        prompt = _ENTRY_USER_TEMPLATE.format(
            anchor=anchor_name,
            schema_block=self._ontology.prompt_schema_block(),
            page_start=page_range[0],
            page_end=page_range[1],
            text=chunk.text,
        )
        raw = self._llm.complete_json(
            system=_SYSTEM, user=prompt, schema=RawExtraction, purpose="complete_entry"
        )
        assert isinstance(raw, RawExtraction)

        # 锚点药名先解析为规范实体，保证条目内所有补全事实指向同一个 canonical_id
        anchor = self._resolver.resolve(anchor_name, "HERB", context=chunk.text)
        resolved = {anchor_name: (anchor.record.canonical_id, "HERB")}

        for relation in raw.relations:
            if relation.subject != anchor_name:
                outcome.rejected.append(
                    {"reason": "subject_not_anchor", "raw": relation.model_dump()}
                )
                continue
            record = self._build_assertion(
                chunk, relation, resolved, outcome,
                chunk_ids=group_chunk_ids,
                page_range=page_range,
                is_cross_chunk=True,
            )
            if record is not None:
                outcome.assertions.append(record)
        return outcome

    def _build_assertion(
        self,
        chunk: ChunkRecord,
        relation: _RawRelation,
        resolved: dict[str, tuple[str, str]],
        outcome: ExtractionOutcome,
        *,
        chunk_ids: list[str] | None = None,
        page_range: tuple[int, int] | None = None,
        is_cross_chunk: bool = False,
    ) -> AssertionRecord | None:
        """把一条通过校验的关系组装成 AssertionRecord。"""
        try:
            predicate = self._ontology.resolve_predicate(relation.predicate)
        except OntologyError:
            outcome.unknown_predicates.append(relation.predicate)
            outcome.rejected.append({"reason": "unknown_predicate", "raw": relation.model_dump()})
            return None

        try:
            object_type = self._ontology.resolve_entity_type(relation.object_type)
        except OntologyError:
            outcome.unknown_types.append(relation.object_type)
            outcome.rejected.append({"reason": "unknown_object_type", "raw": relation.model_dump()})
            return None

        subject = resolved.get(relation.subject)
        if subject is None:
            # 主语未出现在实体列表里 -> 单独解析，但类型由谓词 domain 推断
            domain_types = self._ontology.predicate(predicate).domain
            candidate_type = next((t for t in domain_types if not self._ontology.is_value_object(t)), None)
            if candidate_type is None:
                outcome.rejected.append({"reason": "no_subject_type", "raw": relation.model_dump()})
                return None
            decision = self._resolver.resolve(relation.subject, candidate_type, context=chunk.text)
            subject = (decision.record.canonical_id, candidate_type)

        subject_id, subject_type = subject

        # 值对象宾语（剂量）与实体宾语分开处理
        dosage = None
        object_id: str | None = None
        object_value: dict | None = None
        object_ref: str
        if self._ontology.is_value_object(object_type):
            dosage = parse_dosage(relation.object_text)
            object_value = dosage.model_dump()
            object_ref = dosage.raw
        else:
            decision = self._resolver.resolve(relation.object_text, object_type, context=chunk.text)
            object_id = decision.record.canonical_id
            object_ref = object_id

        evidence = locate_evidence(
            relation.evidence_quote,
            chunk.text,
            max_edit_distance=self._settings.evidence_max_edit_distance,
            min_quote_len=self._settings.evidence_min_quote_len,
        )

        polarity = "NEGATIVE" if relation.polarity.upper() == "NEGATIVE" else "POSITIVE"

        if self._settings.use_relation_validation:
            validation = validate_relation(
                ontology=self._ontology,
                settings=self._settings,
                predicate=predicate,
                subject_type=subject_type,
                object_type=object_type,
                subject_name=relation.subject,
                object_name=relation.object_text,
                evidence=evidence,
                polarity=polarity,
                dosage=dosage,
                llm_verifier=lambda: self._verify(chunk, relation, evidence.evidence_quote),
            )
        else:  # KG0/KG1 消融：不做关系验证时一律进 REVIEW，绝不直接 ACCEPTED
            from ..domain.models import ValidationResult

            validation = ValidationResult(
                rule_score=0.0, rule_flags=["validation_disabled"],
                review_required=True, llm_verdict="NOT_RUN",
                final_score=0.0, status="REVIEW",
            )

        if validation.status == "REJECTED":
            outcome.rejected.append(
                {"reason": "validation_rejected", "flags": validation.rule_flags,
                 "raw": relation.model_dump()}
            )
            return None

        key = fact_key(subject_id, predicate, object_ref, {}, polarity)
        return AssertionRecord(
            assertion_id=assertion_id(
                key, chunk.chunk_id,
                (evidence.evidence_start, evidence.evidence_end),
                self._settings.schema_version,
            ),
            fact_key=key,
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            object_value=object_value,
            polarity=polarity,
            qualifiers={},
            evidence_quote=evidence.evidence_quote,
            evidence_start=evidence.evidence_start,
            evidence_end=evidence.evidence_end,
            chunk_ids=chunk_ids or [chunk.chunk_id],
            page_start=page_range[0] if page_range else chunk.page_start,
            page_end=page_range[1] if page_range else chunk.page_end,
            rule_score=validation.rule_score,
            final_score=validation.final_score,
            validation_status=validation.status,
            prompt_version=self._settings.prompt_version,
            schema_version=self._settings.schema_version,
            run_id=self._run_id,
            match_mode=evidence.match_mode,
            is_cross_chunk=is_cross_chunk,
        )

    def _verify(self, chunk: ChunkRecord, relation: _RawRelation, quote: str) -> str:
        """Second-pass verification. 只依据提供的证据，不允许调用常识。"""

        class _Verdict(BaseModel):
            verdict: str

        # 注意：OpenAI 兼容接口在 response_format=json_object 下要求消息中出现
        # "json" 字样，否则直接报错。这里显式写明输出 JSON。
        prompt = (
            "仅根据【证据】判断【断言】是否成立。不得使用证据之外的任何知识。\n"
            f"【断言】{relation.subject} —{relation.predicate}→ {relation.object_text}"
            f"（极性 {relation.polarity}）\n【证据】{quote}\n"
            '请只返回 JSON：{"verdict": "SUPPORTED"} 或 {"verdict": "UNSUPPORTED"}'
        )
        try:
            result = self._llm.complete_json(
                system="你是严谨的中药学事实校验器。", user=prompt,
                schema=_Verdict, purpose="verify",
            )
            verdict = getattr(result, "verdict", "").upper()
            return verdict if verdict in {"SUPPORTED", "UNSUPPORTED"} else "NOT_RUN"
        except Exception as exc:  # noqa: BLE001 - 降级而非中断整批抽取
            logger.warning("verifier_failed", extra={"chunk_id": chunk.chunk_id, "error": str(exc)})
            return "NOT_RUN"
