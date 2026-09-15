"""Frozen public data contracts for the KG construction pipeline (SKILL §6).

字段名一旦冻结不得自行改名；任何公共字段变更都属于契约变更，
需 ADR（SKILL §15）。建图侧契约与检索层 EvidencePacket 集中在这里，
避免 API、评测和结构化层各自
复制字段定义。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ValidationStatus = Literal["ACCEPTED", "REVIEW", "REJECTED"]
LLMVerdict = Literal["SUPPORTED", "UNSUPPORTED", "NOT_RUN"]
Polarity = Literal["POSITIVE", "NEGATIVE"]
MatchMode = Literal["raw_span", "raw_exact", "normalized_exact", "fuzzy", "not_found"]
ReviewStatus = Literal["AUTO_CONFIRMED", "NEEDS_REVIEW", "MANUAL_CONFIRMED"]
LinkMethod = Literal["exact_alias", "longest_match", "character_candidate", "vector_candidate", "new_entity"]


class _Base(BaseModel):
    """_Base：本模块内部使用的数据结构。"""
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ChunkRecord(_Base):
    """检索/抽取单元。SKILL §6.1 冻结字段。"""

    chunk_id: str
    document_id: str
    parent_chunk_id: str | None = None
    entry_id: str | None = None          # 药物条目 ID，跨块补全的主依据（设计 §2.1.2-D）
    section_path: list[str] = Field(default_factory=list)
    order_in_entry: int | None = None
    text: str                            # 原文，证据定位以此为准
    normalized_text: str                 # 规范化文本，仅用于匹配，不覆盖原文
    page_start: int
    page_end: int
    content_hash: str
    ocr_quality: float | None = None

    @field_validator("page_end")
    @classmethod
    def _pages_ordered(cls, v: int, info) -> int:
        """保证页码起止有序。"""
        start = info.data.get("page_start")
        if start is not None and v < start:
            raise ValueError(f"page_end({v}) 不能小于 page_start({start})")
        return v


class EntityRecord(_Base):
    """规范实体。身份是 canonical_id，不是 name（SKILL §3.3 / §6.2）。"""

    canonical_id: str
    entity_type: str
    preferred_name: str
    normalized_name: str
    aliases: list[str] = Field(default_factory=list)
    source_authority: str | None = None
    link_confidence: float = 1.0
    link_method: LinkMethod = "new_entity"
    review_status: ReviewStatus = "AUTO_CONFIRMED"


class DosageValue(_Base):
    """剂量值对象（设计 §2.1.2-C）：必须保存标准化数值与条件，不能只存字符串。"""

    value_min: float | None = None
    value_max: float | None = None
    unit: str | None = None
    route: str | None = None
    condition: str | None = None
    raw: str


class EvidenceSpan(_Base):
    """证据定位结果（设计 §2.1.2-B）。"""

    evidence_quote: str                  # 逐字原句，来自 chunk.text
    evidence_start: int | None = None
    evidence_end: int | None = None
    match_mode: MatchMode
    edit_distance: int = 0

    @property
    def is_locatable(self) -> bool:
        """判断证据是否已定位到原文跨度。"""
        return self.match_mode != "not_found"


class ValidationResult(_Base):
    """把「规则分」与「最终决策值」彻底拆开（设计 §2.1.2-A）。

    rule_score  : 确定性规则支持度，永远有值。
    llm_verdict : 未进入二次校验时为 NOT_RUN，不参与线性公式。
    final_score : 规则与复核后的决策支持值；**不是事实正确概率**（SKILL §7.3）。
    """

    rule_score: float
    rule_flags: list[str] = Field(default_factory=list)
    review_required: bool = False
    llm_verdict: LLMVerdict = "NOT_RUN"
    final_score: float = 0.0
    status: ValidationStatus = "REVIEW"


class AssertionRecord(_Base):
    """事实源层的一条断言。SKILL §6.3 冻结字段。

    每条 ACCEPTED 断言必须有 chunk_ids、页码和可定位 evidence_quote（SKILL §3.1）。
    """

    assertion_id: str
    fact_key: str
    subject_id: str
    predicate: str
    object_id: str | None = None         # 实体宾语
    object_value: dict | None = None     # 值对象宾语（如 DOSAGE）
    polarity: Polarity = "POSITIVE"
    qualifiers: dict = Field(default_factory=dict)
    evidence_quote: str
    evidence_start: int | None = None
    evidence_end: int | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    page_start: int
    page_end: int
    rule_score: float = 0.0
    final_score: float = 0.0
    validation_status: ValidationStatus = "REVIEW"
    prompt_version: str = ""
    schema_version: str = ""
    run_id: str = ""
    match_mode: MatchMode = "not_found"
    is_cross_chunk: bool = False

    @field_validator("object_value")
    @classmethod
    def _object_present(cls, v: dict | None, info) -> dict | None:
        """检查断言宾语（实体或值对象）是否齐全。"""
        if v is None and not info.data.get("object_id"):
            raise ValueError("object_id 与 object_value 必须至少提供一个")
        return v

    def assert_publishable(self) -> None:
        """ACCEPTED 断言的硬门槛（SKILL §3.1）。"""
        if self.validation_status != "ACCEPTED":
            return
        if not self.evidence_quote or self.match_mode == "not_found":
            raise ValueError(f"ACCEPTED 断言缺少可定位证据: {self.assertion_id}")
        if not self.chunk_ids:
            raise ValueError(f"ACCEPTED 断言缺少来源 chunk: {self.assertion_id}")


class EvidenceItem(_Base):
    """One evidence-backed retrieval item (SKILL section 6.4)."""

    evidence_id: str
    claim_key: str | None = None
    subject_id: str | None = None
    predicate: str | None = None
    object_id: str | None = None
    path: list[dict] | None = None
    evidence_quote: str
    chunk_id: str
    page_start: int
    page_end: int
    support_score: float = 0.0
    retrieval_score: float = 0.0
    qualifiers: dict = Field(default_factory=dict)

    @field_validator("page_end")
    @classmethod
    def _evidence_pages_ordered(cls, v: int, info) -> int:
        """校验证据页码起止合法。"""
        start = info.data.get("page_start")
        if start is not None and v < start:
            raise ValueError(f"page_end({v}) 不能小于 page_start({start})")
        return v


class EvidencePacket(_Base):
    """Frozen hand-off from retrieval to structuring/generation."""

    query_id: str
    intent: Literal["attr", "reverse", "intersect", "compare", "path", "fallback"]
    linked_entities: list[dict] = Field(default_factory=list)
    structure_type: Literal["card", "table", "matrix", "paths", "summary", "chunks"]
    items: list[EvidenceItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    token_budget: int
    trace: dict = Field(default_factory=dict)
