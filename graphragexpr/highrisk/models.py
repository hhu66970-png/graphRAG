"""高风险回答的数据模型。

逐子问题保存 grounded、partial_grounded、source_only 或 abstain 状态，
顶层 abstained 不隐藏已支持的结论。证据卡按完整条件单元组织，
rejections 保存未通过核验的候选结论及原因。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from ..domain.high_risk import (
    ABSTAINED_BY_STATUS,
    CATEGORY_DISPLAY,
    SEVERITY_DISPLAY,
    AnswerStatus,
    ConditionalEvidenceUnit,
    PolarityVerdict,
    RestrictionSeverity,
    RiskCategory,
    SourceKind,
)
from .claims import ClaimKind


class _Base(BaseModel):
    """与 `domain/models.py` 相同的冻结策略：未知字段一律拒绝，防止契约漂移。"""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


def _page_label(unit: ConditionalEvidenceUnit) -> str:
    """页码展示。`page_available=False` 时不编造页码，明确写「页码不可用」。"""
    if not unit.page_available or unit.page_start is None:
        return "页码不可用"
    if unit.page_end is None or unit.page_end == unit.page_start:
        return f"第 {unit.page_start} 页"
    return f"第 {unit.page_start}-{unit.page_end} 页"


class SubQuestionOutcome(_Base):
    """一个子问题的状态。整体状态由全部子问题归并，而不是另起一套判断。"""

    question: str
    risk_category: RiskCategory | None = None
    status: AnswerStatus
    reason: str
    unit_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)


class EvidenceCard(_Base):
    """条件证据单元的前端展示形态：条件与数值整卡不拆。"""

    unit_id: str
    risk_category: RiskCategory
    category_label: str
    predicate: str
    subject_name: str
    #: 定位字段：接口层要用它们从**最终保留或展示的单元**重建 sources 与
    #: subgraph，使顶层溯源与 claims/evidence_units 同源。
    subject_id: str | None = None
    object_id: str | None = None
    object_name: str | None = None
    conditions: str = ""
    severity: RestrictionSeverity
    severity_label: str
    polarity_verdict: PolarityVerdict
    polarity_basis: str = ""
    polarity_ambiguous: bool = False
    dosage_raw: str | None = None
    page_label: str
    page_start: int | None = None
    page_end: int | None = None
    evidence_quote: str
    context: str = ""
    source_kind: SourceKind
    assertion_id: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    validation_status: str | None = None
    is_clearly_parsed: bool
    conflicts: list[str] = Field(default_factory=list)
    undetermined_reason: str | None = None
    is_citable: bool
    #: 面向读者的「未解决说明」整句，由 `render.unresolved_flags` 在**单元**上
    #: 算好（分流要看 `dosage.parse_notes` 与结构标记，卡片字段不足以判断）。
    #: 前端直接显示这一列，不再对 `conflicts` 各自套模板：套错会把解析层
    #: 说明说成「证据存在差异」，从而与「已核验」自相矛盾。
    flags: list[str] = Field(default_factory=list)

    @classmethod
    def from_unit(cls, unit: ConditionalEvidenceUnit) -> "EvidenceCard":
        """由证据单元构造一条候选结论。"""
        from .render import unresolved_flags  # 局部导入：render 依赖本模块

        return cls(
            unit_id=unit.unit_id,
            risk_category=unit.risk_category,
            category_label=CATEGORY_DISPLAY[unit.risk_category],
            predicate=unit.predicate,
            subject_name=unit.subject_name,
            subject_id=unit.subject_id,
            object_id=unit.object_id,
            object_name=unit.object_name,
            conditions=unit.condition_summary(),
            severity=unit.severity,
            severity_label=SEVERITY_DISPLAY[unit.severity],
            polarity_verdict=unit.polarity_verdict,
            polarity_basis=unit.polarity_basis,
            polarity_ambiguous=unit.polarity_ambiguous,
            dosage_raw=unit.dosage.raw if unit.dosage else None,
            page_label=_page_label(unit),
            page_start=unit.page_start,
            page_end=unit.page_end,
            evidence_quote=unit.evidence_quote,
            context=unit.context,
            source_kind=unit.source_kind,
            assertion_id=unit.assertion_id,
            chunk_ids=list(unit.chunk_ids),
            validation_status=unit.validation_status,
            is_clearly_parsed=unit.is_clearly_parsed,
            conflicts=list(unit.conflicts),
            undetermined_reason=unit.undetermined_reason,
            is_citable=unit.is_citable(),
            flags=unresolved_flags(unit),
        )


class VerifiedClaim(_Base):
    """一条通过核验的结论。

    `statement` 由程序模板生成（见 `render.py`），**不是**模型原话：
    关键数值、对象与限制程度一律来自证据单元，避免最终文字重新引入偏差。
    """

    claim_id: str
    kind: ClaimKind
    statement: str
    subject_name: str
    object_name: str | None = None
    conditions: str = ""
    unit_ids: list[str] = Field(default_factory=list)
    pages: list[str] = Field(default_factory=list)


class AnswerBlockKind(str, Enum):
    """答案分块的种类。前端按种类套样式，**不靠字符串查找**加样式。

    块顺序即展示顺序：结论 → 条件解释 → 原文依据 →
    冲突或不足说明。程序核验说明单独走 `verification_note` 字段，
    因为它必须渲染在**全部块之后**且样式独立。
    """

    DECISION = "decision"      # 直接回应用户的第一句
    CONCLUSION = "conclusion"  # 结论与必要的条件解释（结构行）
    QUOTE = "quote"            # 与问题条件匹配的原文依据
    NOTE = "note"              # 冲突或证据不足的说明
    TEXT = "text"              # 未结构化的整段文本（source_only / abstain 沿用）


class AnswerBlock(_Base):
    """答案的一个展示块。`text` 已由后端定稿，前端只做转义、加粗标记处理与实体高亮。"""

    kind: AnswerBlockKind
    text: str
    unit_ids: list[str] = Field(default_factory=list)
    page_label: str = ""


class HighRiskAnswer(_Base):
    """高风险问答的完整结果。"""

    status: AnswerStatus
    abstained: bool
    answer_text: str
    sub_questions: list[SubQuestionOutcome] = Field(default_factory=list)
    claims: list[VerifiedClaim] = Field(default_factory=list)
    evidence_units: list[EvidenceCard] = Field(default_factory=list)
    rejections: list[dict] = Field(default_factory=list)
    coverage_notes: list[str] = Field(default_factory=list)
    is_high_risk: bool = True
    #: 直接结论。`decision` 是机器可读的类别，`decision_text` 是用户可见首句；
    #: 两者都由程序按模板与 `RestrictionSeverity` 生成，**不看模型文本、不做
    #: 字符串猜测**。`not_applicable` 表示该问法没有「能否使用」这一层结论。
    decision: str = "not_applicable"
    decision_text: str = ""
    #: 灰色的程序核验说明。空字符串表示**不显示**（source_only / abstain /
    #: 核验不可用），前端据此决定是否渲染。
    verification_note: str = ""
    #: 分块后的展示内容。为空表示前端按 `answer_text` 整体渲染（既有路径）。
    answer_blocks: list[AnswerBlock] = Field(default_factory=list)
    #: 与问题指定条件**不同**的证据。单独存放，不进 `claims`、不进
    #: `evidence_units`、不进 sources/subgraph，由前端放在默认折叠区。
    other_conditions: list[EvidenceCard] = Field(default_factory=list)

    @classmethod
    def build(
        cls,
        *,
        status: AnswerStatus,
        answer_text: str,
        sub_questions: list[SubQuestionOutcome],
        claims: list[VerifiedClaim],
        evidence_units: list[EvidenceCard],
        rejections: list[dict] | None = None,
        coverage_notes: list[str] | None = None,
        decision: str = "not_applicable",
        decision_text: str = "",
        verification_note: str = "",
        answer_blocks: list[AnswerBlock] | None = None,
        other_conditions: list[EvidenceCard] | None = None,
    ) -> "HighRiskAnswer":
        """按状态推导顶层 `abstained`，避免调用方各写一套映射。

        只有 `grounded` 表示用户所问的关键高风险结论全部给出；
        其余状态顶层为 true，但**不影响**已支持结论与原文的照常输出。
        """
        return cls(
            status=status,
            abstained=ABSTAINED_BY_STATUS[status],
            answer_text=answer_text,
            sub_questions=sub_questions,
            claims=claims,
            evidence_units=evidence_units,
            rejections=list(rejections or []),
            coverage_notes=list(coverage_notes or []),
            decision=decision,
            decision_text=decision_text,
            verification_note=verification_note,
            answer_blocks=list(answer_blocks or []),
            other_conditions=list(other_conditions or []),
        )
