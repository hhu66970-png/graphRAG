"""候选结论的结构化契约（任务书 §6）。

模型**不得**只返回一个「安全／不安全」判断，也不得把自报置信度当成通过依据。
因此候选结论被拆成可程序核对的字段：

- 结论内容与类型；
- 涉及对象（药物、相关对象）；
- 采用的条件（生制品、用法、人群、限制程度）——核验时与所选证据单元逐项比对；
- 支持证据单元的引用标识——必须属于本次最终注入的证据集；
- 结构化的数值与单位——使「数字对但绑定错」可以被检查出来，
  而不是只看数字是否在整段里出现过（任务书 §6）。

`asserted_*` 字段是**模型对结论的自我结构化**，核验程序拿它与证据单元比对；
一旦不一致即构成硬性失败，不能靠模型二次确认或相似度分数推翻。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from ..domain.high_risk import PolarityVerdict, RestrictionSeverity


class ClaimKind(str, Enum):
    """候选结论的类型。与三类风险一一对应。"""

    DOSAGE = "dosage"
    CONTRAINDICATION = "contraindication"
    INCOMPATIBILITY = "incompatibility"
    NO_EVIDENCE = "no_evidence"      # 明确声明「未检索到相关证据」


class _Base(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class CandidateClaim(_Base):
    """一条待核验的候选结论。"""

    claim_id: str
    kind: ClaimKind
    statement: str

    # ---- 涉及对象 -------------------------------------------------------
    subject_name: str = ""
    object_name: str | None = None

    # ---- 采用的条件（与证据单元比对）-------------------------------------
    asserted_preparation: str | None = None
    asserted_route: str | None = None
    asserted_population: str | None = None
    asserted_frequency: str | None = None
    asserted_severity: RestrictionSeverity | None = None
    asserted_polarity: PolarityVerdict | None = None

    # ---- 结构化数值（使绑定错误可被程序发现）-----------------------------
    asserted_numbers: list[float] = Field(default_factory=list)
    asserted_unit: str | None = None

    # ---- 引用 -----------------------------------------------------------
    cited_unit_ids: list[str] = Field(default_factory=list)
    sub_question: str | None = None

    def is_empty_conclusion(self) -> bool:
        """是否只是「没有证据」的声明，而非一条事实性结论。"""
        return self.kind is ClaimKind.NO_EVIDENCE


class ClaimBatch(_Base):
    """模型一次返回的全部候选结论。"""

    claims: list[CandidateClaim] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


CLAIM_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [kind.value for kind in ClaimKind],
                    },
                    "statement": {"type": "string"},
                    "subject_name": {"type": "string"},
                    "object_name": {"type": ["string", "null"]},
                    "asserted_preparation": {"type": ["string", "null"]},
                    "asserted_route": {"type": ["string", "null"]},
                    "asserted_population": {"type": ["string", "null"]},
                    "asserted_frequency": {"type": ["string", "null"]},
                    "asserted_severity": {
                        "type": ["string", "null"],
                        "enum": [item.value for item in RestrictionSeverity] + [None],
                    },
                    "asserted_polarity": {
                        "type": ["string", "null"],
                        "enum": [item.value for item in PolarityVerdict] + [None],
                    },
                    "asserted_numbers": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                    "asserted_unit": {"type": ["string", "null"]},
                    "cited_unit_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "sub_question": {"type": ["string", "null"]},
                },
                "required": ["claim_id", "kind", "statement", "cited_unit_ids"],
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["claims"],
}
