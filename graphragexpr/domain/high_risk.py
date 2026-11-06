"""高风险领域的类型化词汇表：风险类别、限制程度、条件证据单元。

支持三类关系（`HAS_DOSAGE` / `CONTRAINDICATED_IN` / `INCOMPATIBLE_WITH`），
本模块是这三类关系在「读取 → 条件组织 → 核验」全链路上共用的数据契约。
放在 `domain/` 是因为存储层、风险服务层与接口层必须对同一组字段达成一致。

条件单元的约束：

1. 同段原文出现多种用法或不同生制品对应不同剂量时，**生成多个分别绑定条件的单元**；
   严禁从整段不同位置各取一个数字、单位、用法再拼成一条事实。
2. 不自动补充「成人」「每日」「每次」等默认条件，不做个体剂量计算、未授权单位换算、
   **倒置范围修正**或药物形态推断。仅允许可逆的单位别名规范化（保留原文形式）。
3. 教材未说明的字段留空；不得把一般说明推成对特定人群或情境的适用结论。
4. 条件冲突的单元并列保留，不取平均、不求并集。
5. 缺页码与缺原文分开处理：`page_available=False` 与 `evidence_quote=""` 语义不同。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

# 当前支持的三类高风险谓词。不从本体全量 high_risk 推导，
# 避免未来新增高风险谓词时静默扩大本模块的处理范围。
TASK_HIGH_RISK_PREDICATES: tuple[str, ...] = (
    "HAS_DOSAGE",
    "CONTRAINDICATED_IN",
    "INCOMPATIBLE_WITH",
)


class RiskCategory(str, Enum):
    """高风险类别。与谓词一一对应，是单元分组与前端标签的依据。"""

    DOSAGE = "DOSAGE"
    CONTRAINDICATION = "CONTRAINDICATION"
    INCOMPATIBILITY = "INCOMPATIBILITY"


PREDICATE_CATEGORY: dict[str, RiskCategory] = {
    "HAS_DOSAGE": RiskCategory.DOSAGE,
    "CONTRAINDICATED_IN": RiskCategory.CONTRAINDICATION,
    "INCOMPATIBLE_WITH": RiskCategory.INCOMPATIBILITY,
}

CATEGORY_DISPLAY: dict[RiskCategory, str] = {
    RiskCategory.DOSAGE: "用量",
    RiskCategory.CONTRAINDICATION: "使用禁忌",
    RiskCategory.INCOMPATIBILITY: "配伍禁忌",
}


class RestrictionSeverity(str, Enum):
    """禁忌的限制程度。

    要求保留「禁用 / 忌用 / 慎用」的区别，
    **不能用同一个布尔值抹平**——三者在临床含义上并不等价。
    """

    PROHIBITED = "PROHIBITED"    # 禁用
    AVOID = "AVOID"              # 忌用 / 不宜
    CAUTION = "CAUTION"          # 慎用
    UNSPECIFIED = "UNSPECIFIED"  # 原文未给出程度词


SEVERITY_DISPLAY: dict[RestrictionSeverity, str] = {
    RestrictionSeverity.PROHIBITED: "禁用",
    RestrictionSeverity.AVOID: "忌用",
    RestrictionSeverity.CAUTION: "慎用",
    RestrictionSeverity.UNSPECIFIED: "原文未标注程度",
}


class SourceKind(str, Enum):
    """证据来源类型。原文来源**不得**被升级为通过断言。"""

    ASSERTION = "ASSERTION"      # 来自校验通过的断言
    TEXT_CHUNK = "TEXT_CHUNK"    # 仅来自教材原文，独立核验后使用


class PolarityVerdict(str, Enum):
    """旧极性的服务侧解释结果。

    「甲药不能与乙药同用」：对「同用」是否定，对「配伍禁忌」关系是**肯定**。
    因此原始极性与关系是否成立必须分开记录，不能混为一谈。
    """

    RELATION_HOLDS = "RELATION_HOLDS"
    RELATION_DOES_NOT_HOLD = "RELATION_DOES_NOT_HOLD"
    UNDETERMINED = "UNDETERMINED"


class AnswerStatus(str, Enum):
    """高风险回答的状态分级。"""

    GROUNDED = "grounded"
    PARTIAL_GROUNDED = "partial_grounded"
    SOURCE_ONLY = "source_only"
    ABSTAIN = "abstain"


#: 顶层 abstained 字段的映射：只有 grounded 表示「用户所问的关键高风险结论全部给出」。
#: 逐子问题状态另行保存在 sub_questions 中，前端不得因顶层 abstained=true
#: 而隐藏已支持的结论或可信原文。
ABSTAINED_BY_STATUS: dict[AnswerStatus, bool] = {
    AnswerStatus.GROUNDED: False,
    AnswerStatus.PARTIAL_GROUNDED: True,
    AnswerStatus.SOURCE_ONLY: True,
    AnswerStatus.ABSTAIN: True,
}


class _Base(BaseModel):
    """与 `domain/models.py` 相同的冻结策略：未知字段一律拒绝，防止契约漂移。"""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class FieldSpan(_Base):
    """关键字段在原文中的字符跨度。

    要求「每组关键字段支持的原文位置」可核对，因此每个关键槽位
    单独记录跨度，而不是只记录整句位置。
    """

    start: int
    end: int
    text: str

    @model_validator(mode="after")
    def _ordered(self) -> "FieldSpan":
        """按固定顺序去重排列，保证词表输出稳定。"""
        if self.end < self.start:
            raise ValueError(f"字段跨度结束({self.end}) 小于开始({self.start})")
        return self


class DosageBinding(_Base):
    """一组剂量绑定：数值 + 单位 + **该数值自身的**用法/形态/条件。

    与旧版 `DosageValue` 的区别：

    - `is_inverted` 只做标记，**不修正**。
      是否为 OCR 错误交由核验与人工判断，程序不得擅自交换上下界。
    - 单位只允许在同一条件片段内配对，不做整段搜索补单位。
    - `span` 记录该数值在原文中的位置，使「数字确实出现在原文」与
      「数字与本单位匹配」成为两个可分别检查的条件。
    """

    raw: str
    value_min: float | None = None
    value_max: float | None = None
    unit: str | None = None          # 规范化后的可逆别名（克→g），仅作展示
    unit_raw: str | None = None      # 原文形式，必须保留
    is_single_value: bool = False
    is_inverted: bool = False        # 原文上界小于下界：仅标记
    preparation: str | None = None   # 生制品／药物形态
    route: str | None = None         # 用法／给药途径
    frequency: str | None = None     # 频次／时间条件
    population: str | None = None    # 适用人群
    condition_text: str | None = None
    field_spans: dict[str, FieldSpan] = Field(default_factory=dict)
    parse_notes: list[str] = Field(default_factory=list)

    def has_numeric_value(self) -> bool:
        """判断文本里是否含有可核验的剂量数字。"""
        return self.value_min is not None and self.value_max is not None

    def is_bound_to_unit(self) -> bool:
        """数值与单位是否在同一条件片段内配对成功。"""
        return self.has_numeric_value() and bool(self.unit)


class ConditionalEvidenceUnit(_Base):
    """类型化的高风险条件证据单元。

    借鉴 FHIR `Dosage` 对「剂量 / 途径 / 时间安排 / 补充说明 / 自由文本并存」的
    组织思路，并加入中药特有的生制品与来源追溯。**只借鉴数据组织方式**，
    不安装 FHIR 框架、不新建 FHIR 服务器，也不声称因此获得临床安全性。
    参考：https://hl7.org/fhir/dosage-definitions.html

    一个单元 = 一个可以独立成立的事实 + 它成立所依赖的全部条件。
    核验时按单元整体判断「结论是否被这个单元支持」，因此条件和数值不会被拆散。
    """

    unit_id: str
    risk_category: RiskCategory
    predicate: str

    # ---- 药物对象与相关对象 -------------------------------------------
    subject_id: str | None = None
    subject_name: str = ""
    object_id: str | None = None
    object_name: str | None = None
    object_value: dict | None = None

    # ---- 适用条件（教材未说明的留空，不补默认值）------------------------
    preparation: str | None = None
    route: str | None = None
    dosage: DosageBinding | None = None
    frequency: str | None = None
    population: str | None = None
    other_restrictions: list[str] = Field(default_factory=list)

    # ---- 限制程度：禁用/忌用/慎用不能被抹平 -----------------------------
    severity: RestrictionSeverity = RestrictionSeverity.UNSPECIFIED
    severity_raw: str | None = None

    # ---- 来源与校验状态 -----------------------------------------------
    assertion_id: str | None = None
    fact_key: str | None = None
    source_kind: SourceKind = SourceKind.TEXT_CHUNK
    validation_status: str | None = None
    is_cross_chunk: bool = False

    # ---- 极性：原始值与被解释值分开保存----------------------
    raw_polarity: str | None = None
    polarity_verdict: PolarityVerdict = PolarityVerdict.UNDETERMINED
    polarity_basis: str = ""
    polarity_ambiguous: bool = False

    # ---- 原文与定位 -----------------------------------------------------
    evidence_quote: str = ""
    context: str = ""                # 完整相关上下文，供结构化困难时原文展示
    chunk_ids: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    page_available: bool = True      # 与「缺原文」分开处理，不得补造页码
    span: FieldSpan | None = None
    field_spans: dict[str, FieldSpan] = Field(default_factory=dict)

    # ---- 解析质量 -------------------------------------------------------
    is_clearly_parsed: bool = True
    conflicts: list[str] = Field(default_factory=list)
    undetermined_reason: str | None = None

    @model_validator(mode="after")
    def _consistency(self) -> "ConditionalEvidenceUnit":
        """检查高风险相关字段是否自洽。"""
        if self.page_start is not None and self.page_end is not None:
            if self.page_end < self.page_start:
                raise ValueError(
                    f"单元页码结束({self.page_end}) 小于开始({self.page_start})"
                )
        if self.source_kind is SourceKind.ASSERTION and not self.assertion_id:
            raise ValueError("ASSERTION 来源的单元必须有 assertion_id，禁止伪造事实标识")
        if self.source_kind is SourceKind.TEXT_CHUNK and self.assertion_id:
            raise ValueError(
                "TEXT_CHUNK 来源的单元不得携带 assertion_id："
                "原文证据不能被冒充为通过断言"
            )
        return self

    def is_citable(self) -> bool:
        """能否作为引用证据。原文不可定位的单元不能进入核验通过集。"""
        return bool(self.evidence_quote.strip())

    def condition_summary(self) -> str:
        """条件的人类可读摘要，用于核验提示与前端证据卡。"""
        parts: list[str] = []
        if self.preparation:
            parts.append(f"生制品：{self.preparation}")
        if self.route:
            parts.append(f"用法：{self.route}")
        if self.dosage and self.dosage.raw:
            parts.append(f"剂量：{self.dosage.raw}")
        if self.frequency:
            parts.append(f"频次：{self.frequency}")
        if self.population:
            parts.append(f"人群：{self.population}")
        if self.severity is not RestrictionSeverity.UNSPECIFIED:
            parts.append(f"程度：{SEVERITY_DISPLAY[self.severity]}")
        parts.extend(self.other_restrictions)
        return "；".join(parts)


def category_of(predicate: str) -> RiskCategory | None:
    """支持范围内的谓词映射；范围外谓词返回 None（不静默扩大处理范围）。"""
    return PREDICATE_CATEGORY.get(predicate)
