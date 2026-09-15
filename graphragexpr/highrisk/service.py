"""高风险问答编排：读取 → 单元 → 候选 → 核验 → 状态（任务书 §3、§6、§10）。

为什么独立成一个服务
--------------------
任务书 §3 要求把三类高风险关系「集中为可测试的独立服务／策略模块」，
而不是分散在各检索模式的分支里各写一遍。这样做的直接好处是：

- 只有一处决定「证据不足怎么办」，不会出现某个模式悄悄绕过核验；
- 可以用 `FakeLLMClient` 做确定性测试，不依赖真实模型与真实数据库；
- 接口层只负责把结果装进既有响应契约，不承担安全判断。

硬性约束（本模块的实现方式）
----------------------------
1. **没有最终核验就不出结论**。任何路径都必须经过 `verify.verify_claims`。
2. **模型不写最终文字**。最终文本由 `render` 从已核验单元模板化生成，
   模型只提交待核验的候选结论。
3. **条件与数值一起裁剪**。token 预算按**整单元**裁剪，绝不保留数值却丢弃其条件
   （任务书 §5）；被裁掉的单元同时从可引用集合移除，防止引用悬空。
4. **降级要出声**。弃权、仅来源、部分成立都在 `coverage_notes` 与
   `sub_questions[].reason` 里写明原因，不静默变成一句「未检索到」。

只读：不写任何数据，不建图，不发布批次。
"""

from __future__ import annotations

import logging
import re

from ..domain.high_risk import (
    PREDICATE_CATEGORY,
    AnswerStatus,
    ConditionalEvidenceUnit,
    RestrictionSeverity,
    RiskCategory,
)
from ..storage.high_risk_read import (
    DEFAULT_ASSERTION_LIMIT,
    load_high_risk_assertions,
    load_high_risk_assertions_by_object,
)
from .applicability import (
    assess_applicability,
    is_enumeration_question,
    is_generic_object,
    longer_object_span,
    severity_conflicts,
)
from .claims import CandidateClaim, ClaimBatch, ClaimKind
from .models import (
    AnswerBlock,
    AnswerBlockKind,
    EvidenceCard,
    HighRiskAnswer,
    SubQuestionOutcome,
    VerifiedClaim,
)
from .parsing import extract_population
from .relevance import (
    Decision,
    DecisionResult,
    Relevance,
    classify,
    decide,
    ordered,
    sort_units as sort_by_relevance,
)
from .render import (
    NO_EVIDENCE_NOTE,
    render_abstain,
    render_conclusion_line,
    render_quote_line,
    render_source_only,
    render_unit_statement,
    verification_note_for,
)
from .units import build_units_from_assertions
from .verify import VerificationOutcome, verify_claims

logger = logging.getLogger(__name__)

#: 每个证据单元的 token 估算 = 原文 + 上下文 + 固定开销。
#: 沿用检索层「一个字符约一个 token」的保守口径（`retrieval/service.py`）。
_UNIT_TOKEN_OVERHEAD = 64
DEFAULT_UNIT_TOKEN_BUDGET = 3000

#: 模型候选结论的尝试次数上限。重试只允许重新组织说法，
#: 每次重试都注入上一轮的硬性失败原因；证据集合在重试期间保持不变。
MAX_CLAIM_ATTEMPTS = 2

_SUB_QUESTION_SPLIT_RE = re.compile(r"[？?；;。]+")

#: 子问题 → 风险类别。顺序即优先级：配伍禁忌的说法最具体，
#: 「不宜与藜芦同用」同时含「不宜」与「同用」，必须判为配伍禁忌而不是使用禁忌。
_CATEGORY_TERMS: tuple[tuple[RiskCategory, tuple[str, ...]], ...] = (
    (
        RiskCategory.INCOMPATIBILITY,
        (
            "配伍禁忌",
            "十八反",
            "十九畏",
            "相反",
            "相畏",
            "相恶",
            "同用",
            "合用",
            "配伍",
            "不能同时",
            "不能搭配",
            "不能同用",
            "不宜同用",
            "不可同用",
            "不宜与",
            "不可与",
        ),
    ),
    (
        RiskCategory.CONTRAINDICATION,
        ("禁忌", "禁用", "忌用", "慎用", "孕妇", "妊娠", "儿童", "小儿", "不良反应", "不宜"),
    ),
    (
        RiskCategory.DOSAGE,
        (
            "剂量",
            "用量",
            "用法",
            "服用",
            "煎服",
            "多少克",
            "吃多少",
            "用多少",
            "使用多少",
            "多少会",
            "中毒",
            "中毒量",
        ),
    ),
)
_ALL_CATEGORIES: tuple[RiskCategory, ...] = (
    RiskCategory.DOSAGE,
    RiskCategory.CONTRAINDICATION,
    RiskCategory.INCOMPATIBILITY,
)


#: 高风险入口词表的补充项。这个判定是整条核验链路**唯一的总开关**：
#: 漏一个词，整类问题就完全不做核验，而且不会有任何报错——只是悄悄回到
#: 普通问答。实测「附子不宜与什么药同用？」在只有既有词表时判为**非**高风险，
#: 于是配伍禁忌这一整类问题从未进入核验链路。
#:
#: 因此必须补齐教材里禁忌最典型的写法：配伍禁忌几乎总是写成
#: 「不宜与…同用」，而既有词表收的是「配伍禁忌」「相反」这类**标题词**，
#: 问句里基本不会那样写；「禁用」也缺（原有「忌用」「慎用」）。
_EXTRA_HIGH_RISK_TERMS: tuple[str, ...] = (
    # 收「相畏」「十九畏」而不收单字「畏」：「畏寒」「畏光」是常见症状描述，
    # 单字会把它误判成高风险问题。
    "同用", "合用", "配伍", "十八反", "十九畏",
    "不能同时", "不能搭配", "不能同用", "不宜同用", "不可同用",
    "不宜与", "不可与",
    "禁用", "禁服", "切忌",
    "妊娠", "哺乳", "小儿", "新生儿", "婴幼儿",
    "不宜",
)


def is_high_risk_question(question: str, *, plan: object | None = None) -> bool:
    """Is this question one the high-risk path must handle?

    Args:
        question: 用户原始问题。
        plan: 可选的 `RetrievalPlan`。有 plan 且它已判为高风险时直接成立，
            保证与既有路由判定一致；没有 plan（如纯关键词图模式）时按词表自行判断。

    Returns:
        是否走高风险核验链路。这是**叠加策略**，不改变六意图枚举（任务书 §1）。

    Side effects: 无。
    """
    if plan is not None and getattr(plan, "is_high_risk", False):
        return True
    from ..retrieval.router import HIGH_RISK_TERMS

    normalized = "".join((question or "").split())
    return any(
        term in normalized for term in HIGH_RISK_TERMS + _EXTRA_HIGH_RISK_TERMS
    )


_SHARED_INCOMPAT_TERMS = (
    "均不能",
    "都不能",
    "均不宜",
    "都不宜",
    "均不可",
    "都不可",
    "都忌",
    "均忌",
)


def _asks_shared_incompatibility(question: str) -> bool:
    """问的是两味（及以上）药**共同**不能搭配的对象，而不是各药各自的禁忌。"""
    normalized = "".join((question or "").split())
    return any(term in normalized for term in _SHARED_INCOMPAT_TERMS)


def _incompat_objects_by_subject(
    subject_names: list[str], units: dict[str, ConditionalEvidenceUnit]
) -> dict[str, set[str]]:
    """按主语汇总配伍禁忌宾语。"""
    by_subject: dict[str, set[str]] = {name: set() for name in subject_names}
    for unit in units.values():
        if unit.predicate != "INCOMPATIBLE_WITH" or not unit.object_name:
            continue
        if unit.subject_name in by_subject:
            by_subject[unit.subject_name].add(unit.object_name)
    return by_subject


def _incompatibility_coverage_notes(
    question: str,
    subject_names: list[str],
    units: dict[str, ConditionalEvidenceUnit],
) -> list[str]:
    """配伍禁忌证据覆盖不全时生成说明。"""
    if not subject_names:
        return []
    if RiskCategory.INCOMPATIBILITY not in (categories_of(question) or ()):
        return []
    by_subject = _incompat_objects_by_subject(subject_names, units)
    notes: list[str] = []
    missing = [name for name, objects in by_subject.items() if not objects]
    if missing:
        notes.append("以下中药未检索到配伍禁忌证据：" + "、".join(missing))
    if len(subject_names) < 2:
        return notes
    present = [name for name, objects in by_subject.items() if objects]
    if len(present) == len(subject_names):
        common = set.intersection(*(by_subject[name] for name in subject_names))
        if common:
            notes.append("共同不宜同用对象：" + "、".join(sorted(common)))
            return notes
    notes.append(
        "现有证据中没有这些中药共同的不宜同用对象；"
        "不得把单味药的禁忌推广为两药共有"
    )
    return notes


def _incompatibility_answer_lines(
    question: str,
    subject_names: list[str],
    units: dict[str, ConditionalEvidenceUnit],
) -> list[str]:
    """主回答里必须点名每一味药，不能只写有证据的那一味。"""
    if not subject_names:
        return []
    if RiskCategory.INCOMPATIBILITY not in (categories_of(question) or ()):
        return []
    by_subject = _incompat_objects_by_subject(subject_names, units)
    lines: list[str] = []
    for name in subject_names:
        if by_subject.get(name):
            continue
        lines.append(
            f"{name}：现有证据未记载配伍禁忌，不能据此认为可以任意配伍。"
        )
    if len(subject_names) >= 2:
        present = [name for name, objects in by_subject.items() if objects]
        if len(present) == len(subject_names):
            common = set.intersection(*(by_subject[name] for name in subject_names))
            if common:
                lines.append("共同不宜同用对象：" + "、".join(sorted(common)))
            else:
                lines.append(
                    "两药没有共同的不宜同用对象，不得把单味药的禁忌推广为共有。"
                )
        else:
            lines.append(
                "两药没有共同的不宜同用对象，不得把单味药的禁忌推广为共有。"
            )
    return lines


def categories_of(question: str) -> tuple[RiskCategory, ...]:
    """Detect which of the three risk categories a question touches.

    返回空元组表示未识别出具体类别；调用方此时应覆盖全部三类，
    宁可多读也不漏读——漏读高风险证据的代价远高于多读。
    """
    normalized = "".join((question or "").split())
    matched = [
        category
        for category, terms in _CATEGORY_TERMS
        if any(term in normalized for term in terms)
    ]
    return tuple(dict.fromkeys(matched))


def _predicates_for(categories: tuple[RiskCategory, ...]) -> tuple[str, ...]:
    """按问题类型选出要读取的高风险谓词。"""
    effective = categories or _ALL_CATEGORIES
    return tuple(
        predicate
        for predicate, category in PREDICATE_CATEGORY.items()
        if category in effective
    )


_TOXIC_DOSE_TERMS: tuple[str, ...] = ("中毒", "中毒量", "多少会中毒")


def _is_toxic_dose_question(question: str) -> bool:
    """问的是中毒剂量／中毒阈值，而不是常规用量或一般使用注意。"""
    normalized = "".join((question or "").split())
    return any(term in normalized for term in _TOXIC_DOSE_TERMS)


def _split_sub_questions(question: str) -> list[str]:
    """Split a compound question into independently answerable parts.

    每个子问题单独定状态，这样「用量查到了、孕妇禁忌没查到」可以如实表达为
    部分成立，而不是被压成一句笼统的「证据不足」。
    """
    parts = [part.strip() for part in _SUB_QUESTION_SPLIT_RE.split(question or "")]
    return [part for part in parts if part] or [(question or "").strip()]


def _unit_cost(unit: ConditionalEvidenceUnit) -> int:
    """估算一条证据单元占用的上下文预算。"""
    return len(unit.evidence_quote) + len(unit.context) + _UNIT_TOKEN_OVERHEAD


def _trim_units(
    units: list[ConditionalEvidenceUnit],
    budget: int,
    *,
    prefer: tuple[RiskCategory, ...] = (),
) -> tuple[list[ConditionalEvidenceUnit], int]:
    """Trim by **whole units**, best evidence first.

    关键：裁剪的单位是单元而不是字段。数值、单位、用法、生制品、人群、限制程度
    同属一个单元，要么整条保留，要么整条移出可引用集合——不会出现
    「保留了剂量却丢掉了它的慎用程度」（任务书 §5）。

    Args:
        units: 候选单元。
        budget: token 预算。
        prefer: 本子问题问到的风险类别，**按优先级排列**（由 `categories_of`
            给出，越靠前越贴近提问）。不能只按固定类别顺序排：固定顺序会让
            「XX 不宜与什么药同用」（配伍禁忌）在预算不足时把配伍禁忌单元裁掉、
            只留下使用禁忌——问什么就丢什么，而答案读起来仍然像模像样，
            这是最危险的一种错。

            用**位次**而不是集合成员判定：「不宜与…同用」同时命中配伍禁忌
            （同用）与使用禁忌（不宜），按成员判定时两类都是「命中」，
            等于没有区分；按位次则配伍禁忌在前，符合「同用」才是这个问句的
            主张。本类别未命中时统一排在其后，因此不会把证据整类排除。
            传空元组即退回与类别无关的固定顺序（保持既有行为）。

    Returns:
        (保留的单元, 估算 token)。被移除的单元同时从可引用集合中消失，
        因此候选结论无法引用已经不在上下文里的证据。
    """
    ordered = sorted(
        units,
        key=lambda unit: (
            not unit.is_citable(),
            not unit.is_clearly_parsed,
            prefer.index(unit.risk_category) if unit.risk_category in prefer else len(prefer),
            _ALL_CATEGORIES.index(unit.risk_category),
        ),
    )
    selected: list[ConditionalEvidenceUnit] = []
    used = 0
    for unit in ordered:
        cost = _unit_cost(unit)
        if selected and used + cost > budget:
            continue
        selected.append(unit)
        used += cost
    # 恢复稳定顺序，避免排序影响下游的可复现性。
    selected.sort(key=lambda unit: unit.unit_id)
    return selected, used


def claims_from_units(units: list[ConditionalEvidenceUnit]) -> list[CandidateClaim]:
    """Deterministic baseline claims built straight from the structured units.

    这不是「模型候选」，而是「结构化单元的原样陈述」。它使高风险链路在没有
    配置模型时依然可以给出**有据可查**的结论，而不是退化成什么都答不了。
    它同样要过核验：核验在这里检查的是「渲染出来的结论文本是否仍然与单元
    逐项一致」，即渲染环节有没有把数值与条件挪位或漏掉。
    """
    claims: list[CandidateClaim] = []
    for unit in units:
        if not unit.is_citable() or not unit.is_clearly_parsed:
            continue
        if unit.risk_category is RiskCategory.DOSAGE:
            if unit.dosage is None or not unit.dosage.has_numeric_value():
                continue
            kind = ClaimKind.DOSAGE
            numbers = [float(unit.dosage.value_min), float(unit.dosage.value_max)]
        elif unit.risk_category is RiskCategory.CONTRAINDICATION:
            kind = ClaimKind.CONTRAINDICATION
            numbers = []
        else:
            kind = ClaimKind.INCOMPATIBILITY
            numbers = []
        claims.append(
            CandidateClaim(
                claim_id=f"unit-baseline:{unit.unit_id}",
                kind=kind,
                statement=render_unit_statement(unit),
                subject_name=unit.subject_name,
                object_name=unit.object_name,
                asserted_preparation=unit.preparation,
                asserted_route=unit.route,
                asserted_population=unit.population,
                asserted_frequency=unit.frequency,
                asserted_severity=(
                    unit.severity
                    if unit.severity is not RestrictionSeverity.UNSPECIFIED
                    else None
                ),
                asserted_polarity=unit.polarity_verdict,
                asserted_numbers=numbers,
                asserted_unit=unit.dosage.unit if unit.dosage else None,
                cited_unit_ids=[unit.unit_id],
            )
        )
    return claims


def _llm_claims(
    llm,
    *,
    question: str,
    units: list[ConditionalEvidenceUnit],
    feedback: list[dict] | None = None,
) -> list[CandidateClaim]:
    """Ask the model for structured candidate claims over the injected units."""
    evidence_lines = []
    for unit in units:
        evidence_lines.append(
            f"[{unit.unit_id}] 类别={unit.risk_category.value}｜谓词={unit.predicate}"
            f"｜主语={unit.subject_name}"
            + (f"｜对象={unit.object_name}" if unit.object_name else "")
            + (f"｜条件={unit.condition_summary()}" if unit.condition_summary() else "")
            + f"\n原文：{unit.evidence_quote}"
        )
    feedback_block = ""
    if feedback:
        feedback_block = (
            "\n# 上一轮被程序核验驳回的原因（必须修正，不得重复同类错误）\n"
            + "\n".join(
                f"- {item['claim_id']}：{'；'.join(item['notes']) or '、'.join(item['hard_failures'])}"
                for item in feedback
            )
            + "\n"
        )
    prompt = f"""你是中药学教材问答的候选结论生成器。只允许依据下面给出的证据单元作答。

# 硬性规则
1. 每条结论必须引用真实的 unit_id，只能引用下面出现过的标识。
2. 不得补充教材未说明的条件：不写「成人」「每日」「每次」等原文没有的信息。
3. 不得把通用说明推成孕妇、儿童等特定人群的适用结论。
4. 不得把「未检索到禁忌」表述为「可以同用」「无禁忌」「安全」。
5. 数值必须与它的用法、生制品、人群、限制程度整体来自同一个证据单元，
   不得把不同单元的数值与条件拼接。
6. 保留原文的禁用／忌用／慎用区别，不得改写限制程度。

# 问题
{question}
{feedback_block}
# 证据单元（unit_id 是唯一合法引用标识）
{chr(10).join(evidence_lines)}

# 输出
按给定 JSON Schema 输出，字段 asserted_* 必须与你写的结论一致。
没有证据支持的方面不要编写结论。
"""
    result = llm.complete_json(
        system="你是严谨的中药学教材问答助手，只依据给定证据作答。",
        user=prompt,
        schema=ClaimBatch,
        purpose="high_risk_claims",
    )
    if isinstance(result, ClaimBatch):
        return list(result.claims)
    return []


class HighRiskService:
    """Read-only high-risk question answering with mandatory verification.

    典型用法::

        service = HighRiskService(store, run_id=active_run, llm=client)
        answer = service.answer("麻黄用量是多少？孕妇能用吗？", linked_entities=linked)
    """

    def __init__(
        self,
        store,
        *,
        run_id: str,
        llm=None,
        assertion_limit: int = DEFAULT_ASSERTION_LIMIT,
        unit_token_budget: int = DEFAULT_UNIT_TOKEN_BUDGET,
    ) -> None:
        """初始化该对象，保存后续调用所需的配置或依赖。"""
        self._store = store
        self._run_id = run_id
        self._llm = llm
        self._assertion_limit = assertion_limit
        self._unit_token_budget = unit_token_budget

    # ---- 证据读取 --------------------------------------------------------
    def _read_rows(
        self, question: str, *, subject_ids: list[str], object_ids: list[str]
    ) -> list[dict]:
        """从存储层读取高风险相关断言行。"""
        predicates = _predicates_for(categories_of(question))
        if subject_ids:
            return load_high_risk_assertions(
                self._store,
                run_id=self._run_id,
                subject_ids=subject_ids,
                predicates=predicates,
                limit=self._assertion_limit,
            )
        # 没有识别出药物主语时，**只有明确的反向枚举问法**才允许按
        # 「人群／病证约束」反向读取，例如「孕妇禁用的中药有哪些」。
        #
        # 这条兜底曾经对任何无主语问题都生效，于是「布洛芬孕妇能吃吗？」
        # 会把徐长卿、千金子、厚朴的孕妇禁忌当作答案端出来，还标 grounded。
        # 反向读取是正当功能，但只对「列出满足条件的一组对象」这种问法正当；
        # 指定了对象而对象没解析出来时，正确的动作是弃权，不是换个对象作答。
        if not is_enumeration_question(question):
            return []
        population = extract_population(question)
        return load_high_risk_assertions_by_object(
            self._store,
            run_id=self._run_id,
            object_ids=object_ids,
            object_keywords=[population] if population else None,
            predicates=predicates,
            limit=self._assertion_limit,
        )

    def _subject_conflict(self, question: str, subject_names: list[str]) -> str | None:
        """问题里是否出现了比已识别主语**更长**的对象名。

        链接器按子串匹配，「安宫牛黄丸」里的「牛黄」会被当成主语，接着用
        牛黄的禁忌回答安宫牛黄丸的问题。这里只报告「名称被截断」这一事实，
        不重做消歧、不替换主语——由本层拒绝把它当作用户问的对象。

        两条依据：
        1. 本体里存在更长的实体名（`find_longer_entity_mentions`）；
        2. 命中的名称直接续着成药剂型字（`longer_object_span`）。

        第 1 条给出的是**完整**的对象名，优先用它；只有第 1 条不够：
        `龙胆泻肝丸` 在本体里没有实体，第 1 条不触发，
        「龙胆泻肝丸的用量是多少？」就会拿「龙胆」的剂量作答并标记已核验，
        所以再加第 2 条形态判定作为兜底。
        """
        from ..storage import serving_read

        for name in subject_names:
            if not name:
                continue
            longer = serving_read.find_longer_entity_mentions(
                self._store, question, name=name, run_id=self._run_id
            )
            span = longer_object_span(question, name) if not longer else None
            if longer or span:
                mentioned = longer[0]["preferred_name"] if longer else span
                return (
                    f"问题中的「{mentioned}」比已识别的「{name}」更长，"
                    f"链接只命中了名称子串，无法确认为同一对象；"
                    f"不能把「{name}」的证据当作该问题的答案"
                )
        return None

    # ---- 主流程 ----------------------------------------------------------
    def answer(
        self,
        question: str,
        *,
        linked_entities: list[dict] | None = None,
        use_llm: bool | None = None,
    ) -> HighRiskAnswer:
        """Answer a high-risk question end to end.

        Args:
            question: 用户问题。
            linked_entities: 上游既有实体链接的结果（本方法**不重做**消歧）。
                只取 `entity_type == "HERB"` 的项作为主语。
            use_llm: 是否调用模型生成候选结论。None 表示「配置了模型就调用」。

        Returns:
            HighRiskAnswer。`status` 由各子问题的实际证据情况决定；
            只有全部子问题都有通过核验的结论才是 `grounded`。

        Side effects: 只读数据库。
        """
        notes: list[str] = []
        linked = list(linked_entities or [])
        # 链接器返回行用 `name`，上游检索层传下来的用 `preferred_name`，两者都认。
        subject_pairs = [
            (
                str(entity["canonical_id"]),
                str(entity.get("preferred_name") or entity.get("name") or ""),
            )
            for entity in linked
            if entity.get("entity_type") == "HERB" and entity.get("canonical_id")
        ]
        # 泛称不算「用户指定的对象」。本体里「中药」是一个 HERB，若不排除，
        # 「孕妇禁用的中药有哪些？」会走正向读取并读空——真正该走反向枚举的
        # 问法反而答不出来。
        subject_pairs = [
            (canonical_id, name)
            for canonical_id, name in subject_pairs
            if not is_generic_object(name)
        ]
        subject_ids = [canonical_id for canonical_id, _ in subject_pairs]
        subject_names = [name for _, name in subject_pairs]
        object_ids = [
            str(entity["canonical_id"])
            for entity in linked
            if entity.get("entity_type") != "HERB" and entity.get("canonical_id")
        ]

        sub_questions = _split_sub_questions(question)

        # 对象核对：链接只命中了名称子串时，用户问的对象并未被识别出来。
        # 此时**不得**用那个更短名称的证据作答——「安宫牛黄丸」的问题不能用
        # 「牛黄」的禁忌来回答。这是对象错配，与条件错配分开处理。
        conflict = (
            self._subject_conflict(question, subject_names) if subject_ids else None
        )
        if conflict:
            early = _first_decision(sub_questions)
            text = _lead_with_decision(early.text, render_abstain(conflict))
            return HighRiskAnswer.build(
                status=AnswerStatus.ABSTAIN,
                answer_text=text,
                sub_questions=[
                    SubQuestionOutcome(
                        question=sub_question,
                        risk_category=next(iter(categories_of(sub_question)), None),
                        status=AnswerStatus.ABSTAIN,
                        reason=conflict,
                    )
                    for sub_question in sub_questions
                ],
                claims=[],
                evidence_units=[],
                coverage_notes=notes + [NO_EVIDENCE_NOTE],
                decision=early.decision.value,
                decision_text=early.text,
                answer_blocks=[AnswerBlock(kind=AnswerBlockKind.TEXT, text=text)],
            )
        units_by_id: dict[str, ConditionalEvidenceUnit] = {}
        units_by_sub: dict[str, list[str]] = {}
        #: 读到的全部单元（**裁剪之前**）。只用于冲突检测：被预算裁掉的来源
        #: 也是来源，它与保留下来的一处在程度上不一致时，用户必须知道。
        read_units_by_id: dict[str, ConditionalEvidenceUnit] = {}

        for sub_question in sub_questions:
            rows = self._read_rows(
                sub_question, subject_ids=subject_ids, object_ids=object_ids
            )
            sub_units = build_units_from_assertions(rows)
            for unit in sub_units:
                read_units_by_id.setdefault(unit.unit_id, unit)
            trimmed, _ = _trim_units(
                sub_units,
                self._unit_token_budget,
                prefer=categories_of(sub_question),
            )
            if len(trimmed) < len(sub_units):
                notes.append(
                    f"子问题「{sub_question}」证据按 token 预算按单元裁剪："
                    f"保留 {len(trimmed)}/{len(sub_units)} 个单元"
                )
            for unit in trimmed:
                units_by_id[unit.unit_id] = unit
            units_by_sub[sub_question] = [unit.unit_id for unit in trimmed]

        # 全部子问题共享同一批最终注入证据，交叉引用因此不会越界。
        injected = [units_by_id[unit_id] for unit_id in sorted(units_by_id)]
        injected_ids = [unit.unit_id for unit in injected]

        if not injected:
            notes.extend(
                _incompatibility_coverage_notes(question, subject_names, units_by_id)
            )
            if subject_ids or object_ids:
                reason = "未在 ACTIVE 批次中检索到这三类高风险关系的证据"
            elif is_enumeration_question(question):
                reason = "问题中没有识别到药材或限定对象，也没有检索到可用证据"
            else:
                # 说清楚这是**拒绝替换对象**，而不是「没查到」：两者对用户的
                # 含义完全不同——「没查到」听起来像「大概没事」，而这里
                # 的问题是问的对象根本没识别出来。
                reason = (
                    "问题中没有识别到药材或限定对象，且该问法不是反向枚举，"
                    "因此不按人群反向读取其他对象的证据："
                    "其他对象的禁忌不能回答本例，本问题不予结论"
                )
            early = _first_decision(sub_questions, subject_pool=subject_names)
            text = _lead_with_decision(early.text, render_abstain(reason))
            return HighRiskAnswer.build(
                status=AnswerStatus.ABSTAIN,
                answer_text=text,
                sub_questions=[
                    SubQuestionOutcome(
                        question=sub_question,
                        risk_category=next(iter(categories_of(sub_question)), None),
                        status=AnswerStatus.ABSTAIN,
                        reason=reason,
                    )
                    for sub_question in sub_questions
                ],
                claims=[],
                evidence_units=[],
                coverage_notes=notes + [NO_EVIDENCE_NOTE],
                decision=early.decision.value,
                decision_text=early.text,
                answer_blocks=[AnswerBlock(kind=AnswerBlockKind.TEXT, text=text)],
            )

        if use_llm is None:
            use_llm = self._llm is not None

        baseline = claims_from_units(injected)
        outcome = verify_claims(baseline, injected, allowed_unit_ids=injected_ids)
        accepted_units = _accepted_units(outcome, injected)
        rejected: list[dict] = list(outcome.rejections)

        if use_llm and self._llm is not None:
            llm_outcome = self._verified_llm_claims(
                question=question, injected=injected, injected_ids=injected_ids
            )
            if llm_outcome is not None:
                rejected.extend(llm_outcome.rejections)
                accepted_units = _merge_units(
                    accepted_units, _accepted_units(llm_outcome, injected)
                )
                if llm_outcome.rejected and not llm_outcome.accepted:
                    notes.append(
                        "模型候选结论全部被程序核验驳回，已回退到结构化单元直接陈述："
                        + "；".join(
                            sorted(
                                {
                                    failure
                                    for item in llm_outcome.rejections
                                    for failure in item["hard_failures"]
                                }
                            )
                        )
                    )

        # 同一对象同一条件下程度不一致时，并列保留并标明冲突：不取平均、
        # 不求并集，**也不按更严的一档表述**。自行挑一档等于替用户做了
        # 一次未经核验的临床取舍。
        # 冲突在**全部读到的单元**里查，而不是只看已注入／已采纳的：
        # 两处来源程度不一致时用户必须知道这里有分歧——哪怕其中一处没过
        # 第一层核验，也哪怕另一处被 token 预算裁掉了。若只在注入集合里查，
        # 裁掉「慎用」只留「禁用」会让回答**看起来**是一条没有分歧的结论，
        # 等于由裁剪替用户选了一档（实测：川乌孕妇，第 179 页慎用被裁）。
        conflict_map = severity_conflicts(
            [unit for unit in read_units_by_id.values() if unit.is_citable()]
        )
        if conflict_map:
            for note in dict.fromkeys(conflict_map.values()):
                notes.append(note)
            # 被裁掉的那一处无法并列展示，必须写明，否则「并列保留」名不副实。
            for unit_id in conflict_map:
                if unit_id not in injected_ids:
                    notes.append(
                        f"[{unit_id}] 与本次结论程度不一致的来源单元因 token 预算"
                        "未纳入本次上下文，无法并列展示；本次结论因此降级为部分成立，"
                        "不得当作已经核验的唯一结论"
                    )

        accepted_ids = {unit.unit_id for unit in accepted_units}

        sub_outcomes = [
            self._sub_outcome(
                sub_question,
                units_by_sub,
                units_by_id,
                accepted_ids,
                conflict_map=conflict_map,
            )
            for sub_question in sub_questions
        ]
        incompat_notes = _incompatibility_coverage_notes(
            question, subject_names, units_by_id
        )
        notes.extend(incompat_notes)
        no_shared = next(
            (
                note
                for note in incompat_notes
                if "没有这些中药共同的不宜同用对象" in note
            ),
            None,
        )
        if no_shared and _asks_shared_incompatibility(question):
            sub_outcomes = [
                (
                    item.model_copy(
                        update={
                            "status": AnswerStatus.PARTIAL_GROUNDED,
                            "reason": no_shared,
                        }
                    )
                    if item.status is AnswerStatus.GROUNDED
                    else item
                )
                for item in sub_outcomes
            ]
        status = _overall_status(sub_outcomes)

        # 只有**适用于本问题对象与条件**的单元才作为结论输出。第一层核验通过
        # 只说明结论忠实于证据，不说明证据回答了用户问的问题；被适用性核验
        # 拦下的单元仍以原文形式留在 `evidence_units` 里供核对，但不得进入
        # `claims`，也不得进入结论文本——否则「孕妇慎用」会继续以结论的样子
        # 出现在「哺乳期能否使用」的回答里。
        #
        # 展示顺序按限制程度从重到轻，使「禁用」与「慎用」一眼可分；
        # 只改顺序，不改内容，也不把不同程度合并。
        presentable_ids = {
            unit_id
            for item in sub_outcomes
            if item.status in (AnswerStatus.GROUNDED, AnswerStatus.PARTIAL_GROUNDED)
            for unit_id in item.unit_ids
        }
        # 第三部分：问题条件相关性选择（任务书 §2）。只作用于**展示**：
        # 检索、两层核验与状态判定都已经在各自的位置完成，这里只回答
        # 「哪些证据是对本题明说的条件说的」。
        relevance_map = _relevance_map(sub_outcomes, units_by_id)
        presentable_all = [
            units_by_id[unit_id]
            for unit_id in sorted(presentable_ids)
            if unit_id in units_by_id
        ]
        # C 类（明确指向别的条件）不进主回答、不进 claims、不进 evidence_units，
        # 也不进 sources/subgraph——本题问的是孕妇，「阴虚阳亢者忌用」在本题里
        # 不是证据，只是相邻条目。它单独放进 `other_conditions`，
        # 由前端放在默认折叠的「其他使用注意（与本题指定人群不同）」里。
        presentable = [
            unit
            for unit in presentable_all
            if relevance_map.get(unit.unit_id, Relevance.DIRECT)
            is not Relevance.MISMATCH
        ]
        other_units = [
            units_by_id[unit_id]
            for unit_id, verdict in sorted(relevance_map.items())
            if verdict is Relevance.MISMATCH and unit_id in units_by_id
        ]
        # 相关性优先于限制程度：精确匹配排前，无条件通用其次，同级内才比重。
        shown = sort_by_relevance(presentable, relevance_map)
        # `evidence_units` 是本答案里**可核查的全部证据**，必须涵盖 claims 引用的
        # 每一个单元（同源）：除 C 类之外一律保留，包括仅可定位原文的单元。
        diverted_ids = {unit.unit_id for unit in other_units}
        evidence_units = [
            unit for unit in injected if unit.unit_id not in diverted_ids
        ]
        if other_units:
            notes.append(
                "以下证据与本题指定条件不同，已移出主要回答，仅供对照："
                + "；".join(
                    f"[{unit.unit_id}] {unit.condition_summary() or unit.object_name or ''}"
                    for unit in other_units
                )
            )

        # 首句判定与冲突判定看同一批证据：`conflict_map` 是在**裁剪之前**的全部
        # 读到单元上算出来的，首句也必须看到同一批，否则会出现「首句说禁用、
        # 下面又说存在不一致」的自相矛盾（实测川乌孕妇：第 179 页的慎用被裁掉）。
        decision_result = _decision_for(
            sub_outcomes,
            units_by_id,
            relevance_map,
            read_units=[
                unit for unit in read_units_by_id.values() if unit.is_citable()
            ],
        )
        verification_note = verification_note_for(status)

        claims = [
            VerifiedClaim(
                claim_id=f"claim:{unit.unit_id}",
                kind=_kind_of(unit),
                statement=render_unit_statement(unit),
                subject_name=unit.subject_name,
                object_name=unit.object_name,
                conditions=unit.condition_summary(),
                unit_ids=[unit.unit_id],
                pages=[_page_label(unit)],
            )
            for unit in shown
        ]
        answer_text, blocks = _compose_answer(
            status=status,
            decision=decision_result,
            verification_note=verification_note,
            shown_units=shown,
            relevance=relevance_map,
            all_units=injected,
            sub_outcomes=sub_outcomes,
            subject_gap_lines=_incompatibility_answer_lines(
                question, subject_names, units_by_id
            ),
        )
        if status is not AnswerStatus.GROUNDED:
            notes.append(NO_EVIDENCE_NOTE)
        for unit in injected:
            for conflict in unit.conflicts:
                notes.append(f"[{unit.unit_id}] {conflict}")

        return HighRiskAnswer.build(
            status=status,
            answer_text=answer_text,
            sub_questions=sub_outcomes,
            claims=claims,
            evidence_units=[EvidenceCard.from_unit(unit) for unit in evidence_units],
            rejections=rejected,
            coverage_notes=list(dict.fromkeys(notes)),
            decision=decision_result.decision.value,
            decision_text=decision_result.text,
            verification_note=verification_note,
            answer_blocks=blocks,
            other_conditions=[EvidenceCard.from_unit(unit) for unit in other_units],
        )

    def _verified_llm_claims(
        self,
        *,
        question: str,
        injected: list[ConditionalEvidenceUnit],
        injected_ids: list[str],
    ) -> VerificationOutcome | None:
        """Call the model, verify, and retry once with the failure reasons.

        重试只重新组织说法，**不改变证据集合**：被驳回的是表达方式，不是证据。
        重试次数有上限，避免在同一个错误上循环消耗。
        """
        feedback: list[dict] | None = None
        last: VerificationOutcome | None = None
        for attempt in range(1, MAX_CLAIM_ATTEMPTS + 1):
            try:
                claims = _llm_claims(
                    self._llm, question=question, units=injected, feedback=feedback
                )
            except Exception as exc:  # 模型不可用不应让整个请求失败
                logger.warning("high_risk_llm_failed: %s", exc)
                return last
            if not claims:
                return last
            last = verify_claims(claims, injected, allowed_unit_ids=injected_ids)
            if last.accepted or attempt == MAX_CLAIM_ATTEMPTS:
                return last
            feedback = last.rejections
        return last

    def _sub_outcome(
        self,
        sub_question: str,
        units_by_sub: dict[str, list[str]],
        units_by_id: dict[str, ConditionalEvidenceUnit],
        accepted_ids: set[str],
        *,
        conflict_map: dict[str, str] | None = None,
    ) -> SubQuestionOutcome:
        """组装一个子问题的核验结果。"""
        sub_ids = units_by_sub.get(sub_question, [])
        categories = categories_of(sub_question)
        category = categories[0] if categories else None
        allowed = set(categories)
        scoped_ids = [
            unit_id
            for unit_id in sub_ids
            if not allowed or units_by_id[unit_id].risk_category in allowed
        ]
        citable = [
            unit_id
            for unit_id in scoped_ids
            if units_by_id[unit_id].is_citable()
        ]
        grounded_ids = [unit_id for unit_id in citable if unit_id in accepted_ids]

        # 第二层核验：这些证据是否**回答了用户所问的对象和条件**。
        # grounded 不能只由「存在通过核验的证据单元」决定——那会把
        # 「忠实复述了孕妇慎用」当成「回答了哺乳期能否使用」。
        assessment = assess_applicability(
            sub_question, [units_by_id[unit_id] for unit_id in grounded_ids]
        )
        if not assessment.allows_grounded:
            return SubQuestionOutcome(
                question=sub_question,
                risk_category=category,
                status=AnswerStatus.SOURCE_ONLY,
                reason=assessment.reason,
                unit_ids=citable,
            )

        # 程度冲突未解决时不能给出单一确定结论，降级为部分成立并并列说明。
        conflicting = [
            unit_id
            for unit_id in grounded_ids
            if conflict_map and unit_id in conflict_map
        ]
        if conflicting:
            return SubQuestionOutcome(
                question=sub_question,
                risk_category=category,
                status=AnswerStatus.PARTIAL_GROUNDED,
                reason=conflict_map[conflicting[0]],
                unit_ids=grounded_ids,
                claim_ids=[f"claim:{unit_id}" for unit_id in grounded_ids],
            )

        if grounded_ids and _is_toxic_dose_question(sub_question):
            toxic_ids = [
                unit_id
                for unit_id in grounded_ids
                if "中毒" in (units_by_id[unit_id].evidence_quote or "")
            ]
            if not toxic_ids:
                return SubQuestionOutcome(
                    question=sub_question,
                    risk_category=category,
                    status=AnswerStatus.PARTIAL_GROUNDED,
                    reason=(
                        "教材未记载明确中毒剂量；下列仅为常规用量，"
                        "不能当作中毒阈值，也不能用其他使用注意代替回答"
                    ),
                    unit_ids=grounded_ids,
                    claim_ids=[f"claim:{unit_id}" for unit_id in grounded_ids],
                )
        if grounded_ids:
            return SubQuestionOutcome(
                question=sub_question,
                risk_category=category,
                status=AnswerStatus.GROUNDED,
                reason="存在通过程序核验、且适用于本问题对象与条件的证据单元",
                unit_ids=grounded_ids,
                claim_ids=[f"claim:{unit_id}" for unit_id in grounded_ids],
            )
        if citable:
            blocked = [
                units_by_id[unit_id].undetermined_reason
                or "；".join(units_by_id[unit_id].conflicts)
                or "该单元未能形成确定结论"
                for unit_id in citable
            ]
            return SubQuestionOutcome(
                question=sub_question,
                risk_category=category,
                status=AnswerStatus.SOURCE_ONLY,
                reason="以下原文可定位但未能形成通过核验的确定结论：" + "；".join(blocked),
                unit_ids=citable,
            )
        return SubQuestionOutcome(
            question=sub_question,
            risk_category=category,
            status=AnswerStatus.ABSTAIN,
            reason="未检索到支持该子问题的证据",
        )


def _kind_of(unit: ConditionalEvidenceUnit) -> ClaimKind:
    """判断结论种类（剂量/禁忌/配伍禁忌）。"""
    if unit.risk_category is RiskCategory.DOSAGE:
        return ClaimKind.DOSAGE
    if unit.risk_category is RiskCategory.CONTRAINDICATION:
        return ClaimKind.CONTRAINDICATION
    return ClaimKind.INCOMPATIBILITY


def _page_label(unit: ConditionalEvidenceUnit) -> str:
    """生成页码标签。"""
    if not unit.page_available or unit.page_start is None:
        return "页码不可用"
    if unit.page_end in (None, unit.page_start):
        return f"第 {unit.page_start} 页"
    return f"第 {unit.page_start}-{unit.page_end} 页"


def _accepted_units(
    outcome: VerificationOutcome, injected: list[ConditionalEvidenceUnit]
) -> list[ConditionalEvidenceUnit]:
    """筛出核验通过的证据单元。"""
    accepted_ids = {
        unit_id for verdict in outcome.accepted for unit_id in verdict.supporting_unit_ids
    }
    return [unit for unit in injected if unit.unit_id in accepted_ids]


def _merge_units(
    baseline: list[ConditionalEvidenceUnit],
    extra: list[ConditionalEvidenceUnit],
) -> list[ConditionalEvidenceUnit]:
    """合并重复或可并列的证据单元。"""
    merged = {unit.unit_id: unit for unit in baseline}
    for unit in extra:
        merged.setdefault(unit.unit_id, unit)
    return sorted(merged.values(), key=lambda unit: unit.unit_id)


def _overall_status(sub_outcomes: list[SubQuestionOutcome]) -> AnswerStatus:
    """Fold per-sub-question states into one overall state.

    只有每个子问题都成立才是 `grounded`——「问了三件事答上两件」必须如实
    表达为部分成立，不能被一句笼统的完整回答盖过去（任务书 §6）。
    """
    if not sub_outcomes:
        return AnswerStatus.ABSTAIN
    grounded = [item for item in sub_outcomes if item.status is AnswerStatus.GROUNDED]
    if len(grounded) == len(sub_outcomes):
        return AnswerStatus.GROUNDED
    # 子问题级的部分成立（例如程度冲突未解决）同样构成「部分」，
    # 不能因为没有子问题达到 grounded 就被压成弃权——那会连已经支持的
    # 部分一起丢掉。
    partial = [
        item for item in sub_outcomes if item.status is AnswerStatus.PARTIAL_GROUNDED
    ]
    if grounded or partial:
        return AnswerStatus.PARTIAL_GROUNDED
    if any(item.status is AnswerStatus.SOURCE_ONLY for item in sub_outcomes):
        return AnswerStatus.SOURCE_ONLY
    return AnswerStatus.ABSTAIN


#: 能由「能否使用」首句一言以蔽之的类别。剂量类证据有自己的首句形态
#: （数值在结论行里），首句不覆盖它们。
_DECISION_CATEGORIES: tuple[RiskCategory, ...] = (
    RiskCategory.CONTRAINDICATION,
    RiskCategory.INCOMPATIBILITY,
)


def _relevance_map(
    sub_outcomes: list[SubQuestionOutcome],
    units_by_id: dict[str, ConditionalEvidenceUnit],
) -> dict[str, Relevance]:
    """每个单元相对**提出它的那个子问题**的相关档位。

    覆盖全部状态，不只 grounded：仅来源（source_only）的回答正文同样是用户
    读到的主要文字，那里出现一条别的病证的「忌用」一样会被当成答案。

    一个单元可能同时属于多个子问题（例如「附子孕妇能用吗？用量多少？」），
    此时取**最相关**的那一档：只要它在任何一个子问题里与问题条件一致，
    就不该因为另一个子问题而被判成其他条件。
    """
    best: dict[str, Relevance] = {}
    for item in sub_outcomes:
        for unit_id in item.unit_ids:
            unit = units_by_id.get(unit_id)
            if unit is None:
                continue
            verdict = classify(item.question, unit)
            current = best.get(unit_id)
            if current is None or ordered(verdict) < ordered(current):
                best[unit_id] = verdict
    return best


def _decision_for(
    sub_outcomes: list[SubQuestionOutcome],
    units_by_id: dict[str, ConditionalEvidenceUnit],
    relevance: dict[str, Relevance],
    *,
    read_units: list[ConditionalEvidenceUnit] | None = None,
) -> DecisionResult:
    """整份答案的首句：取**第一个能给出首句的子问题**。

    只用该子问题自己的证据，别的子问题的证据不参与它的判定——否则
    「用量是多少？孕妇能用吗？」里的用量证据会被算进孕妇那一句。
    子问题级的部分成立同样要有一个首句，所以 ABSTAIN 的子问题也参与：
    「无法判断孕妇能否使用该药」正是这种情况该说的话。

    `read_units` 是本题**读到**的全部可引用单元（含因 token 预算未进入上下文的）。
    只在同一主语、同一人群下补齐程度集合：两处来源程度不一致时首句必须按冲突
    表述，不能因为其中一处没进上下文就把首句写成单一的一档。
    """
    pool = [
        unit.subject_name for unit in units_by_id.values() if unit.subject_name
    ]
    for item in sub_outcomes:
        own = [units_by_id[unit_id] for unit_id in item.unit_ids if unit_id in units_by_id]
        result = decide(
            item.question,
            own,
            relevance=relevance,
            subject_pool=pool,
            read_units=read_units,
        )
        if result.decision is not Decision.NONE:
            return result
    return DecisionResult(Decision.NONE)


def _first_decision(
    sub_questions: list[str], *, subject_pool: list[str] | None = None
) -> DecisionResult:
    """还没有读到任何证据时（弃权分支）的首句。"""
    for sub_question in sub_questions:
        result = decide(sub_question, [], subject_pool=list(subject_pool or []))
        if result.decision is not Decision.NONE:
            return result
    return DecisionResult(Decision.NONE)


def _lead_with_decision(decision_text: str, body: str) -> str:
    """把直接结论放在最前面（§7 第 1 位）。没有首句时原样返回。"""
    if not decision_text:
        return body
    return f"{decision_text}\n\n{body}"


def _compose_answer(
    *,
    status: AnswerStatus,
    decision: DecisionResult,
    verification_note: str,
    shown_units: list[ConditionalEvidenceUnit],
    relevance: dict[str, Relevance],
    all_units: list[ConditionalEvidenceUnit],
    sub_outcomes: list[SubQuestionOutcome],
    subject_gap_lines: list[str] | None = None,
) -> tuple[str, list[AnswerBlock]]:
    """按状态组装最终文字与展示分块（任务书第三部分 §7）。

    返回 `(answer_text, blocks)`：两者描述**同一批内容**。`answer_text` 供
    不解析分块的调用方使用（接口契约、评测脚本），`blocks` 供前端按种类套样式。
    前端不再靠查字符串决定样式——核验说明尤其如此，它有自己的字段。

    顺序固定为：直接结论 → 必要的条件解释 → 原文依据 → 冲突或不足说明 →
    程序核验说明。核验说明是**收尾担保**，因此从原来的开头挪到末尾。
    """
    reasons = "；".join(
        f"{item.question}：{item.reason}"
        for item in sub_outcomes
        if item.status is not AnswerStatus.GROUNDED
    )

    if status in (AnswerStatus.GROUNDED, AnswerStatus.PARTIAL_GROUNDED):
        blocks: list[AnswerBlock] = []
        parts: list[str] = []
        if decision.text:
            blocks.append(AnswerBlock(kind=AnswerBlockKind.DECISION, text=decision.text))
            parts.append(decision.text)

        # 「必要的条件解释」：首句已经完整说出「药名＋人群＋程度」的证据不必
        # 再列一遍结构行。列出来只会让同一结论在答案里出现两次；
        # 而首句覆盖不到的证据（别的程度、别的类别、无首句的问法）
        # 必须逐条列出，否则它的限制程度就在答案里丢失了。
        covered = decision.severities
        explained = [
            unit
            for unit in shown_units
            if not (
                covered
                and unit.severity in covered
                and unit.risk_category in _DECISION_CATEGORIES
                and relevance.get(unit.unit_id) is Relevance.DIRECT
            )
        ]
        if explained:
            text = "\n".join(f"- {render_conclusion_line(unit)}" for unit in explained)
            blocks.append(
                AnswerBlock(
                    kind=AnswerBlockKind.CONCLUSION,
                    text=text,
                    unit_ids=[unit.unit_id for unit in explained],
                )
            )
            parts.append(text)

        if subject_gap_lines:
            text = "\n".join(f"- {line}" for line in subject_gap_lines)
            blocks.append(AnswerBlock(kind=AnswerBlockKind.CONCLUSION, text=text))
            parts.append(text)

        if shown_units:
            for unit in shown_units:
                blocks.append(
                    AnswerBlock(
                        kind=AnswerBlockKind.QUOTE,
                        text=render_quote_line(unit),
                        unit_ids=[unit.unit_id],
                        page_label=_page_label(unit),
                    )
                )
            parts.append(
                "\n".join(render_quote_line(unit) for unit in shown_units)
            )

        if (
            status is AnswerStatus.PARTIAL_GROUNDED
            and reasons
            and not subject_gap_lines
        ):
            note = f"未成立的部分：{reasons}"
            blocks.append(AnswerBlock(kind=AnswerBlockKind.NOTE, text=note))
            parts.append(note)

        if verification_note:
            parts.append(verification_note)
        return "\n\n".join(parts), blocks

    if status is AnswerStatus.SOURCE_ONLY:
        # 仅展示原文可定位的单元：没有原文可核对的单元不进入用户可见文本。
        # 相关性选择在这里同样生效——「未经确定性结构化结论」的原文列表
        # 也是用户读到的主要文字，里面出现别的病证的忌用一样会被当成答案。
        citable = [unit for unit in all_units if unit.is_citable()]
        shown_here = [
            unit
            for unit in citable
            if relevance.get(unit.unit_id, Relevance.DIRECT)
            is not Relevance.MISMATCH
        ]
        body = render_source_only(
            shown_here,
            reasons or "证据结构未能确定",
            diverted=len(citable) - len(shown_here),
        )
    else:
        body = render_abstain(reasons or "未检索到可用证据")
    text = _lead_with_decision(decision.text, body)
    return text, [AnswerBlock(kind=AnswerBlockKind.TEXT, text=text)]


def _card_payload(card: EvidenceCard) -> dict:
    """证据卡的 JSON 形态。`evidence_units` 与 `other_conditions` 共用同一形状，
    前端因此可以用同一个渲染函数处理两处，不必各自解析一遍。"""
    return {
        "unit_id": card.unit_id,
        "risk_category": card.risk_category.value,
        "category_label": card.category_label,
        "predicate": card.predicate,
        "subject_name": card.subject_name,
        "subject_id": card.subject_id,
        "object_id": card.object_id,
        "object_name": card.object_name,
        "conditions": card.conditions,
        "severity": card.severity.value,
        "severity_label": card.severity_label,
        "polarity_verdict": card.polarity_verdict.value,
        "polarity_basis": card.polarity_basis,
        "polarity_ambiguous": card.polarity_ambiguous,
        "dosage_raw": card.dosage_raw,
        "page_label": card.page_label,
        "page_start": card.page_start,
        "page_end": card.page_end,
        "evidence_quote": card.evidence_quote,
        "context": card.context,
        "source_kind": card.source_kind.value,
        "assertion_id": card.assertion_id,
        "chunk_ids": list(card.chunk_ids),
        "validation_status": card.validation_status,
        "is_clearly_parsed": card.is_clearly_parsed,
        "conflicts": list(card.conflicts),
        "undetermined_reason": card.undetermined_reason,
        "is_citable": card.is_citable,
        #: 已定稿的说明整句（含「原文说明」「证据存在差异」「当前证据不足以确定」
        #: 三类措辞）。前端直接显示，不再自己对 `conflicts` 套模板。
        "flags": list(card.flags),
    }


def pack_answer_payload(answer: HighRiskAnswer, **extra) -> dict:
    """Convert a HighRiskAnswer into JSON-ready fields for the API response.

    Args:
        answer: 高风险回答。
        **extra: 追加到载荷顶层的字段（例如 method、intent）。

    Returns:
        可直接交给 `jsonify` 的字典。枚举按值序列化。
        既有响应字段由调用方保留，本函数只负责**新增**字段，
        因此不会破坏现成前端契约。

    Side effects: 无。
    """
    payload = {
        "answer_status": answer.status.value,
        "abstained": answer.abstained,
        "decision": answer.decision,
        "decision_text": answer.decision_text,
        "verification_note": answer.verification_note,
        "answer_blocks": [
            {
                "kind": block.kind.value,
                "text": block.text,
                "unit_ids": list(block.unit_ids),
                "page_label": block.page_label,
            }
            for block in answer.answer_blocks
        ],
        # 与本题指定条件**不同**的证据。前端放在默认折叠的「其他使用注意」里，
        # 不参与主回答、claims 与 sources。
        "other_conditions": [
            _card_payload(card) for card in answer.other_conditions
        ],
        "claims": [
            {
                "claim_id": claim.claim_id,
                "kind": claim.kind.value,
                "statement": claim.statement,
                "subject_name": claim.subject_name,
                "object_name": claim.object_name,
                "conditions": claim.conditions,
                "unit_ids": list(claim.unit_ids),
                "pages": list(claim.pages),
            }
            for claim in answer.claims
        ],
        "evidence_units": [
            _card_payload(card) for card in answer.evidence_units
        ],
        "sub_questions": [
            {
                "question": item.question,
                "risk_category": item.risk_category.value if item.risk_category else None,
                "status": item.status.value,
                "reason": item.reason,
                "unit_ids": list(item.unit_ids),
                "claim_ids": list(item.claim_ids),
            }
            for item in answer.sub_questions
        ],
        "rejections": list(answer.rejections),
        "coverage_notes": list(answer.coverage_notes),
        "high_risk": {
            "is_high_risk": answer.is_high_risk,
            "status": answer.status.value,
            "predicates": list(PREDICATE_CATEGORY),
        },
    }
    payload.update(extra)
    return payload
