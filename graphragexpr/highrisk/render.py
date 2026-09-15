"""高风险最终文字的程序化渲染（任务书 §6）。

为什么不让模型写最终文字
------------------------
核验只能证明「这条候选结论与证据单元一致」，无法保证模型在最终自然语言里
逐字复现该结论。高风险场景里，一次改写就足以把「忌用」写成「禁用」、
把「煎服 3～9g」写成「常用 3～9g」。因此最终文字由本模块按模板从
**已核验的证据单元**生成：

- 数值、单位、对象、限制程度、用法、生制品、人群、页码全部取自单元字段；
- 模型原话只作为核验输入，不进入最终答案；
- 模板分支是封闭的（`if/elif` 列举），不做任何字符串拼凑式润色。

因此最终文字里出现的每一个高风险要素，都能在 `evidence_units` 里找到同源字段。
"""

from __future__ import annotations

import re

from ..domain.high_risk import (
    AnswerStatus,
    ConditionalEvidenceUnit,
    PolarityVerdict,
    RestrictionSeverity,
    RiskCategory,
    SEVERITY_DISPLAY,
    SourceKind,
)
from .models import EvidenceCard
from .parsing import ALL_CONDITION_CUES
from .units import SEVERITY_UNSPECIFIED_NOTE

#: 弃权与降级时必须一并说明的替代路径说明。禁止把「没查到」写成「可以」。
#:
#: 措辞刻意避开「可以同用」「无禁忌」这类字面串：即使上下文是在**否定**它们
#: （「不等于可以同用」），字面出现仍会让下游的自动安全性扫描误报，
#: 也会让只读到半句的人产生反向理解。
NO_EVIDENCE_NOTE = (
    "未检索到支持该问题的证据。没有证据不等于该用法是安全的，"
    "也不能据此认为相关禁忌并不存在，更不能反过来当作没有限制的依据；"
    "请以教材原文与专业人员意见为准。"
)

_SOURCE_KIND_LABEL: dict[SourceKind, str] = {
    SourceKind.ASSERTION: "已校验断言",
    SourceKind.TEXT_CHUNK: "教材原文（未经断言校验）",
}


#: 原文依据的统一标签（任务书第三部分 §4）。前端用同样字面量渲染证据卡原文，
#: 使「结论」与「原文」在视觉上一眼可分：结论是程序对证据的复述，
#: 引号里才是教材逐字原文。
QUOTE_LABEL = "原文依据："


def _page_text(unit: ConditionalEvidenceUnit) -> str:
    """把页码格式化成「第 x 页」。"""
    if not unit.page_available or unit.page_start is None:
        return "页码不可用"
    if unit.page_end in (None, unit.page_start):
        return f"第 {unit.page_start} 页"
    return f"第 {unit.page_start}-{unit.page_end} 页"


def _quote_line(unit: ConditionalEvidenceUnit) -> str:
    """渲染一条带引用的证据行。"""
    quote = " ".join(unit.evidence_quote.split())
    return f"{QUOTE_LABEL}「{quote}」（{_page_text(unit)}）"


def _severity_clause(unit: ConditionalEvidenceUnit) -> str:
    """渲染程度用语（禁用/慎用）。"""
    if unit.severity is RestrictionSeverity.UNSPECIFIED:
        # 教材未给程度词时如实说明，不用「慎用」等近似词替代。
        return "原文未标注禁用／忌用／慎用程度"
    return f"限制程度：{SEVERITY_DISPLAY[unit.severity]}"


#: 条件槽位的中文标签。**标签本身不得包含任何解析线索词**：
#: 结构行会被核验程序重新解析，标签若含线索就会凭空多出一个条件。
#: 例如用「生制品：」作标签时，「生制品」里的「制品」会被读成生制品线索，
#: 把「生品」的数值配到「制品」上——正是本任务要拦截的那类绑定错误。
_LABEL_PREPARATION = "炮制"
_LABEL_ROUTE = "用法"
_LABEL_DOSAGE = "剂量"
_LABEL_FREQUENCY = "频次"
_LABEL_POPULATION = "人群"
_LABEL_SEVERITY = "程度"
_LABEL_RAW_CONDITION = "原文条件"

#: 判断条件窗口是否还有「未被结构化槽位吸收」的内容时，需要剔除的字符：
#: 标点、空白与数字（数字由剂量槽位负责）。
_RESIDUE_STRIP_RE = re.compile(r"[，。；、：（）()「」\s0-9.~～\-—－]+")


def _unstructured_condition(unit: ConditionalEvidenceUnit) -> str | None:
    """Return the local condition window when it carries more than the parsed slots.

    为什么必须暴露这段原文：教材里同一味药的不同剂量往往靠**非结构化条件**区分，
    例如「煎服，3～9g；挽救虚脱可用 15～30g」——「挽救虚脱可用」不在任何
    已知线索词表里。若渲染时只保留数值，15～30g 就会脱离它的前提，
    看起来像常规用量，这是直接的安全隐患。因此只要窗口里还有线索词表
    未覆盖的实质内容，就把整段窗口原样带出。

    判定「还有实质内容」时要剔除**整份线索词表**，而不只是本单元绑定的那几个：
    窗口是「上一处数值之后到下一处数值之前」，可能顺带包含邻近数值的条件词
    （「生品 1.5～3g，制品」里的「制品」属后一个数值）。只剔除绑定词会让
    这段邻接文字被误当成未结构化内容，把另一个分支的炮制方式显示到本单元上。
    """
    if unit.dosage is None or not unit.dosage.condition_text:
        return None
    residue = unit.dosage.condition_text
    for cue in ALL_CONDITION_CUES:
        if cue in residue:
            residue = residue.replace(cue, "", 1)
    if unit.dosage.raw:
        residue = residue.replace(unit.dosage.raw, "", 1)
    if len(_RESIDUE_STRIP_RE.sub("", residue)) < 2:
        return None
    return " ".join(unit.dosage.condition_text.split())


def _condition_parts(
    unit: ConditionalEvidenceUnit,
    *,
    include_dosage: bool = True,
    include_severity: bool = True,
) -> list[str]:
    """条件槽位，逐项来自单元字段，教材未说明的槽位直接缺席（不补默认值）。"""
    parts: list[str] = []
    if unit.preparation:
        parts.append(f"{_LABEL_PREPARATION}：{unit.preparation}")
    if unit.route:
        parts.append(f"{_LABEL_ROUTE}：{unit.route}")
    if include_dosage and unit.dosage and unit.dosage.raw:
        parts.append(f"{_LABEL_DOSAGE}：{unit.dosage.raw}")
    if unit.frequency:
        parts.append(f"{_LABEL_FREQUENCY}：{unit.frequency}")
    if unit.population:
        parts.append(f"{_LABEL_POPULATION}：{unit.population}")
    if include_severity and unit.severity is not RestrictionSeverity.UNSPECIFIED:
        parts.append(f"{_LABEL_SEVERITY}：{SEVERITY_DISPLAY[unit.severity]}")
    raw_condition = _unstructured_condition(unit)
    if raw_condition:
        parts.append(f"{_LABEL_RAW_CONDITION}：{raw_condition}")
    parts.extend(unit.other_restrictions)
    return parts


def _dosage_line(unit: ConditionalEvidenceUnit) -> str:
    """渲染一条剂量表述。"""
    head = f"{unit.subject_name}：用量{unit.dosage.raw if unit.dosage else ''}"
    if unit.dosage and not unit.dosage.unit_raw:
        head += "（原文此处未紧邻单位，未做补全）"
    # 剂量本身已经写在行首，条件里不再重复，避免同一数值在一行内出现两次。
    conditions = _condition_parts(unit, include_dosage=False)
    return f"{head}（{'；'.join(conditions)}）" if conditions else head


def _restriction_line(unit: ConditionalEvidenceUnit) -> str:
    """渲染一条使用限制。"""
    target = unit.object_name or "（未识别到明确对象）"
    category_label = (
        "使用禁忌" if unit.risk_category is RiskCategory.CONTRAINDICATION else "配伍禁忌"
    )
    if unit.polarity_verdict is PolarityVerdict.RELATION_DOES_NOT_HOLD:
        # 「无禁忌」是关系被否定的情形，必须单独表述，不能套用禁忌模板。
        head = f"{unit.subject_name} 与 {target}：证据表明该{category_label}关系不成立"
    elif unit.polarity_verdict is PolarityVerdict.UNDETERMINED:
        head = (
            f"{unit.subject_name} 与 {target}：极性无法确定，"
            f"不能据此判定是否属于{category_label}"
        )
    else:
        head = (
            f"{unit.subject_name} 与 {target}：{category_label}"
            f"（{_severity_clause(unit)}）"
        )
    conditions = _condition_parts(unit, include_severity=False)
    return f"{head}；{'；'.join(conditions)}" if conditions else head


def render_unit_statement(unit: ConditionalEvidenceUnit) -> str:
    """结构行：**只含可由单元字段逐项核对的要素**，不含原文引用。

    这条行会被核验程序重新解析，用来确认渲染没有把数值与条件挪位或漏掉。
    刻意不拼接 `evidence_quote`：原文里可能同时出现别的剂量或别的程度词，
    把原文混进待核验文本会让核验去检查它无权判断的内容。
    """
    body = (
        _dosage_line(unit)
        if unit.risk_category is RiskCategory.DOSAGE
        else _restriction_line(unit)
    )
    return f"{body}。"


def render_unit(unit: ConditionalEvidenceUnit) -> str:
    """展示行：结构行 + 逐字原文与页码，供人复核。

    一条单元一行：数值与它的条件始终同行，遵循「一个单元 = 一个可独立成立的事实」。
    """
    return f"{render_unit_statement(unit)}{_quote_line(unit)}"


#: 解析器内部告警码，形如 `code:面向维护者的说明`（`parsing.py` 产出）。
#: 展示层只**去掉码**，不改写正文——正文里的数值与条件是解析结果的一部分。
_PARSE_CODE_RE = re.compile(r"^[a-z][a-z_]*:")

#: 已由下面「当前证据不足以确定」专门行讲过的解析码，**不再重复**。
#: `inverted_range` 对应 `dosage.is_inverted`，`unit_not_adjacent` 对应
#: `not dosage.is_bound_to_unit()`；两者都必然带出 `is_clearly_parsed=False`
#: 与 `undetermined_reason`。重复显示会把同一件事说两遍，还会把
#: `inverted_range:` 这类内部码暴露给用户。
_COVERED_PARSE_CODES: frozenset[str] = frozenset(
    {"inverted_range", "unit_not_adjacent"}
)


def _parse_code(note: str) -> str | None:
    """取出解析告警码；不是告警码则返回 None。"""
    match = _PARSE_CODE_RE.match(note)
    return match.group(0)[:-1] if match else None


def _display_note(note: str) -> str:
    """渲染给前端看的补充说明。"""
    return _PARSE_CODE_RE.sub("", note).strip()


def _conflict_lines(
    conflicts: list[str], *, covered: set[str | None], parse_notes: set[str]
) -> list[str]:
    """把 `conflicts` 逐条分流成面向读者的一句话。

    分流依据是**结构化来源**，不是中文正文匹配：
    - 已被 `undetermined_reason` / `polarity_basis` 等专门行覆盖的条目不再重复；
    - 只由解析器产出的条目（带告警码，或同一单元里出现在 `dosage.parse_notes`
      中）是解析层**说明**，用「原文说明：」；
    - 其余（存量解析值与按原文重解析不一致等）才是真正的**差异**，
      如实并列、不静默采用其一。
    """
    lines: list[str] = []
    for conflict in conflicts:
        if conflict in covered:
            continue
        if _parse_code(conflict) in _COVERED_PARSE_CODES:
            continue
        if conflict in parse_notes or _parse_code(conflict) is not None:
            lines.append(f"原文说明：{_display_note(conflict)}")
        elif conflict == SEVERITY_UNSPECIFIED_NOTE:
            lines.append(f"当前证据不足以确定：{conflict}。系统暂不作确定结论。")
        else:
            lines.append(f"证据存在差异：{conflict}。系统暂不作单一结论。")
    return lines


def unresolved_flags(unit: ConditionalEvidenceUnit) -> list[str]:
    """尚未解决的问题，写成人话（任务书第三部分 §6）。

    措辞要点：**普通用户不执行人工核对**，因此不再使用「待核对」「需人工复核」
    这类把工作推给读者的说法；每一条都直接说明「这是哪一类不确定」以及
    「系统因此不作什么结论」。把问题分流给维护人员的提示只出现在
    `rejections` / `coverage_notes` 这类调试通道，不出现在答案正文里。

    为什么不能对 `unit.conflicts` 一律用「证据存在差异」：那个字段里既有真的
    分歧（存量值与原文不一致），也有解析层**说明**——`merged_clause` 说的正是
    「后续小句不是另一处独立剂量」。套上差异文案后，一条已核验的剂量会对自己
    的数值声称有分歧、并宣称「系统暂不作单一结论」（实测附子用量卡即如此），
    与同一张卡的徽标「高风险·已核验」互相矛盾。
    """
    notes: list[str] = []
    if unit.polarity_ambiguous:
        notes.append(
            f"证据含义存在歧义：{unit.polarity_basis}。系统暂不作确定结论。"
        )
    # 「为什么这条不能给出确定值」只讲一次：倒置范围与数值／单位未绑定都会带出
    # `undetermined_reason`，那一行已经是权威说法，此处不再另起一句。
    reason_line = (
        f"当前证据不足以确定：{unit.undetermined_reason}"
        if not unit.is_clearly_parsed and unit.undetermined_reason
        else None
    )
    if unit.dosage and unit.dosage.is_inverted and reason_line is None:
        notes.append(
            "当前证据不足以确定：原文上界小于下界，未做自动交换。系统暂不作确定结论。"
        )
    if reason_line is not None:
        notes.append(reason_line)
    if unit.is_cross_chunk:
        notes.append("该证据跨文本块，引用时需回看完整上下文")
    notes.extend(
        _conflict_lines(
            unit.conflicts,
            #: 只把**真的显示过**的说明算作已覆盖，避免同一件事一句话都不说
            covered={
                unit.polarity_basis if unit.polarity_ambiguous else None,
                unit.undetermined_reason if reason_line is not None else None,
            },
            parse_notes=set(unit.dosage.parse_notes) if unit.dosage else set(),
        )
    )
    return list(dict.fromkeys(notes))


def card_flags(card: EvidenceCard) -> list[str]:
    """卡片上的「未解决说明」。

    单元在组装时已经把说明算好（`EvidenceCard.flags`，见 `models.py`），这里
    优先用它；只有手工构造、没有该字段的老卡片才按卡上字段重算。重算时拿不到
    `dosage.parse_notes`，因此只按告警码分流。
    """
    if card.flags:
        return list(card.flags)
    notes: list[str] = []
    if card.polarity_ambiguous:
        notes.append(
            f"证据含义存在歧义：{card.polarity_basis}。系统暂不作确定结论。"
        )
    reason_printed = bool(not card.is_clearly_parsed and card.undetermined_reason)
    if reason_printed:
        notes.append(f"当前证据不足以确定：{card.undetermined_reason}")
    notes.extend(
        _conflict_lines(
            card.conflicts,
            covered={
                card.polarity_basis if card.polarity_ambiguous else None,
                card.undetermined_reason if reason_printed else None,
            },
            parse_notes=set(),
        )
    )
    return list(dict.fromkeys(notes))


def render_evidence_card(card: EvidenceCard) -> str:
    """Render a user-facing evidence block, including unresolved caveats."""
    lines = [
        f"【{card.category_label}】{card.subject_name}"
        + (f" → {card.object_name}" if card.object_name else ""),
        f"条件：{card.conditions or '原文未给出附加条件'}；程度：{card.severity_label}",
        f"来源类型：{_SOURCE_KIND_LABEL[card.source_kind]}",
        f"{QUOTE_LABEL}「{' '.join(card.evidence_quote.split())}」（{card.page_label}）",
    ]
    if card.context.strip() and card.context.strip() != card.evidence_quote.strip():
        lines.append(f"完整相关上下文：{' '.join(card.context.split())}")
    # 说明行与卡片同源：单元组装时算好（`EvidenceCard.flags`），此处只取用，
    # 不再各自套一遍模板——两处模板会漂移，也会把「说明」说成「差异」。
    lines.extend(card_flags(card))
    return "\n".join(lines)


def render_claims(units: list[ConditionalEvidenceUnit]) -> str:
    """Render the verified units as the body of the answer.

    只有**通过核验**的单元才会被传进来；本函数不接触模型文本。
    """
    return "\n".join(f"- {render_unit(unit)}" for unit in units)


def render_source_only(
    units: list[ConditionalEvidenceUnit], reason: str, *, diverted: int = 0
) -> str:
    """Template used when structure could not be confirmed but原文 is trustworthy.

    Args:
        units: 与本题条件一致、可定位的单元（已通过相关性选择）。
        reason: 未能形成确定结论的原因。
        diverted: 因**与本题指定条件不同**而未在此列出的单元数。必须写明条数：
            一条也不提会让人以为教材里什么都没有，而这与「有、但说的是别人」是两回事。
    """
    parts: list[str] = []
    if units:
        header = (
            "以下为与问题直接相关的教材原文与已抽取证据，未经确定性结构化结论："
            f"{reason}。"
        )
        blocks = []
        for unit in units:
            block = [f"- {render_unit(unit)}"]
            for note in unresolved_flags(unit):
                block.append(f"  {note}")
            blocks.append("\n".join(block))
        parts.append(header + "\n" + "\n".join(blocks))
    else:
        parts.append(f"没有与本题指定条件一致的可核对原文：{reason}。")
    if diverted:
        parts.append(
            f"教材中另有 {diverted} 条证据与本题指定的人群或条件不同，"
            "见下方「其他使用注意（与本题指定人群不同）」，它们不参与本题结论。"
        )
    parts.append(NO_EVIDENCE_NOTE)
    return "\n".join(parts)


def render_abstain(reason: str) -> str:
    """Template used when nothing can be said. Never phrased as safety."""
    return f"证据不足，无法就该高风险问题给出结论：{reason}。{NO_EVIDENCE_NOTE}"


# ------------------------------------------------- 程序核验说明（§3）

#: 全部结论与原文都对上时**位于答案末尾**的核验说明。
#:
#: 为什么从「开头」挪到「末尾」：说明写在同一段结论前面时，读者要先读一句
#: 与问题无关的元信息才看到答案；而核验说明的作用是**收尾担保**，紧跟被担保的
#: 内容才成立。措辞保持原样（「以上结论均通过程序核验…」），只换位置。
VERIFICATION_NOTE_GROUNDED = (
    "以上结论均通过程序核验，数值、单位、用法与限制程度与所选证据单元逐项一致。"
)

#: 只有部分结论成立时用。**不得**写成「均通过」——那会把未成立的部分
#: 一并担保进去。
VERIFICATION_NOTE_PARTIAL = (
    "以上已标记为有据的结论通过程序核验；其余部分因证据不足或存在差异，"
    "未形成确定结论。"
)


def verification_note_for(status: AnswerStatus) -> str:
    """该状态下是否显示通过核验的说明。

    `source_only` 与 `abstain` 下没有任何结论是通过核验的，
    此时一句「通过核验」的说明只会误导，因此返回空串表示不显示。
    """
    if status is AnswerStatus.GROUNDED:
        return VERIFICATION_NOTE_GROUNDED
    if status is AnswerStatus.PARTIAL_GROUNDED:
        return VERIFICATION_NOTE_PARTIAL
    return ""


# ------------------------------------------------- 分块渲染（§7）

def render_conclusion_line(unit: ConditionalEvidenceUnit) -> str:
    """结论块的一行：结构化结论 + 条件解释，**不含**原文引文。"""
    return render_unit_statement(unit)


def render_quote_line(unit: ConditionalEvidenceUnit) -> str:
    """原文依据块的一行：`原文依据：「…」（第 N 页）`。"""
    return _quote_line(unit)
