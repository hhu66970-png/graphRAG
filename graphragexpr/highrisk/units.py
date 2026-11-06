"""从断言与原文组装条件证据单元。

组装规则的核心是**一个单元 = 一个可独立成立的事实 + 它成立所依赖的全部条件**：

- 剂量断言会按 `parse_dosage_units` 展开成多个单元（同段两种用法 → 两个单元，
  各自绑定自己的用法与生制品），并**重新解析 `raw` 原文**而不是信任存量解析结果。
  这样既让历史数据自动获得保守解析语义，也不必迁移数据库。
- 存量解析值与重新解析值不一致时，不做静默取舍：并列记录为 conflict，
  由核验阶段决定是否降级。
- 仅有原文、没有断言的证据走 `TEXT_CHUNK` 通道，**不伪造** assertion_id 或页码。
"""

from __future__ import annotations

import hashlib

from ..domain.high_risk import (
    PREDICATE_CATEGORY,
    ConditionalEvidenceUnit,
    DosageBinding,
    FieldSpan,
    PolarityVerdict,
    RestrictionSeverity,
    RiskCategory,
    SourceKind,
)
from .parsing import (
    classify_severity,
    extract_population,
    extract_preparation,
    parse_dosage_units,
)
from .polarity import interpret_polarity


def _stable_unit_id(*parts: object) -> str:
    """单元标识：由来源与内容派生，同一输入重复组装得到同一标识。

    单元标识会出现在候选结论的引用里，因此必须稳定——不能用自增序号或
    运行期随机数，否则引用无法跨请求核对。
    """
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8"))
    return f"unit:{digest.hexdigest()[:16]}"


def _first(values: list, index: int):
    """取序列中第一个非空值。"""
    return values[index] if index < len(values) else None


def _pages(row: dict) -> tuple[int | None, int | None, bool]:
    """从来源块推断页码。缺页码与缺原文分开处理：没有页码就不补造。"""
    starts = [p for p in (row.get("chunk_page_starts") or []) if isinstance(p, int)]
    ends = [p for p in (row.get("chunk_page_ends") or []) if isinstance(p, int)]
    if not starts and not ends and row.get("page_start") is None:
        return None, None, False
    page_start = min(starts) if starts else row.get("page_start")
    page_end = max(ends) if ends else (row.get("page_end") or page_start)
    available = page_start is not None
    return page_start, page_end, available


def _context_of(row: dict) -> str:
    """完整相关上下文：优先取来源块原文，退化到证据句。

    结构化困难时上层要展示完整相关上下文，
    因此上下文必须与证据一起带出来，而不是到渲染时再去查库。
    """
    texts = [text for text in (row.get("chunk_texts") or []) if text]
    if texts:
        return "\n".join(texts)
    return row.get("evidence_quote") or ""


#: 「原文没写限制程度词」这条记录**不是两处来源打架**，而是「确定不了」。
#: 展示层要按「不足」而不是「差异」措辞，因此把它定义成具名常量：
#: 渲染层按**常量同一性**分流，不去猜中文正文。
SEVERITY_UNSPECIFIED_NOTE = "原文未给出禁用／忌用／慎用程度词"


def _dosage_conflicts(stored: dict, parsed: DosageBinding) -> list[str]:
    """比对存量解析值与重新解析值，只记录冲突，不擅自取舍。"""
    conflicts: list[str] = []
    stored_min = stored.get("value_min")
    stored_max = stored.get("value_max")
    stored_unit = stored.get("unit")
    if stored_min is not None and parsed.value_min is not None:
        if float(stored_min) != float(parsed.value_min):
            conflicts.append(
                f"存量解析下界({stored_min}) 与按原文重解析({parsed.value_min}) 不一致"
            )
    if stored_max is not None and parsed.value_max is not None:
        if float(stored_max) != float(parsed.value_max):
            conflicts.append(
                f"存量解析上界({stored_max}) 与按原文重解析({parsed.value_max}) 不一致"
            )
    if stored_unit and parsed.unit:
        if str(stored_unit) != parsed.unit:
            conflicts.append(
                f"存量单位({stored_unit}) 与原文单位({parsed.unit}) 不一致"
            )
    if stored_unit is None and parsed.unit is not None:
        conflicts.append(
            f"存量记录缺单位，原文紧邻单位为 {parsed.unit_raw or parsed.unit}"
        )
    return conflicts


def _dosage_units_from_assertion(row: dict) -> list[ConditionalEvidenceUnit]:
    """Expand one dosage assertion into one unit per bound dosage."""
    stored = dict(row.get("object_value") or {})
    raw_text = str(stored.get("raw") or row.get("evidence_quote") or "")
    subject_name = row.get("subject_name") or ""
    page_start, page_end, page_available = _pages(row)
    quote = row.get("evidence_quote") or ""
    context = _context_of(row)
    assertion_id = row.get("assertion_id")
    bindings = parse_dosage_units(raw_text)

    if not bindings:
        # 解析不出数值（如「适量外用」）：保留为不明确单元，
        # 让上层走原文展示或降级，而不是凭存量字符串作答。
        severity, severity_raw = classify_severity(quote)
        return [
            ConditionalEvidenceUnit(
                unit_id=_stable_unit_id(assertion_id, "HAS_DOSAGE", raw_text, "unparsed"),
                risk_category=RiskCategory.DOSAGE,
                predicate="HAS_DOSAGE",
                subject_id=row.get("subject_id"),
                subject_name=subject_name,
                object_value=stored or None,
                severity=severity,
                severity_raw=severity_raw,
                population=extract_population(raw_text) or extract_population(quote),
                assertion_id=assertion_id,
                fact_key=row.get("fact_key"),
                source_kind=SourceKind.ASSERTION,
                validation_status=row.get("validation_status"),
                is_cross_chunk=bool(row.get("is_cross_chunk")),
                raw_polarity=row.get("polarity"),
                polarity_verdict=PolarityVerdict.RELATION_HOLDS,
                polarity_basis="剂量为正向事实陈述",
                evidence_quote=quote,
                context=context,
                chunk_ids=list(row.get("chunk_ids") or []),
                page_start=page_start,
                page_end=page_end,
                page_available=page_available,
                is_clearly_parsed=False,
                undetermined_reason=(
                    f"剂量原文「{raw_text}」未能解析出可绑定的数值与单位"
                    if raw_text
                    else "该剂量断言没有可用的原文表述"
                ),
            )
        ]

    units: list[ConditionalEvidenceUnit] = []
    for index, binding in enumerate(bindings):
        # 解析器记录的告警（倒置范围、单位未紧邻）必须并入单元的 conflicts，
        # 否则这些信息只存在于解析结果内部，前端与覆盖度说明都看不到，
        # 单元会显得比实际更「干净」。
        conflicts = _dosage_conflicts(stored, binding) + list(binding.parse_notes)
        is_clear = binding.is_bound_to_unit() and not binding.is_inverted
        reason = None
        if binding.is_inverted:
            # 面向读者：说明**系统不据此给出剂量**，而不是把核对工作推给读者
            reason = "原文剂量上界小于下界，未做自动交换；系统不据此给出确定剂量"
        elif not binding.is_bound_to_unit():
            reason = "数值与单位未能绑定，不能据此给出确定剂量"
        units.append(
            ConditionalEvidenceUnit(
                unit_id=_stable_unit_id(
                    assertion_id, "HAS_DOSAGE", binding.raw, index
                ),
                risk_category=RiskCategory.DOSAGE,
                predicate="HAS_DOSAGE",
                subject_id=row.get("subject_id"),
                subject_name=subject_name,
                object_value=stored or None,
                preparation=binding.preparation,
                route=binding.route,
                dosage=binding,
                frequency=binding.frequency,
                population=binding.population or extract_population(quote),
                assertion_id=assertion_id,
                fact_key=row.get("fact_key"),
                source_kind=SourceKind.ASSERTION,
                validation_status=row.get("validation_status"),
                is_cross_chunk=bool(row.get("is_cross_chunk")),
                raw_polarity=row.get("polarity"),
                polarity_verdict=PolarityVerdict.RELATION_HOLDS,
                polarity_basis="剂量为正向事实陈述",
                evidence_quote=quote,
                context=context,
                chunk_ids=list(row.get("chunk_ids") or []),
                page_start=page_start,
                page_end=page_end,
                page_available=page_available,
                span=binding.field_spans.get("value"),
                field_spans=dict(binding.field_spans),
                is_clearly_parsed=is_clear,
                conflicts=conflicts,
                undetermined_reason=reason,
            )
        )
    return units


def _restriction_unit_from_assertion(row: dict) -> ConditionalEvidenceUnit:
    """Assemble one contraindication / incompatibility unit with polarity interpretation."""
    predicate = row.get("predicate") or ""
    category = PREDICATE_CATEGORY.get(predicate, RiskCategory.CONTRAINDICATION)
    quote = row.get("evidence_quote") or ""
    raw_polarity = row.get("polarity")
    object_name = row.get("object_name")
    interpretation = interpret_polarity(
        predicate=predicate,
        raw_polarity=raw_polarity,
        evidence_quote=quote,
        object_name=object_name,
    )
    severity, severity_raw = classify_severity(quote)
    page_start, page_end, page_available = _pages(row)
    conflicts: list[str] = []
    if interpretation.ambiguous:
        conflicts.append(interpretation.basis)
    if severity is RestrictionSeverity.UNSPECIFIED:
        conflicts.append(SEVERITY_UNSPECIFIED_NOTE)

    return ConditionalEvidenceUnit(
        unit_id=_stable_unit_id(
            row.get("assertion_id"), predicate, object_name, quote[:24]
        ),
        risk_category=category,
        predicate=predicate,
        subject_id=row.get("subject_id"),
        subject_name=row.get("subject_name") or "",
        object_id=row.get("object_id"),
        object_name=object_name,
        object_value=row.get("object_value") or None,
        preparation=extract_preparation(quote),
        population=extract_population(quote),
        severity=severity,
        severity_raw=severity_raw,
        assertion_id=row.get("assertion_id"),
        fact_key=row.get("fact_key"),
        source_kind=SourceKind.ASSERTION,
        validation_status=row.get("validation_status"),
        is_cross_chunk=bool(row.get("is_cross_chunk")),
        raw_polarity=raw_polarity,
        polarity_verdict=interpretation.verdict,
        polarity_basis=interpretation.basis,
        polarity_ambiguous=interpretation.ambiguous,
        evidence_quote=quote,
        context=_context_of(row),
        chunk_ids=list(row.get("chunk_ids") or []),
        page_start=page_start,
        page_end=page_end,
        page_available=page_available,
        is_clearly_parsed=not interpretation.ambiguous,
        conflicts=conflicts,
        undetermined_reason=(
            interpretation.basis if interpretation.ambiguous else None
        ),
    )


def build_units_from_assertions(rows: list[dict]) -> list[ConditionalEvidenceUnit]:
    """Turn high-risk assertion rows into conditional evidence units.

    Args:
        rows: `storage.high_risk_read.load_high_risk_assertions` 的返回值。

    Returns:
        条件证据单元列表。剂量断言会展开为多个单元；范围外的谓词被忽略
        （不静默扩大处理范围）。

    Side effects: 无（纯组装）。
    """
    units: list[ConditionalEvidenceUnit] = []
    for row in rows:
        predicate = row.get("predicate") or ""
        if predicate not in PREDICATE_CATEGORY:
            continue  # 此处只处理三类关系
        if predicate == "HAS_DOSAGE":
            units.extend(_dosage_units_from_assertion(row))
        else:
            units.append(_restriction_unit_from_assertion(row))
    return units


def build_units_from_chunks(
    chunks: list[dict],
    *,
    predicate: str = "CONTRAINDICATED_IN",
) -> list[ConditionalEvidenceUnit]:
    """Adapter: turn retrieved raw text into TEXT_CHUNK-sourced units.

    向量／全文检索返回的原文经此进入同一证据处理流程。
    这类单元的 `assertion_id` 恒为 None——原文证据**不得**被升级为通过断言，
    只记录稳定的文本块标识与位置标识。
    """
    category = PREDICATE_CATEGORY.get(predicate, RiskCategory.CONTRAINDICATION)
    units: list[ConditionalEvidenceUnit] = []
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
        text = str(chunk.get("text") or "").strip()
        if not chunk_id or not text:
            continue
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end") or page_start
        severity, severity_raw = classify_severity(text)
        units.append(
            ConditionalEvidenceUnit(
                unit_id=_stable_unit_id(chunk_id, "TEXT_CHUNK", predicate),
                risk_category=category,
                predicate=predicate,
                subject_name=str(chunk.get("subject_name") or ""),
                population=extract_population(text),
                severity=severity,
                severity_raw=severity_raw,
                assertion_id=None,           # 禁止伪造事实标识
                source_kind=SourceKind.TEXT_CHUNK,
                validation_status=None,
                evidence_quote=text,
                context=text,
                chunk_ids=[chunk_id],
                page_start=page_start if isinstance(page_start, int) else None,
                page_end=page_end if isinstance(page_end, int) else None,
                page_available=isinstance(page_start, int),
                span=FieldSpan(start=0, end=len(text), text=text),
                is_clearly_parsed=False,
                undetermined_reason="仅有教材原文，未经断言校验，不能单独支撑确定性结论",
            )
        )
    return units
