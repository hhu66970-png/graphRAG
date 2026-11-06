"""I2 关系校验：规则检查与模型复核。

非法关系直接拒绝；高分非高风险关系直接接受，llm_verdict 保持 NOT_RUN。
灰区和高风险谓词进入模型复核。final_score 用于排序与决策，不是事实正确概率。
"""

from __future__ import annotations

import re
from typing import Callable

from ..config.settings import KGSettings
from ..domain.errors import DomainRangeViolation, UnknownPredicateError
from ..domain.models import DosageValue, EvidenceSpan, ValidationResult
from ..domain.ontology import Ontology
from .dosage import is_well_formed

# 否定线索词。中药教材里「不宜」「忌」「禁」「勿」是强否定信号。
_NEGATION_CUES = ("不宜", "不可", "不能", "忌", "禁", "勿", "非", "无效", "不得")
_NEGATION_WINDOW = 12

# 证据定位模式 -> 规则加权。raw_span/raw_exact 最可信，fuzzy 明显打折。
_MATCH_MODE_WEIGHT = {
    "raw_span": 1.00,
    "raw_exact": 1.00,
    "normalized_exact": 0.90,
    "fuzzy": 0.65,
    "not_found": 0.0,
}


def _has_negation_near(evidence: str, object_name: str) -> bool:
    """Detect a negation cue close to the object mention inside the evidence."""
    if not evidence:
        return False
    position = evidence.find(object_name) if object_name else -1
    window = (
        evidence[max(0, position - _NEGATION_WINDOW) : position + len(object_name)]
        if position >= 0
        else evidence
    )
    return any(cue in window for cue in _NEGATION_CUES)


def _looks_like_ocr_noise(name: str) -> bool:
    """Reject obviously broken entity surfaces produced by OCR."""
    if not name or len(name) > 40:
        return True
    # 纯标点/数字、或含替换字符
    if re.fullmatch(r"[\W\d_]+", name, flags=re.UNICODE):
        return True
    return "�" in name


def validate_relation(
    *,
    ontology: Ontology,
    settings: KGSettings,
    predicate: str,
    subject_type: str,
    object_type: str,
    subject_name: str,
    object_name: str,
    evidence: EvidenceSpan,
    polarity: str = "POSITIVE",
    dosage: DosageValue | None = None,
    llm_verifier: Callable[[], str] | None = None,
) -> ValidationResult:
    """Validate one relation candidate.

    Args:
        ontology: 受控本体，用于 domain/range 校验。
        settings: 阈值来源；禁止在此文件内联魔法常量。
        predicate: 稳定谓词 ID（已由调用方解析）。
        subject_type/object_type: 稳定类型 ID。
        subject_name/object_name: 展示名，仅用于噪声与否定检查。
        evidence: 证据定位结果。
        polarity: 抽取器判定的极性。
        dosage: 当 predicate 为高风险剂量类时的值对象。
        llm_verifier: 仅在灰区被调用；返回 SUPPORTED/UNSUPPORTED。
            为 None 或开关关闭时不调用，verdict 保持 NOT_RUN。

    Returns:
        ValidationResult，字段语义互不重叠。

    Side effects: 可能通过 `llm_verifier` 触发一次 LLM 调用（仅灰区）。
    """
    flags: list[str] = []

    # ---- 硬失败：直接 REJECTED，不浪费 LLM 调用 -------------------------
    try:
        ontology.check_domain_range(predicate, subject_type, object_type)
    except (DomainRangeViolation, UnknownPredicateError) as exc:
        return ValidationResult(
            rule_score=0.0,
            rule_flags=[f"domain_range_violation:{exc}"],
            review_required=False,
            llm_verdict="NOT_RUN",
            final_score=0.0,
            status="REJECTED",
        )

    if subject_name == object_name:
        return ValidationResult(
            rule_score=0.0, rule_flags=["self_loop"], llm_verdict="NOT_RUN",
            final_score=0.0, status="REJECTED",
        )

    for label, name in (("subject", subject_name), ("object", object_name)):
        if _looks_like_ocr_noise(name):
            return ValidationResult(
                rule_score=0.0, rule_flags=[f"ocr_noise_{label}"], llm_verdict="NOT_RUN",
                final_score=0.0, status="REJECTED",
            )

    # 证据不可定位 -> 绝不能 ACCEPTED
    if not evidence.is_locatable:
        return ValidationResult(
            rule_score=0.0, rule_flags=["evidence_not_locatable"], review_required=True,
            llm_verdict="NOT_RUN", final_score=0.0, status="REVIEW",
        )

    # ---- 规则打分 -------------------------------------------------------
    score = _MATCH_MODE_WEIGHT[evidence.match_mode]
    if evidence.match_mode == "fuzzy":
        flags.append(f"fuzzy_edit_distance={evidence.edit_distance}")

    # 宾语是否真的出现在证据里：图谱里最常见的错误是「证据在讲 A，边却连到 B」
    if object_name and object_name not in evidence.evidence_quote:
        score -= 0.25
        flags.append("object_not_in_evidence")

    # 否定一致性
    negated = _has_negation_near(evidence.evidence_quote, object_name)
    if negated and polarity == "POSITIVE":
        score -= 0.45
        flags.append("negation_polarity_mismatch")
    elif negated and polarity == "NEGATIVE":
        score += 0.05
        flags.append("negation_consistent")

    is_high_risk = ontology.is_high_risk(predicate)
    if is_high_risk:
        flags.append("high_risk_predicate")
        # 高风险谓词要求更强证据：模糊对齐一律不能自动通过
        if evidence.match_mode == "fuzzy":
            score -= 0.20
            flags.append("high_risk_requires_exact_evidence")

    # 剂量结构完整性
    if dosage is not None:
        if is_well_formed(dosage):
            score += 0.05
            flags.append("dosage_well_formed")
        else:
            score -= 0.30
            flags.append("dosage_missing_value_or_unit")

    rule_score = max(0.0, min(1.0, score))

    # ---- 分流 -----------------------------------------------------------
    if rule_score < settings.rule_reject_below:
        return ValidationResult(
            rule_score=rule_score, rule_flags=flags, review_required=False,
            llm_verdict="NOT_RUN", final_score=rule_score, status="REJECTED",
        )

    if rule_score >= settings.rule_accept_above and not is_high_risk:
        # 明确支持：不调用 LLM，verdict 保持 NOT_RUN，语义不被污染
        return ValidationResult(
            rule_score=rule_score, rule_flags=flags, review_required=False,
            llm_verdict="NOT_RUN", final_score=rule_score, status="ACCEPTED",
        )

    # 灰区（含全部高风险谓词）：调用二次校验
    if not settings.use_llm_verifier or llm_verifier is None:
        return ValidationResult(
            rule_score=rule_score, rule_flags=[*flags, "verifier_unavailable"],
            review_required=True, llm_verdict="NOT_RUN",
            final_score=rule_score, status="REVIEW",
        )

    verdict = llm_verifier()
    if verdict == "SUPPORTED":
        final = min(1.0, 0.7 * rule_score + 0.3)
        status = "ACCEPTED" if final >= settings.final_accept_threshold else "REVIEW"
    elif verdict == "UNSUPPORTED":
        final = 0.3 * rule_score
        status = "REJECTED"
    else:
        final = rule_score
        status = "REVIEW"

    return ValidationResult(
        rule_score=rule_score,
        rule_flags=flags,
        review_required=status == "REVIEW",
        llm_verdict=verdict,  # type: ignore[arg-type]
        final_score=round(final, 4),
        status=status,
    )
