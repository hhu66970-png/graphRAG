"""Evidence localisation with character spans.

`evidence_quote in source_text` 对 OCR 文本过于脆弱：OCR 会引入空格、换行、
全半角与标点差异。本模块按四级顺序定位证据，并**始终返回原文字符跨度**：

    1. raw_span         —— LLM 给出的跨度经原文校验通过
    2. raw_exact        —— 原文精确子串
    3. normalized_exact —— 规范化后精确子串，再映射回原文跨度
    4. fuzzy            —— 极小编辑距离对齐（受 max_edit 限制）
    否则 not_found      —— 只能 REVIEW，不得正式入图

规范化只用于匹配，**绝不覆盖原始证据文本**。
"""

from __future__ import annotations

from difflib import SequenceMatcher

from ..domain.ids import _PUNCT_MAP
from ..domain.models import EvidenceSpan


def normalize_with_map(raw: str) -> tuple[str, list[int]]:
    """Normalize text while keeping a normalized-index -> raw-index map.

    丢弃空白字符、统一全角标点、小写化。返回 (规范化文本, 每个字符的原文下标)。
    """
    chars: list[str] = []
    positions: list[int] = []
    for index, ch in enumerate(raw):
        mapped = ch.translate(_PUNCT_MAP)
        if mapped.isspace():
            continue
        chars.append(mapped.lower())
        positions.append(index)
    return "".join(chars), positions


def bounded_edit_distance(left: str, right: str, max_distance: int) -> int:
    """Levenshtein distance, early-exit once it exceeds max_distance.

    返回值 > max_distance 表示「超出阈值」，不保证是精确距离。
    复杂度 O(len(left) * len(right))，本场景 quote 很短，可接受。
    """
    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, lch in enumerate(left, start=1):
        current = [i]
        row_best = i
        for j, rch in enumerate(right, start=1):
            cost = 0 if lch == rch else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            row_best = min(row_best, value)
        if row_best > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def _span_from_normalized(positions: list[int], start: int, end: int) -> tuple[int, int]:
    """Map a normalized [start, end) span back to a raw [start, end) span."""
    raw_start = positions[start]
    raw_end = positions[end - 1] + 1
    return raw_start, raw_end


def locate_evidence(
    quote: str,
    chunk_text: str,
    *,
    hint_start: int | None = None,
    hint_end: int | None = None,
    max_edit_distance: int = 2,
    min_quote_len: int = 2,
) -> EvidenceSpan:
    """Locate `quote` inside `chunk_text`, returning a span and match mode.

    Args:
        quote: LLM 返回的逐字证据。
        chunk_text: **原始** chunk 文本，跨度以它为准。
        hint_start/hint_end: LLM 自报的跨度，先行校验。
        max_edit_distance: 模糊对齐允许的最大编辑距离。
        min_quote_len: 过短的证据不予采信（避免「甘」这类单字误配）。

    Returns:
        EvidenceSpan；`match_mode == "not_found"` 表示无法定位。

    Side effects: 无。
    """
    cleaned = (quote or "").strip()
    if len(cleaned) < min_quote_len or not chunk_text:
        return EvidenceSpan(evidence_quote=cleaned, match_mode="not_found")

    # 1) LLM 自报跨度：以原文切片是否与 quote 规范化一致为准
    if hint_start is not None and hint_end is not None:
        if 0 <= hint_start < hint_end <= len(chunk_text):
            sliced = chunk_text[hint_start:hint_end]
            if normalize_with_map(sliced)[0] == normalize_with_map(cleaned)[0]:
                return EvidenceSpan(
                    evidence_quote=sliced,
                    evidence_start=hint_start,
                    evidence_end=hint_end,
                    match_mode="raw_span",
                )

    # 2) 原文精确子串
    position = chunk_text.find(cleaned)
    if position >= 0:
        return EvidenceSpan(
            evidence_quote=cleaned,
            evidence_start=position,
            evidence_end=position + len(cleaned),
            match_mode="raw_exact",
        )

    norm_chunk, positions = normalize_with_map(chunk_text)
    norm_quote, _ = normalize_with_map(cleaned)
    if not norm_quote or not norm_chunk:
        return EvidenceSpan(evidence_quote=cleaned, match_mode="not_found")

    # 3) 规范化精确子串，再映射回原文跨度
    position = norm_chunk.find(norm_quote)
    if position >= 0:
        raw_start, raw_end = _span_from_normalized(
            positions, position, position + len(norm_quote)
        )
        return EvidenceSpan(
            evidence_quote=chunk_text[raw_start:raw_end],
            evidence_start=raw_start,
            evidence_end=raw_end,
            match_mode="normalized_exact",
        )

    # 4) 模糊对齐：先用 SequenceMatcher 找最相似区域，再核验编辑距离
    matcher = SequenceMatcher(None, norm_chunk, norm_quote, autojunk=False)
    block = matcher.find_longest_match(0, len(norm_chunk), 0, len(norm_quote))
    if block.size == 0:
        return EvidenceSpan(evidence_quote=cleaned, match_mode="not_found")

    window_start = max(0, block.a - block.b - max_edit_distance)
    best: tuple[int, int, int] | None = None  # (distance, start, end)
    for start in range(window_start, min(len(norm_chunk), window_start + 2 * max_edit_distance + 1)):
        for length in range(
            max(1, len(norm_quote) - max_edit_distance),
            min(len(norm_chunk) - start, len(norm_quote) + max_edit_distance) + 1,
        ):
            candidate = norm_chunk[start : start + length]
            distance = bounded_edit_distance(candidate, norm_quote, max_edit_distance)
            if distance <= max_edit_distance and (best is None or distance < best[0]):
                best = (distance, start, start + length)
                if distance == 0:
                    break
        if best and best[0] == 0:
            break

    if best is None:
        return EvidenceSpan(evidence_quote=cleaned, match_mode="not_found")

    distance, norm_start, norm_end = best
    raw_start, raw_end = _span_from_normalized(positions, norm_start, norm_end)
    return EvidenceSpan(
        evidence_quote=chunk_text[raw_start:raw_end],
        evidence_start=raw_start,
        evidence_end=raw_end,
        match_mode="fuzzy",
        edit_distance=distance,
    )
