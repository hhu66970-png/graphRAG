"""证据定位与剂量值对象测试。

覆盖 要求的：正常路径、空输入、OCR 噪声、否定与数值、降级路径。
"""

from __future__ import annotations

import pytest

from graphragexpr.extract.dosage import is_well_formed, parse_dosage
from graphragexpr.extract.evidence import (
    bounded_edit_distance,
    locate_evidence,
    normalize_with_map,
)

CHUNK = (
    "人参　味甘、微苦，性微温。归脾、肺、心、肾经。\n"
    "大补元气，复脉固脱，补脾益肺，生津养血，安神益智。\n"
    "煎服，3～9g；挽救虚脱可用 15～30g。不宜与藜芦同用。"
)


class TestNormalizeWithMap:
    def test_positions_map_back_to_raw(self):
        normalized, positions = normalize_with_map("归脾、肺 经")
        assert "　" not in normalized and " " not in normalized
        assert len(normalized) == len(positions)
        # 每个规范化字符都能回到原文对应位置
        raw = "归脾、肺 经"
        for index, char in enumerate(normalized):
            assert raw[positions[index]].lower() == char or raw[positions[index]] in "、"

    def test_empty_input(self):
        assert normalize_with_map("") == ("", [])


class TestBoundedEditDistance:
    def test_identical(self):
        assert bounded_edit_distance("归脾经", "归脾经", 2) == 0

    def test_within_bound(self):
        assert bounded_edit_distance("归脾肺经", "归脾经", 2) == 1

    def test_exceeds_bound_returns_sentinel(self):
        assert bounded_edit_distance("完全不同的文本", "归脾经", 1) > 1


class TestLocateEvidence:
    def test_raw_exact(self):
        span = locate_evidence("大补元气，复脉固脱", CHUNK)
        assert span.match_mode == "raw_exact"
        assert CHUNK[span.evidence_start : span.evidence_end] == "大补元气，复脉固脱"

    def test_raw_span_hint_verified(self):
        quote = "归脾、肺、心、肾经"
        start = CHUNK.index(quote)
        span = locate_evidence(quote, CHUNK, hint_start=start, hint_end=start + len(quote))
        assert span.match_mode == "raw_span"
        assert span.evidence_start == start

    def test_bad_hint_falls_through_to_exact(self):
        """LLM 自报跨度错误时必须降级重定位，而不是直接采信。"""
        quote = "大补元气"
        span = locate_evidence(quote, CHUNK, hint_start=0, hint_end=4)
        assert span.match_mode == "raw_exact"
        assert CHUNK[span.evidence_start : span.evidence_end] == quote

    def test_normalized_exact_handles_ocr_whitespace(self):
        """OCR 常见：原文有全角空格/换行，模型回引时丢掉了。"""
        span = locate_evidence("人参味甘、微苦", CHUNK)
        assert span.match_mode == "normalized_exact"
        # 返回的必须是**原文切片**，保留原始空格
        assert span.evidence_quote == CHUNK[span.evidence_start : span.evidence_end]
        assert "　" in span.evidence_quote

    def test_normalized_exact_handles_fullwidth_punctuation(self):
        span = locate_evidence("煎服,3~9g", CHUNK)
        assert span.match_mode in {"normalized_exact", "fuzzy"}
        assert span.is_locatable

    def test_fuzzy_within_edit_distance(self):
        span = locate_evidence("大补元气，复脉固托", CHUNK, max_edit_distance=2)
        assert span.match_mode == "fuzzy"
        assert span.edit_distance <= 2

    def test_not_found_when_absent(self):
        span = locate_evidence("此句教材中根本不存在无论如何", CHUNK)
        assert span.match_mode == "not_found"
        assert not span.is_locatable

    def test_too_short_quote_rejected(self):
        """单字证据不予采信，否则「甘」会在全书命中无数次。"""
        assert locate_evidence("甘", CHUNK).match_mode == "not_found"

    def test_empty_inputs(self):
        assert locate_evidence("", CHUNK).match_mode == "not_found"
        assert locate_evidence("大补元气", "").match_mode == "not_found"


class TestParseDosage:
    def test_range_with_unit(self):
        dosage = parse_dosage("煎服，3～9g")
        assert (dosage.value_min, dosage.value_max, dosage.unit) == (3.0, 9.0, "g")
        assert dosage.route == "煎服"
        assert is_well_formed(dosage)

    def test_ascii_hyphen_and_chinese_unit(self):
        dosage = parse_dosage("每日 1.5-3 克")
        assert (dosage.value_min, dosage.value_max, dosage.unit) == (1.5, 3.0, "g")

    def test_single_value(self):
        dosage = parse_dosage("研末服，0.5g")
        assert dosage.value_min == dosage.value_max == 0.5
        assert dosage.route == "研末服"

    def test_reversed_range_is_corrected(self):
        """OCR 可能把 3～9 识别成 9～3。"""
        dosage = parse_dosage("9～3g")
        assert (dosage.value_min, dosage.value_max) == (3.0, 9.0)

    def test_condition_captured(self):
        dosage = parse_dosage("煎服，9～30g；儿童减量")
        assert dosage.condition is not None and "减量" in dosage.condition

    def test_pregnancy_condition(self):
        dosage = parse_dosage("3～9g，孕妇慎用")
        assert dosage.condition is not None and "孕妇" in dosage.condition

    def test_no_numeric_value_is_not_well_formed(self):
        """高风险剂量断言若解析不出数值，必须能被识别出来并进 REVIEW。"""
        dosage = parse_dosage("适量外用")
        assert dosage.value_min is None
        assert not is_well_formed(dosage)
        assert dosage.raw == "适量外用"

    def test_empty_input(self):
        dosage = parse_dosage("")
        assert dosage.raw == "" and not is_well_formed(dosage)

    def test_raw_always_preserved(self):
        text = "先煎，15～30g，大剂量可用至60g"
        assert parse_dosage(text).raw == text
