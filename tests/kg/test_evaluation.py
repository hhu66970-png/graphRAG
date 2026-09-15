"""参照集解析与评测指标的单元测试（设计 §9.1 / §9.2）。"""

from __future__ import annotations

from graphragexpr.domain.ids import normalize_text
from graphragexpr.domain.models import ChunkRecord
from graphragexpr.evaluation.extraction_eval import _match
from graphragexpr.evaluation.reference import (
    build_reference,
    coverage_summary,
    parse_efficacies,
    parse_nature_flavor,
)

MAHUANG = (
    "麻黄\n"
    "Mahuang（《神农本草经》）\n"
    "本品为麻黄科植物草麻黄的干燥草质茎。\n"
    "【药性】辛、微苦，温。归肺、膀胱经。\n"
    "【功效】发汗解表，宣肺平喘，利水消肿。\n"
    "【用法用量】煎服，2～10g。\n"
)
GUIZHI = (
    "桂枝\n"
    "Guizhi（《名医别录》）\n"
    "【药性】辛、甘，温。归心、肺、膀胱经。\n"
    "【功效】发汗解肌，温通经脉，助阳化气，平冲降逆。\n"
)


def _chunk(i: int, text: str) -> ChunkRecord:
    """测试用：构造一个教材文本块。"""
    return ChunkRecord(
        chunk_id=f"c{i}", document_id="doc", section_path=["解表药"], order_in_entry=i,
        text=text, normalized_text=normalize_text(text),
        page_start=80 + i, page_end=80 + i, content_hash=f"h{i}",
    )


class TestParseNatureFlavor:
    """测试类：覆盖「ParseNatureFlavor」相关行为。"""
    def test_splits_flavour_and_meridian(self):
        """测试：拆分药味且归经。"""
        flavours, raw, meridians = parse_nature_flavor("辛、微苦，温。归肺、膀胱经。")
        assert flavours == {"辛", "微苦", "温"}
        assert meridians == {"肺经", "膀胱经"}
        assert raw == "辛、微苦，温"

    def test_multiple_meridians(self):
        """测试：多个归经。"""
        _, _, meridians = parse_nature_flavor("辛、甘，温。归心、肺、膀胱经。")
        assert meridians == {"心经", "肺经", "膀胱经"}

    def test_tolerates_ocr_bracket_noise(self):
        """测试：容忍OCR括号噪声。"""
        flavours, _, meridians = parse_nature_flavor("］辛，温。归肺、脾、胃经。")
        assert "辛" in flavours and "］辛" not in flavours
        assert meridians == {"肺经", "脾经", "胃经"}

    def test_missing_meridian_section(self):
        """测试：缺失归经章节。"""
        flavours, _, meridians = parse_nature_flavor("苦，寒。")
        assert flavours == {"苦", "寒"} and meridians == set()

    def test_empty_input(self):
        """测试：空输入。"""
        assert parse_nature_flavor("") == (set(), "", set())


class TestParseEfficacies:
    """测试类：覆盖「ParseEfficacies」相关行为。"""
    def test_splits_terms(self):
        """测试：拆分术语。"""
        assert parse_efficacies("发汗解表，宣肺平喘，利水消肿。") == {
            "发汗解表", "宣肺平喘", "利水消肿"
        }

    def test_drops_single_char_fragments(self):
        """测试：丢弃单个字符碎片。"""
        assert "，" not in parse_efficacies("解表散寒，透疹，消疮。")
        assert parse_efficacies("解表散寒，透疹，消疮。") == {"解表散寒", "透疹", "消疮"}

    def test_stops_at_sentence_end(self):
        """测试：停止在句子末尾。"""
        result = parse_efficacies("解表散寒，行气和胃。用于风寒感冒，咳嗽呕恶。")
        assert result == {"解表散寒", "行气和胃"}


class TestBuildReference:
    """测试类：覆盖「BuildReference」相关行为。"""
    def test_builds_per_herb(self):
        """测试：构建按每味中药。"""
        refs = build_reference([_chunk(0, MAHUANG), _chunk(1, GUIZHI)])
        assert set(refs) == {normalize_text("麻黄"), normalize_text("桂枝")}
        mahuang = refs[normalize_text("麻黄")]
        assert mahuang.meridians == {"肺经", "膀胱经"}
        assert "发汗解表" in mahuang.efficacies
        assert "2～10g" in mahuang.dosage_raw

    def test_fields_without_entry_name_are_skipped(self):
        """字段必须能明确归属到某味药，否则不能进参照集。"""
        orphan = "【药性】辛，温。归肺经。\n【功效】发散风寒。\n"
        assert build_reference([_chunk(0, orphan)]) == {}

    def test_continuation_chunk_merges_into_same_herb(self):
        """测试：续写文本块合并并入相同中药。"""
        cont = "【用法用量】煎服，2～10g。\n"
        refs = build_reference([_chunk(0, MAHUANG.replace("【用法用量】煎服，2～10g。\n", "")),
                                _chunk(1, cont)])
        assert refs[normalize_text("麻黄")].dosage_raw

    def test_coverage_summary(self):
        """测试：覆盖率summary。"""
        refs = build_reference([_chunk(0, MAHUANG), _chunk(1, GUIZHI)])
        summary = coverage_summary(refs)
        assert summary["herbs"] == 2
        assert summary["with_meridians"] == 2
        assert summary["total_efficacy_terms"] == 7   # 3 + 4

    def test_empty_input(self):
        """测试：空输入。"""
        assert build_reference([]) == {}


class TestMatching:
    """测试类：覆盖「Matching」相关行为。"""
    def test_exact_match(self):
        """测试：精确match。"""
        hit, missed, extra = _match({"发汗解表"}, {"发汗解表"})
        assert hit == {"发汗解表"} and not missed and not extra

    def test_containment_counts_as_hit(self):
        """教材写「发汗解表」，图谱抽成「发汗」也应算召回，否则低估。"""
        hit, missed, _ = _match({"发汗解表"}, {"发汗"})
        assert hit == {"发汗解表"} and not missed

    def test_single_char_does_not_match(self):
        """单字不得碰瓷：「温」不应命中「温通经脉」。"""
        hit, missed, _ = _match({"温通经脉"}, {"温"})
        assert not hit and missed == {"温通经脉"}

    def test_extra_beyond_field_is_reported(self):
        """测试：额外超出字段为被报告。"""
        _, _, extra = _match({"发汗解表"}, {"发汗解表", "祛风除湿"})
        assert extra == {"祛风除湿"}

    def test_empty_prediction(self):
        """测试：空预测。"""
        hit, missed, extra = _match({"发汗解表"}, set())
        assert not hit and missed == {"发汗解表"} and not extra


class TestCanonicalizePredicted:
    """图谱取值与参照集必须归一到同一粒度后再比（首次评测因此低估性味召回）。"""

    def test_meridian_string_is_split(self):
        """测试：归经字符串为拆分。"""
        from graphragexpr.evaluation.extraction_eval import canonicalize_predicted
        assert canonicalize_predicted("meridians", {"归肺、膀胱经"}) == {"肺经", "膀胱经"}

    def test_already_split_meridian_kept(self):
        """测试：已经拆分归经保留。"""
        from graphragexpr.evaluation.extraction_eval import canonicalize_predicted
        assert canonicalize_predicted("meridians", {"肺经", "膀胱"}) == {"肺经", "膀胱经"}

    def test_nature_flavor_string_is_split(self):
        """测试：性flavor字符串为拆分。"""
        from graphragexpr.evaluation.extraction_eval import canonicalize_predicted
        assert canonicalize_predicted("nature_flavor", {"辛、微苦，温"}) == {"辛", "微苦", "温"}

    def test_efficacy_string_is_split(self):
        """测试：功效字符串为拆分。"""
        from graphragexpr.evaluation.extraction_eval import canonicalize_predicted
        assert canonicalize_predicted("efficacies", {"发汗解表，宣肺平喘"}) == {
            "发汗解表", "宣肺平喘"
        }

    def test_single_char_flavour_matches_after_canonicalisation(self):
        """测试：单个字符药味匹配之后规范化。"""
        from graphragexpr.evaluation.extraction_eval import canonicalize_predicted
        predicted = canonicalize_predicted("nature_flavor", {"辛、微苦，温"})
        hit, missed, _ = _match({"辛", "微苦", "温"}, predicted, min_token_len=1)
        assert hit == {"辛", "微苦", "温"} and not missed
