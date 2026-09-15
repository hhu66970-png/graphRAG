"""药物条目边界识别测试（设计 §2.1.2-D 的前置条件）。"""

from __future__ import annotations

from graphragexpr.domain.ids import normalize_text
from graphragexpr.domain.models import ChunkRecord
from graphragexpr.ingest.structure import (
    assign_entry_ids,
    entry_coverage,
    find_entry_starts,
    has_entry_fields,
)

GUIZHI = (
    "第八章解表药57\n"
    "桂枝\n"
    "Guizhi（《名医别录》）\n"
    "本品为樟科植物肉桂CinnamomumcassiaPresl的干燥嫩枝。主产于广东、广西。\n"
    "【药性】辛、甘，温。归心、肺、膀胱经。\n"
    "【功效】发汗解肌，温通经脉，助阳化气，平冲降逆。\n"
)
ZISUYE = (
    "紫苏叶\n"
    "Zisuye（《名医别录》）\n"
    "本品为唇形科植物紫苏的干燥叶。\n"
    "【药性】辛，温。归肺、脾、胃经。\n"
)
CONTINUATION = "此外，桂枝有镇痛、抗炎、增加冠脉血流量等作用。属于上一条目的续写内容。\n"


def _chunk(index: int, text: str) -> ChunkRecord:
    """测试用：构造一个教材文本块。"""
    return ChunkRecord(
        chunk_id=f"c{index}", document_id="doc", section_path=["第八章解表药"],
        order_in_entry=index, text=text, normalized_text=normalize_text(text),
        page_start=57 + index, page_end=57 + index, content_hash=f"h{index}",
    )


class TestFindEntryStarts:
    """测试类：覆盖「FindEntryStarts」相关行为。"""
    def test_detects_single_entry(self):
        """测试：检出单个条目。"""
        starts = find_entry_starts(GUIZHI)
        assert len(starts) == 1
        assert starts[0].name == "桂枝"
        assert starts[0].pinyin == "Guizhi"

    def test_detects_multiple_entries_in_order(self):
        """测试：检出多个条目在顺序。"""
        starts = find_entry_starts(GUIZHI + ZISUYE)
        assert [s.name for s in starts] == ["桂枝", "紫苏叶"]
        assert starts[0].offset < starts[1].offset

    def test_ignores_plain_prose(self):
        """测试：忽略纯散文。"""
        prose = "本品辛甘温煦，甘温通阳扶卫，开滕发汗之力较麻黄温和。\n常与麻黄同用。\n"
        assert find_entry_starts(prose) == []

    def test_ignores_latin_without_classic_reference(self):
        """正文里的拉丁学名不得被误判为条目起始。"""
        text = "本品为樟科植物肉桂\nCinnamomumcassiaPresl的干燥嫩枝。\n"
        assert find_entry_starts(text) == []

    def test_empty_input(self):
        """测试：空输入。"""
        assert find_entry_starts("") == []


class TestEntryFields:
    """测试类：覆盖「EntryFields」相关行为。"""
    def test_recognises_structured_fields(self):
        """测试：识别结构化字段。"""
        assert has_entry_fields(GUIZHI)

    def test_plain_text_has_no_fields(self):
        """测试：纯文本具有没有字段。"""
        assert not has_entry_fields("这是一段没有结构标记的正文。")


class TestAssignEntryIds:
    """测试类：覆盖「AssignEntryIds」相关行为。"""
    def test_continuation_chunk_inherits_entry(self):
        """测试：续写文本块继承条目。"""
        chunks = [_chunk(0, GUIZHI), _chunk(1, CONTINUATION)]
        updated = assign_entry_ids(chunks, document_id="doc")
        assert updated[0].entry_id is not None
        assert updated[1].entry_id == updated[0].entry_id
        assert updated[1].order_in_entry == 1

    def test_new_entry_starts_new_id_and_resets_order(self):
        """测试：新条目开始新ID且重置顺序。"""
        chunks = [_chunk(0, GUIZHI), _chunk(1, CONTINUATION), _chunk(2, ZISUYE)]
        updated = assign_entry_ids(chunks, document_id="doc")
        assert updated[2].entry_id != updated[0].entry_id
        assert updated[2].order_in_entry == 0

    def test_front_matter_has_no_entry(self):
        """测试：文前内容具有没有条目。"""
        chunks = [_chunk(0, "目录\n第一章中药的起源\n"), _chunk(1, GUIZHI)]
        updated = assign_entry_ids(chunks, document_id="doc")
        assert updated[0].entry_id is None
        assert updated[1].entry_id is not None

    def test_chunk_with_two_entries_belongs_to_the_last(self):
        """一个块里出现两条目起始时就近归属后者，有利于与后续块配对。"""
        chunks = [_chunk(0, GUIZHI + ZISUYE), _chunk(1, CONTINUATION)]
        updated = assign_entry_ids(chunks, document_id="doc")
        assert updated[0].entry_id == updated[1].entry_id

    def test_entry_id_is_stable(self):
        """测试：条目ID为稳定。"""
        chunks = [_chunk(0, GUIZHI)]
        first = assign_entry_ids(chunks, document_id="doc")[0].entry_id
        second = assign_entry_ids(chunks, document_id="doc")[0].entry_id
        assert first == second

    def test_coverage_report(self):
        """测试：覆盖率报告。"""
        chunks = [_chunk(0, "目录\n"), _chunk(1, GUIZHI), _chunk(2, CONTINUATION)]
        updated = assign_entry_ids(chunks, document_id="doc")
        report = entry_coverage(updated)
        assert report["chunks"] == 3
        assert report["chunks_with_entry"] == 2
        assert report["distinct_entries"] == 1

    def test_empty_input(self):
        """测试：空输入。"""
        assert assign_entry_ids([], document_id="doc") == []


class TestAnchorNameGuard:
    """条目名必须 ≥2 字：单字命中一律是 OCR/正则边界造成的截断。"""

    def test_single_char_name_rejected(self):
        # 「蓟」应为大蓟/小蓟，「黄」为某药名被切断——实测出现过的两例
        """测试：单个字符名称被拒绝。"""
        for bad in ("蓟", "黄"):
            text = f"{bad}\nJi（《神农本草经》）\n本品为菊科植物。\n"
            assert find_entry_starts(text) == [], f"{bad} 不应被识别为条目"

    def test_two_char_name_accepted(self):
        """测试：两字符名称通过。"""
        starts = find_entry_starts("大蓟\nDaji（《名医别录》）\n本品为菊科植物。\n")
        assert [s.name for s in starts] == ["大蓟"]
