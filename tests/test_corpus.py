"""Tests for page-aware OCR corpus chunking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "graphragexpr" / "extract"))

from corpus import load_chunk_records


def test_chunk_records_keep_page_and_section(tmp_path):
    """测试：文本块记录保留页码且章节。"""
    corpus = tmp_path / "book.md"
    corpus.write_text(
        """# 中药学

<!-- source-page: 28 -->

## 第一章 中药的起源

中药是在中医药理论指导下使用的物质。

<!-- source-page: 29 -->

中药包括中药材、中药饮片和中成药。
""",
        encoding="utf-8",
    )
    records = load_chunk_records(corpus, chunk_size=80, overlap=10)
    assert records
    assert records[0].page_start == 28
    assert any(record.page_end == 29 for record in records)
    assert all(record.section == "第一章 中药的起源" for record in records)
    assert all(record.source == "book.md" for record in records)


def test_chunk_limit(tmp_path):
    """测试：文本块上限。"""
    corpus = tmp_path / "book.md"
    corpus.write_text("<!-- source-page: 1 -->\n\n" + "中药具有功效。" * 100, encoding="utf-8")
    records = load_chunk_records(corpus, chunk_size=80, overlap=10, max_chunks=2)
    assert len(records) == 2
