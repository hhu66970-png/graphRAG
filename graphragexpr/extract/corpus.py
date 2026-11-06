"""Load page-aware OCR Markdown and create retrieval chunks."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path


PAGE_RE = re.compile(r"<!--\s*source-page:\s*(\d+)\s*-->")
HEADING_RE = re.compile(r"^(#{2,4})\s+(.+)$", re.MULTILINE)
SENTENCE_END_RE = re.compile(r"[。！？；]\s*$")


@dataclass(frozen=True)
class ChunkRecord:
    index: int
    text: str
    page_start: int | None
    page_end: int | None
    section: str
    source: str

    def to_dict(self) -> dict:
        """把数据类转成普通字典。"""
        return asdict(self)


def _split_pages(markdown: str) -> list[tuple[int | None, str]]:
    """按页码标记切开 Markdown。"""
    matches = list(PAGE_RE.finditer(markdown))
    if not matches:
        return [(None, markdown)]
    pages: list[tuple[int | None, str]] = []
    for position, match in enumerate(matches):
        start = match.end()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(markdown)
        pages.append((int(match.group(1)), markdown[start:end].strip()))
    return pages


def _paragraphs(text: str) -> list[str]:
    """按段落切分页面文本。"""
    return [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def load_chunk_records(
    path: str | Path,
    chunk_size: int = 900,
    overlap: int = 120,
    max_chunks: int | None = None,
) -> list[ChunkRecord]:
    """把教材 Markdown 切成带页码的 ChunkRecord。"""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(
            f"语料不存在: {source_path}\n"
            "请先运行 scripts/preprocess_pdf.py 生成 origdata/zhongyaoxue.md。"
        )
    if chunk_size <= overlap or overlap < 0:
        raise ValueError("chunk_size 必须大于 overlap，且 overlap 不能为负数")

    markdown = source_path.read_text(encoding="utf-8")
    records: list[ChunkRecord] = []
    current_section = ""
    carry = ""
    carry_page: int | None = None
    carry_end_page: int | None = None
    carry_section = ""

    def emit(text: str, start_page: int | None, end_page: int | None, section: str) -> None:
        """向结果列表追加一条记录。"""
        clean = text.strip()
        if clean:
            records.append(ChunkRecord(len(records), clean, start_page, end_page, section, source_path.name))

    for page_number, page_text in _split_pages(markdown):
        for paragraph in _paragraphs(page_text):
            heading = HEADING_RE.match(paragraph)
            if heading:
                current_section = heading.group(2).strip()
                paragraph = current_section

            candidate = f"{carry}\n{paragraph}".strip() if carry else paragraph
            start_page = carry_page if carry else page_number
            candidate_section = carry_section if carry else current_section
            while len(candidate) > chunk_size:
                window = candidate[:chunk_size]
                split_at = max(window.rfind(mark) for mark in "。！？；\n") + 1
                if split_at < int(chunk_size * 0.55):
                    split_at = chunk_size
                emit(candidate[:split_at], start_page, page_number, candidate_section)
                candidate = candidate[max(0, split_at - overlap):].strip()
                start_page = page_number
                candidate_section = current_section
                if max_chunks and len(records) >= max_chunks:
                    return records
            carry = candidate
            carry_page = start_page
            carry_end_page = page_number
            carry_section = candidate_section

            if len(carry) >= int(chunk_size * 0.72) and SENTENCE_END_RE.search(carry):
                emit(carry, carry_page, page_number, carry_section)
                carry = carry[-overlap:].strip() if overlap else ""
                carry_page = page_number
                carry_end_page = page_number
                carry_section = current_section
                if max_chunks and len(records) >= max_chunks:
                    return records

    emit(carry, carry_page, carry_end_page, carry_section)
    return records[:max_chunks] if max_chunks else records


def records_from_environment() -> list[ChunkRecord]:
    """按 .env 中的语料路径与块大小加载文本块。"""
    root = Path(__file__).resolve().parents[2]
    corpus_path = Path(os.getenv("CORPUS_PATH", root / "origdata" / "zhongyaoxue.md"))
    max_chunks = int(os.getenv("MAX_CHUNKS", "30"))
    return load_chunk_records(
        corpus_path,
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        max_chunks=max_chunks or None,
    )
