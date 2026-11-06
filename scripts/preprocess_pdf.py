"""OCR a scanned Chinese textbook PDF into page-aware Markdown.

Each page is cached separately so a long OCR job can resume safely. The script
does not call an online OCR service and never modifies the source PDF.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pymupdf
from rapidocr_onnxruntime import RapidOCR


PAGE_HEADER_RE = re.compile(r"^\s*\d{1,3}\s*(中药学|目录)\s*$")
CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百零〇0-9]+章")
SECTION_RE = re.compile(r"^第[一二三四五六七八九十百零〇0-9]+节")
SUBSECTION_RE = re.compile(r"^[一二三四五六七八九十]+、")


def normalize_line(text: str) -> str:
    """规范化 OCR 行（空白、全半角）。"""
    text = text.replace("\u3000", " ").strip()
    return re.sub(r"[ \t]+", " ", text)


def markdown_line(text: str) -> str:
    """把 OCR 行写成 Markdown。"""
    if CHAPTER_RE.match(text):
        return f"## {text}"
    if SECTION_RE.match(text):
        return f"### {text}"
    if SUBSECTION_RE.match(text) and len(text) <= 36:
        return f"#### {text}"
    return text


def merge_split_headings(lines: list[str]) -> list[str]:
    """Join headings such as ``第一章`` followed by its title on the next OCR line."""
    merged: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if (
            line.startswith("## ")
            and len(line) <= 10
            and index + 1 < len(lines)
            and not lines[index + 1].startswith("#")
            and len(lines[index + 1]) <= 36
        ):
            merged.append(f"{line} {lines[index + 1]}")
            index += 2
            continue
        merged.append(line)
        index += 1
    return merged


def ocr_page(page: pymupdf.Page, engine: RapidOCR, dpi: int, min_score: float) -> list[str]:
    """对单页 PDF 做 OCR。"""
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    result, _ = engine(image)
    lines: list[str] = []
    for item in result or []:
        text = normalize_line(item[1])
        score = float(item[2])
        if not text or score < min_score or PAGE_HEADER_RE.match(text):
            continue
        lines.append(markdown_line(text))
    return merge_split_headings(lines)


def build_parser() -> argparse.ArgumentParser:
    """构造 preprocess_pdf 的命令行参数。"""
    parser = argparse.ArgumentParser(description="扫描版中文 PDF OCR 为 Markdown")
    parser.add_argument("--pdf", required=True, type=Path, help="源 PDF 路径")
    parser.add_argument("--output", required=True, type=Path, help="输出 Markdown 路径")
    parser.add_argument("--start-page", type=int, default=1, help="起始 PDF 页码（从 1 开始）")
    parser.add_argument("--end-page", type=int, help="结束 PDF 页码（包含）")
    parser.add_argument("--dpi", type=int, default=150, help="OCR 渲染 DPI")
    parser.add_argument("--min-score", type=float, default=0.45, help="保留文字的最低置信度")
    parser.add_argument("--force", action="store_true", help="忽略页面缓存并重新 OCR")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pdf_path = args.pdf.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output_path.parent / ".ocr-pages"
    cache_dir.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(pdf_path)
    end_page = args.end_page or document.page_count
    if not (1 <= args.start_page <= end_page <= document.page_count):
        raise ValueError(f"页码必须在 1..{document.page_count} 内，且起始页不大于结束页")

    engine = RapidOCR()
    page_files: list[Path] = []
    for page_number in range(args.start_page, end_page + 1):
        page_file = cache_dir / f"page-{page_number:04d}.md"
        page_files.append(page_file)
        if page_file.exists() and not args.force:
            print(f"[{page_number}/{end_page}] 使用缓存")
            continue

        print(f"[{page_number}/{end_page}] OCR...", flush=True)
        lines = ocr_page(document[page_number - 1], engine, args.dpi, args.min_score)
        body = "\n\n".join(lines).strip()
        page_file.write_text(f"<!-- source-page: {page_number} -->\n\n{body}\n", encoding="utf-8")

    header = (
        "# 中药学（新世纪第五版）OCR 语料\n\n"
        f"> 源文件：{pdf_path.name}\n\n"
        f"> PDF 页码范围：{args.start_page}-{end_page}\n\n"
    )
    combined = header + "\n\n".join(path.read_text(encoding="utf-8").strip() for path in page_files) + "\n"
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(combined, encoding="utf-8")
    tmp_path.replace(output_path)
    print(f"完成：{output_path}（{len(combined):,} 字符）")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已中断；已完成页面保留在缓存中，可用同一命令续跑。", file=sys.stderr)
        raise SystemExit(130)
