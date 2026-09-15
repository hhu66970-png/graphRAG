"""Structured-field reference set parsed from the textbook itself (设计 §9.1).

《中药学》每味药的条目都有高度规整的结构化字段：

    【药性】辛、微苦，温。归肺、膀胱经。
    【功效】发汗解表，宣肺平喘，利水消肿。
    【用法用量】煎服，2～10g。

这些字段是教材自己给出的**规范表述**，可以纯规则解析、不调用 LLM，
因此可作为抽取质量的参照基准。

**必须说清楚它不是什么**：这是「教材结构化字段参照集」，
不是设计 §9.1 要求的「双人标注并仲裁的金标准集」。两点差别：

1. 覆盖面窄：只覆盖 药性/功效/用法用量 三类字段，不含主治、配伍、禁忌等；
2. **Precision 会被低估**：图谱里合理地包含字段之外的事实（【应用】段落、
   其它章节的交叉引用都会产出正确的功效/主治），这些不在字段里但并不算错。

因此本模块的头号指标是 **字段召回率（Recall@field）**：教材明确写出来的，
我们抽到了没有。Precision 一栏须标注为「字段内精确率」，不得当作总体准确率。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from ..domain.ids import normalize_text
from ..domain.models import ChunkRecord
from ..ingest.structure import find_entry_starts

_FIELD_RE = re.compile(r"【(药性|功效|用法用量|使用注意)】\s*([^【】]{0,300})")
# 药性字段：性味部分在第一个句号之前，归经部分以「归」开头
_MERIDIAN_RE = re.compile(r"归([^。；]{1,40})经")
_SPLIT_TERMS = re.compile(r"[，,、；;]")
# OCR 常见的插入噪声：孤立的方括号、竖线、空白
_NOISE_CHARS = str.maketrans("", "", "］［」「｜|]")


@dataclass
class HerbReference:
    """Structured fields the textbook states for one herb."""

    herb: str
    page_start: int
    page_end: int
    nature_flavor: set[str] = field(default_factory=set)   # 性味成分，如 {辛, 微苦, 温}
    nature_flavor_raw: str = ""
    meridians: set[str] = field(default_factory=set)       # 归经，如 {肺经, 膀胱经}
    efficacies: set[str] = field(default_factory=set)      # 功效，如 {发汗解表, ...}
    dosage_raw: str = ""

    def has_any(self) -> bool:
        """判断参照对象是否含有任一取值。"""
        return bool(self.nature_flavor or self.meridians or self.efficacies or self.dosage_raw)


def _clean(text: str) -> str:
    """清洗教材字段文本以便对齐。"""
    return text.translate(_NOISE_CHARS).replace("\n", "").strip()


def parse_nature_flavor(field_text: str) -> tuple[set[str], str, set[str]]:
    """Split 【药性】 into flavour components and meridians.

    Args:
        field_text: 【药性】后面的原文，如「辛、微苦，温。归肺、膀胱经。」

    Returns:
        (性味成分集合, 性味原串, 归经集合)。归经统一补上「经」字后缀。

    Side effects: 无。
    """
    text = _clean(field_text)
    if not text:
        return set(), "", set()

    meridians: set[str] = set()
    match = _MERIDIAN_RE.search(text)
    if match:
        for part in _SPLIT_TERMS.split(match.group(1)):
            name = part.strip().rstrip("经")
            if name:
                meridians.add(f"{name}经")

    # 性味 = 归经之前的部分
    flavour_part = text[: match.start()] if match else text
    flavour_part = flavour_part.split("。")[0]
    flavours = {
        term.strip() for term in _SPLIT_TERMS.split(flavour_part) if term.strip()
    }
    return flavours, flavour_part.strip(), meridians


def parse_efficacies(field_text: str) -> set[str]:
    """Split 【功效】 into individual efficacy terms."""
    text = _clean(field_text).split("。")[0]
    return {
        term.strip()
        for term in _SPLIT_TERMS.split(text)
        if len(term.strip()) >= 2
    }


def build_reference(chunks: Sequence[ChunkRecord]) -> dict[str, HerbReference]:
    """Extract the reference set from chunks, keyed by normalised herb name.

    条目名取自本块内最后一个条目起始标记；找不到条目名的块跳过——
    字段必须能明确归属到某味药，否则不能作为参照。

    Args:
        chunks: 按 index 升序的文本块。

    Returns:
        {规范化药名: HerbReference}。同一味药跨多块时字段做并集。

    Side effects: 无。
    """
    references: dict[str, HerbReference] = {}
    current_herb: str | None = None
    current_pages: tuple[int, int] = (0, 0)

    for chunk in chunks:
        starts = find_entry_starts(chunk.text)
        if starts:
            current_herb = starts[-1].name
            current_pages = (chunk.page_start, chunk.page_end)
        if current_herb is None:
            continue

        fields = list(_FIELD_RE.finditer(chunk.text))
        if not fields:
            continue

        key = normalize_text(current_herb)
        ref = references.get(key)
        if ref is None:
            ref = HerbReference(
                herb=current_herb, page_start=current_pages[0], page_end=chunk.page_end
            )
            references[key] = ref
        ref.page_end = max(ref.page_end, chunk.page_end)

        for match in fields:
            name, body = match.group(1), match.group(2)
            if name == "药性":
                flavours, raw, meridians = parse_nature_flavor(body)
                ref.nature_flavor |= flavours
                ref.meridians |= meridians
                if raw and not ref.nature_flavor_raw:
                    ref.nature_flavor_raw = raw
            elif name == "功效":
                ref.efficacies |= parse_efficacies(body)
            elif name == "用法用量" and not ref.dosage_raw:
                ref.dosage_raw = _clean(body).split("。")[0]

    return {k: v for k, v in references.items() if v.has_any()}


def coverage_summary(references: dict[str, HerbReference]) -> dict:
    """How much of the corpus the reference set covers."""
    return {
        "herbs": len(references),
        "with_nature_flavor": sum(1 for r in references.values() if r.nature_flavor),
        "with_meridians": sum(1 for r in references.values() if r.meridians),
        "with_efficacies": sum(1 for r in references.values() if r.efficacies),
        "with_dosage": sum(1 for r in references.values() if r.dosage_raw),
        "total_efficacy_terms": sum(len(r.efficacies) for r in references.values()),
        "total_meridian_terms": sum(len(r.meridians) for r in references.values()),
    }
