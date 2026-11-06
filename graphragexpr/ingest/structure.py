"""Herb-entry structure detection.

跨块补全的主依据是 `entry_id` 而非「相邻块」。本模块从《中药学》OCR 文本中识别
药物条目边界，为每个 Chunk 赋予稳定的 `entry_id`。

教材的条目版式高度规整：

    桂枝
    Guizhi（《名医别录》）
    本品为樟科植物肉桂……的干燥嫩枝。
    【药性】辛、甘，温。归心、肺、膀胱经。
    【功效】发汗解肌，温通经脉，助阳化气，平冲降逆。
    【应用】……

其中「药名换行后接拼音 + （《典籍》）」这一行是可靠的条目起始标记：
中文正文里几乎不会出现「拉丁字母串紧跟书名号左括号」的组合。

本模块属 ingest 层，只做结构解析，不调用 Neo4j、不调用 LLM。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ..domain.ids import section_id
from ..domain.models import ChunkRecord

# 药名独占一行，下一行以拼音开头并紧跟（《
# 名称要求 **≥2 个汉字**：教材条目名不存在单字药，单字命中一律是 OCR 换行或
# 正则边界造成的截断（实测出现过「蓟」应为大蓟/小蓟、「黄」为某药名被切断），
# 而错误的条目名会成为 I3 补全的锚点，把事实归给一个不存在的中药。
ENTRY_START_RE = re.compile(
    r"(?m)^[ \t]*(?P<name>[一-鿿][一-鿿·（）()]{1,9})[ \t]*\n"
    r"[ \t]*(?P<pinyin>[A-Za-z][A-Za-z \-]{1,30})[ \t]*[（(]《"
)

# 条目内的结构化字段标记，用于确认「这确实是一条药物条目」
ENTRY_FIELD_MARKERS = ("【药性】", "【功效】", "【应用】", "【用法用量】", "【使用注意】")


@dataclass(frozen=True)
class EntryStart:
    """One detected herb entry boundary."""

    offset: int          # 在拼接全文中的字符偏移
    name: str            # 药名
    pinyin: str


def find_entry_starts(text: str) -> list[EntryStart]:
    """Locate herb-entry boundaries in a block of textbook text.

    Args:
        text: 教材正文（可以是单块，也可以是拼接后的全书）。

    Returns:
        按偏移升序排列的 EntryStart 列表。

    Side effects: 无。
    """
    starts: list[EntryStart] = []
    for match in ENTRY_START_RE.finditer(text):
        starts.append(
            EntryStart(
                offset=match.start(),
                name=match.group("name").strip(),
                pinyin=match.group("pinyin").strip(),
            )
        )
    return starts


def has_entry_fields(text: str) -> bool:
    """True when the text carries at least one structured herb-entry field."""
    return any(marker in text for marker in ENTRY_FIELD_MARKERS)


def assign_entry_ids(
    chunks: Sequence[ChunkRecord], *, document_id: str
) -> list[ChunkRecord]:
    """Attach a stable `entry_id` (and `order_in_entry`) to each chunk.

    规则：
      - 块内出现条目起始标记时，该块归属**最后一个**起始的条目（前半段属于上一条目，
        但块是最小单元，就近归属更有利于跨块配对）；
      - 未出现起始标记的块继承前一块的 entry_id；
      - 条目开始之前的前言/目录块 entry_id 为 None，不参与条目级配对。

    Args:
        chunks: 已按 order_in_entry / index 升序排列的块。
        document_id: 用于构造稳定 entry_id。

    Returns:
        新的 ChunkRecord 列表（ChunkRecord 是 frozen 模型，返回副本）。

    Side effects: 无。
    """
    updated: list[ChunkRecord] = []
    current_entry: str | None = None
    current_name: str | None = None
    order_in_entry = 0

    for chunk in chunks:
        starts = find_entry_starts(chunk.text)
        if starts:
            last = starts[-1]
            current_name = last.name
            # entry_id 与 section_id 同构：文档 + 结构路径哈希，跨版本稳定
            current_entry = section_id(document_id, [*chunk.section_path, f"entry:{last.name}"])
            order_in_entry = 0
        elif current_entry is not None:
            order_in_entry += 1

        updated.append(
            chunk.model_copy(
                update={
                    "entry_id": current_entry,
                    "order_in_entry": order_in_entry if current_entry else chunk.order_in_entry,
                }
            )
        )
    return updated


def build_entry_index(
    chunks: Sequence[ChunkRecord], *, document_id: str
) -> dict[str, str]:
    """Map each `entry_id` to the herb name that entry is about.

    I3 跨块补全需要知道「这一段在讲哪味药」，才能把「本品味辛性凉」这类
    省略主语的句子正确归属。entry_id 本身是哈希，不含可读信息，故单独建索引。

    Args:
        chunks: 原始块（未经 assign_entry_ids 亦可，本函数自行识别边界）。
        document_id: 与 assign_entry_ids 使用同一个，保证 entry_id 一致。

    Returns:
        {entry_id: 药名}。

    Side effects: 无。
    """
    index: dict[str, str] = {}
    for chunk in chunks:
        for start in find_entry_starts(chunk.text):
            entry = section_id(document_id, [*chunk.section_path, f"entry:{start.name}"])
            index.setdefault(entry, start.name)
    return index


def entry_coverage(chunks: Sequence[ChunkRecord]) -> dict:
    """Report how much of the corpus was assigned to herb entries."""
    total = len(chunks)
    with_entry = sum(1 for c in chunks if c.entry_id)
    entries = {c.entry_id for c in chunks if c.entry_id}
    return {
        "chunks": total,
        "chunks_with_entry": with_entry,
        "coverage": (with_entry / total) if total else 0.0,
        "distinct_entries": len(entries),
    }
