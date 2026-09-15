"""I3 cross-chunk completion (设计 §2.1.1-3 与 §2.1.2-D).

关键修正：**相邻 Chunk 只是回退策略，结构条目 ID 才是主依据**。
教材里「人参」的性味在一块、用法用量与禁忌在下一块，二者同属一个 `entry_id`；
按 entry_id 配对既准确又不会把隔壁药材的属性错接过来。只有当结构解析失败
（entry_id 缺失）时，才退到「相邻 Chunk + 锚点实体」。

新增候选必须重新经过 I2 验证（由调用方负责），本模块只产出候选。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..config.settings import KGSettings
from ..domain.models import ChunkRecord

PairingStrategy = str  # "entry_id" | "section_path" | "adjacent"


@dataclass(frozen=True)
class CrossChunkGroup:
    """One group of chunks that may jointly express a single fact."""

    anchor_entity: str
    chunk_ids: tuple[str, ...]
    strategy: PairingStrategy
    page_start: int
    page_end: int


def _anchor_entities(chunk: ChunkRecord, entity_surfaces: Sequence[str]) -> list[str]:
    """Entities explicitly mentioned in this chunk's raw text."""
    return [surface for surface in entity_surfaces if surface and surface in chunk.text]


def build_groups(
    chunks: Sequence[ChunkRecord],
    entity_surfaces_by_chunk: dict[str, Sequence[str]],
    settings: KGSettings,
) -> list[CrossChunkGroup]:
    """Group chunks that should be considered together for fact completion.

    Args:
        chunks: 已按 order_in_entry / index 排好序的块。
        entity_surfaces_by_chunk: chunk_id -> 该块抽出的实体表面形式。
        settings: 提供 cross_chunk_max_neighbors（回退策略的邻居半径）。

    Returns:
        CrossChunkGroup 列表，按锚点实体去重；strategy 标明配对依据，
        便于评测时区分「结构配对」与「相邻回退」的贡献。

    Side effects: 无。
    """
    if not settings.use_cross_chunk:
        return []

    by_entry: dict[str, list[ChunkRecord]] = defaultdict(list)
    without_entry: list[ChunkRecord] = []
    for chunk in chunks:
        if chunk.entry_id:
            by_entry[chunk.entry_id].append(chunk)
        else:
            without_entry.append(chunk)

    groups: list[CrossChunkGroup] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    def _emit(anchor: str, members: Iterable[ChunkRecord], strategy: PairingStrategy) -> None:
        """把当前分组候选输出到跨块补全结果。"""
        ordered = sorted(members, key=lambda c: (c.order_in_entry or 0, c.chunk_id))
        ids = tuple(c.chunk_id for c in ordered)
        if len(ids) < 2:
            return
        key = (anchor, ids)
        if key in seen:
            return
        seen.add(key)
        groups.append(
            CrossChunkGroup(
                anchor_entity=anchor,
                chunk_ids=ids,
                strategy=strategy,
                page_start=min(c.page_start for c in ordered),
                page_end=max(c.page_end for c in ordered),
            )
        )

    # 主依据：同一 entry_id
    for entry_chunks in by_entry.values():
        anchors: set[str] = set()
        for chunk in entry_chunks:
            anchors.update(_anchor_entities(chunk, entity_surfaces_by_chunk.get(chunk.chunk_id, ())))
        for anchor in anchors:
            _emit(anchor, entry_chunks, "entry_id")

    # 次依据：entry_id 缺失但 section_path 相同
    by_section: dict[tuple[str, ...], list[ChunkRecord]] = defaultdict(list)
    for chunk in without_entry:
        by_section[tuple(chunk.section_path)].append(chunk)
    for section_chunks in by_section.values():
        if len(section_chunks) < 2:
            continue
        anchors = set()
        for chunk in section_chunks:
            anchors.update(_anchor_entities(chunk, entity_surfaces_by_chunk.get(chunk.chunk_id, ())))
        for anchor in anchors:
            # 只有该锚点在多个块中都出现，才认为它们在讲同一件事
            members = [
                c for c in section_chunks
                if anchor in entity_surfaces_by_chunk.get(c.chunk_id, ()) or anchor in c.text
            ]
            _emit(anchor, members, "section_path")

    # 回退：相邻块 + 锚点实体
    radius = settings.cross_chunk_max_neighbors
    for position, chunk in enumerate(without_entry):
        window = without_entry[max(0, position - radius) : position + radius + 1]
        if len(window) < 2:
            continue
        for anchor in _anchor_entities(chunk, entity_surfaces_by_chunk.get(chunk.chunk_id, ())):
            members = [c for c in window if anchor in c.text]
            _emit(anchor, members, "adjacent")

    return groups
