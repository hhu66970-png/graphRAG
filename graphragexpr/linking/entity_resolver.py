"""I1 entity linking and alias normalisation (设计 §2.1.1-1).

分层流程：文本规范化 → 权威别名精确匹配 → 包含式候选 → 字符候选 → 向量候选
→ 类型与上下文确认。比「按 embedding 阈值直接合并」安全得多，尤其适合
「人参 / 红参 / 生晒参」这类**有关联但不等价**的实体。

三条不可违反的规则：
  1. 身份是 canonical_id，不是 name（SKILL §3.3）。
  2. **绝不跨实体类型合并**——旧图 506 个名字同时挂在多个类型下，正是无类型
     匹配造成的；本模块所有候选召回都先按 entity_type 分桶。
  3. **除第 1 层精确匹配外，任何层都只产生候选，不得自动合并**。
     曾因「最长匹配」层做包含式自动合并，导致 麻黄→麻黄根、黄连→胡黄连、
     沙参→南沙参 等错误归并（麻黄发汗、麻黄根止汗，功效相反）。
向量阈值同样只能生成候选，不能直接确认同一实体（设计 §2.2.3）。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable

from ..config.settings import KGSettings
from ..domain.ids import canonical_id, normalize_text
from ..domain.models import EntityRecord
from ..extract.evidence import bounded_edit_distance

# 向量候选提供者：(surface, entity_type, top_k) -> [(canonical_id, similarity)]
VectorCandidateProvider = Callable[[str, str, int], list[tuple[str, float]]]


@dataclass
class AliasStore:
    """权威别名表 + 已解析实体的注册表，按 (type, normalized_name) 建索引。"""

    _by_canonical: dict[str, EntityRecord] = field(default_factory=dict)
    # (entity_type, normalized_name) -> canonical_id
    _by_name: dict[tuple[str, str], str] = field(default_factory=dict)
    # entity_type -> {normalized_name}
    _names_by_type: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def register(self, record: EntityRecord) -> EntityRecord:
        """Register (or merge aliases into) an entity. Idempotent."""
        existing = self._by_canonical.get(record.canonical_id)
        if existing:
            merged_aliases = sorted({*existing.aliases, *record.aliases})
            record = existing.model_copy(update={"aliases": merged_aliases})
        self._by_canonical[record.canonical_id] = record
        for surface in {record.preferred_name, *record.aliases}:
            key = (record.entity_type, normalize_text(surface))
            self._by_name[key] = record.canonical_id
            self._names_by_type[record.entity_type].add(normalize_text(surface))
        return record

    def load_authority(self, entries: Iterable[dict]) -> None:
        """Load a curated alias list: {canonical_id?, entity_type, preferred_name, aliases}."""
        for entry in entries:
            entity_type = entry["entity_type"]
            preferred = entry["preferred_name"]
            self.register(
                EntityRecord(
                    canonical_id=entry.get("canonical_id")
                    or canonical_id(entity_type, preferred),
                    entity_type=entity_type,
                    preferred_name=preferred,
                    normalized_name=normalize_text(preferred),
                    aliases=list(entry.get("aliases", [])),
                    source_authority=entry.get("source_authority", "curated"),
                    link_confidence=1.0,
                    link_method="exact_alias",
                    review_status="MANUAL_CONFIRMED",
                )
            )

    def get(self, canonical: str) -> EntityRecord | None:
        return self._by_canonical.get(canonical)

    def lookup(self, entity_type: str, surface: str) -> str | None:
        return self._by_name.get((entity_type, normalize_text(surface)))

    def names_of_type(self, entity_type: str) -> set[str]:
        return self._names_by_type[entity_type]

    def all_records(self) -> list[EntityRecord]:
        return list(self._by_canonical.values())


@dataclass(frozen=True)
class LinkDecision:
    record: EntityRecord
    is_new: bool
    candidates: tuple[tuple[str, float], ...] = ()


class EntityResolver:
    """Resolve a surface form to a canonical entity within its declared type."""

    def __init__(
        self,
        store: AliasStore,
        settings: KGSettings,
        vector_candidates: VectorCandidateProvider | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._vector_candidates = vector_candidates

    def resolve(self, surface: str, entity_type: str, *, context: str = "") -> LinkDecision:
        """Resolve `surface` of `entity_type` to a canonical entity.

        Args:
            surface: 教材中出现的表面形式。
            entity_type: 稳定类型 ID；候选召回严格限制在该类型内。
            context: 所在 chunk 文本，用于上下文确认（当前用于共现加权）。

        Returns:
            LinkDecision；`is_new` 为 True 表示新建了规范实体。

        Side effects: 命中或新建的实体会被写入 AliasStore。
        """
        normalized = normalize_text(surface)
        if not normalized:
            raise ValueError("空的实体表面形式")

        # 第 1 层：权威别名/已注册名精确匹配（同类型内）
        hit = self._store.lookup(entity_type, surface)
        if hit:
            record = self._store.get(hit)
            assert record is not None
            return LinkDecision(record=record, is_new=False)

        # 第 2 层：包含式匹配 —— **只生成候选，绝不自动合并**。
        # 中文药名里一个字的前后缀常常就换了物种或药用部位：
        # 麻黄(发汗) / 麻黄根(止汗)、黄连(Coptis) / 胡黄连(Picrorhiza)、
        # 沙参 / 南沙参、防已 / 广防已。设计 §2.1.1-1 的流程本就要求候选经
        # 「类型与上下文确认」，这里按长度比给分，交由下面的阈值裁决。
        containment_candidates = self._containment_candidates(entity_type, normalized)

        # 第 3 层：字符候选（受限编辑距离），同样只做候选
        char_candidates = self._character_candidates(entity_type, normalized)

        # 第 4 层：向量候选，同样只做候选（设计 §2.2.3）
        vector_candidates: list[tuple[str, float]] = []
        if self._vector_candidates is not None:
            vector_candidates = self._vector_candidates(
                surface, entity_type, self._settings.link_max_candidates
            )

        candidates = self._merge_candidates(
            containment_candidates, char_candidates, vector_candidates
        )

        # 第 5 层：类型与上下文确认
        if candidates:
            best_id, best_score = candidates[0]
            if best_score >= self._settings.link_auto_confirm_threshold:
                record = self._merge_alias(best_id, surface, "character_candidate", best_score)
                return LinkDecision(record=record, is_new=False, candidates=tuple(candidates))
            if best_score >= self._settings.link_candidate_threshold:
                # 灰区：建新实体但标记待复核，绝不静默合并
                record = self._create(
                    surface, entity_type, "vector_candidate", best_score, "NEEDS_REVIEW"
                )
                return LinkDecision(record=record, is_new=True, candidates=tuple(candidates))

        record = self._create(surface, entity_type, "new_entity", 1.0, "AUTO_CONFIRMED")
        return LinkDecision(record=record, is_new=True, candidates=tuple(candidates))

    # ---- 内部 -------------------------------------------------------------

    def _containment_candidates(
        self, entity_type: str, normalized: str
    ) -> list[tuple[str, float]]:
        """Names that contain (or are contained by) the surface form, as CANDIDATES only.

        相似度按长度比计算：麻黄(2) vs 麻黄根(3) -> 0.667，低于候选确认阈值，
        因此会新建实体而不是被吞并。完全相同的名字已在第 1 层命中，不会走到这里。
        """
        results: list[tuple[str, float]] = []
        for known in self._store.names_of_type(entity_type):
            if len(known) < 2 or known == normalized:
                continue
            if known in normalized or normalized in known:
                similarity = min(len(known), len(normalized)) / max(len(known), len(normalized))
                candidate_id = self._store.lookup(entity_type, known)
                if candidate_id:
                    results.append((candidate_id, round(similarity, 4)))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[: self._settings.link_max_candidates]

    def _character_candidates(self, entity_type: str, normalized: str) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        max_edit = self._settings.link_max_edit_distance
        for known in self._store.names_of_type(entity_type):
            distance = bounded_edit_distance(normalized, known, max_edit)
            if distance <= max_edit:
                similarity = 1.0 - distance / max(len(normalized), len(known), 1)
                candidate_id = self._store.lookup(entity_type, known)
                if candidate_id:
                    results.append((candidate_id, round(similarity, 4)))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[: self._settings.link_max_candidates]

    @staticmethod
    def _merge_candidates(*candidate_lists: list[tuple[str, float]]) -> list[tuple[str, float]]:
        merged: dict[str, float] = {}
        for candidate_id, score in [c for lst in candidate_lists for c in lst]:
            merged[candidate_id] = max(merged.get(candidate_id, 0.0), score)
        return sorted(merged.items(), key=lambda item: item[1], reverse=True)

    def _merge_alias(
        self, target_id: str, surface: str, method: str, confidence: float
    ) -> EntityRecord:
        record = self._store.get(target_id)
        assert record is not None
        updated = record.model_copy(
            update={
                "aliases": sorted({*record.aliases, surface}),
                "link_confidence": min(record.link_confidence, confidence),
                "link_method": method,
            }
        )
        return self._store.register(updated)

    def _create(
        self, surface: str, entity_type: str, method: str, confidence: float, review: str
    ) -> EntityRecord:
        return self._store.register(
            EntityRecord(
                canonical_id=canonical_id(entity_type, surface),
                entity_type=entity_type,
                preferred_name=surface,
                normalized_name=normalize_text(surface),
                aliases=[],
                link_confidence=confidence,
                link_method=method,  # type: ignore[arg-type]
                review_status=review,  # type: ignore[arg-type]
            )
        )
