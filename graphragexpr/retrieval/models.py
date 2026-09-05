"""Internal retrieval plans and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..domain.models import EvidencePacket

Intent = Literal["attr", "reverse", "intersect", "compare", "path", "fallback"]
StructureType = Literal["card", "table", "matrix", "paths", "summary", "chunks"]


@dataclass(frozen=True)
class RetrievalPlan:
    """Deterministic query plan emitted by the rule-first router."""

    intent: Intent
    linked_entity_ids: tuple[str, ...]
    allowed_predicates: tuple[str, ...]
    max_hops: int
    require_provenance: bool
    structure_type: StructureType
    reason: str
    confidence: float
    is_high_risk: bool = False


@dataclass(frozen=True)
class RetrievalResult:
    """Evidence packet plus the exact same-evidence subgraph used by the UI."""

    packet: EvidencePacket
    subgraph: dict = field(default_factory=lambda: {"nodes": [], "edges": []})
