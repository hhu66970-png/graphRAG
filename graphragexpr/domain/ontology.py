"""Controlled ontology loader and validator.

唯一事实源是 ontology.yaml。本模块只负责加载、校验与查询，不得内联任何
硬编码的类型或关系清单。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .errors import (
    DomainRangeViolation,
    OntologyError,
    UnknownEntityTypeError,
    UnknownPredicateError,
)

ONTOLOGY_PATH = Path(__file__).with_name("ontology.yaml")


@dataclass(frozen=True)
class Predicate:
    id: str
    display: str
    domain: tuple[str, ...]
    range: tuple[str, ...]
    high_risk: bool
    qualifiers: tuple[str, ...]


class Ontology:
    """Immutable view over ontology.yaml."""

    def __init__(self, raw: dict) -> None:
        self.schema_version: str = raw["schema_version"]
        self.entity_types: dict[str, dict] = raw["entity_types"]
        self.value_object_types: dict[str, dict] = raw.get("value_object_types", {})
        self.predicates: dict[str, Predicate] = {
            pid: Predicate(
                id=pid,
                display=spec["display"],
                domain=tuple(spec["domain"]),
                range=tuple(spec["range"]),
                high_risk=bool(spec.get("high_risk", False)),
                qualifiers=tuple(spec.get("qualifiers", []) or ()),
            )
            for pid, spec in raw["predicates"].items()
        }
        self.legacy_relation_aliases: dict[str, str] = raw.get("legacy_relation_aliases", {}) or {}
        self.legacy_entity_aliases: dict[str, str] = raw.get("legacy_entity_aliases", {}) or {}
        self._validate()

    def _validate(self) -> None:
        """Fail fast on a malformed ontology rather than at write time."""
        known = set(self.entity_types) | set(self.value_object_types)
        for pid, pred in self.predicates.items():
            bad = (set(pred.domain) | set(pred.range)) - known
            if bad:
                raise OntologyError(f"谓词 {pid} 引用了未声明的类型: {sorted(bad)}")
        for legacy, pid in self.legacy_relation_aliases.items():
            if pid not in self.predicates:
                raise OntologyError(f"legacy_relation_aliases[{legacy}] 指向未声明的谓词 {pid}")
        for legacy, tid in self.legacy_entity_aliases.items():
            if tid not in known:
                raise OntologyError(f"legacy_entity_aliases[{legacy}] 指向未声明的类型 {tid}")

    # ---- 类型 -------------------------------------------------------------

    def is_entity_type(self, type_id: str) -> bool:
        """判断是否为本体中的实体类型。"""
        return type_id in self.entity_types

    def is_value_object(self, type_id: str) -> bool:
        """判断是否为值对象类型（如剂量）。"""
        return type_id in self.value_object_types

    def resolve_entity_type(self, raw: str, *, allow_legacy: bool = True) -> str:
        """Resolve an entity type ID, raising on unknown input.

        未知类型必须记录并拒绝，**不得统一降级为「概念」**。
        """
        value = (raw or "").strip()
        if value in self.entity_types or value in self.value_object_types:
            return value
        if allow_legacy and value in self.legacy_entity_aliases:
            return self.legacy_entity_aliases[value]
        raise UnknownEntityTypeError(value)

    def display_of(self, type_id: str) -> str:
        """返回类型或谓词的中文展示名。"""
        if type_id in self.entity_types:
            return self.entity_types[type_id]["display"]
        if type_id in self.value_object_types:
            return self.value_object_types[type_id]["display"]
        raise UnknownEntityTypeError(type_id)

    # ---- 谓词 -------------------------------------------------------------

    def resolve_predicate(self, raw: str, *, allow_legacy: bool = True) -> str:
        """把自由文本/别名解析成本体谓词 ID，未知则拒绝。"""
        value = (raw or "").strip()
        if value in self.predicates:
            return value
        if allow_legacy and value in self.legacy_relation_aliases:
            return self.legacy_relation_aliases[value]
        raise UnknownPredicateError(value)

    def predicate(self, predicate_id: str) -> Predicate:
        """按 ID 取出谓词定义。"""
        try:
            return self.predicates[predicate_id]
        except KeyError as exc:
            raise UnknownPredicateError(predicate_id) from exc

    def is_high_risk(self, predicate_id: str) -> bool:
        """判断该谓词是否标记为高风险。"""
        return self.predicate(predicate_id).high_risk

    def check_domain_range(self, predicate_id: str, subject_type: str, object_type: str) -> None:
        """Raise DomainRangeViolation if the triple's endpoint types are illegal."""
        pred = self.predicate(predicate_id)
        if subject_type not in pred.domain or object_type not in pred.range:
            raise DomainRangeViolation(predicate_id, subject_type, object_type)

    def allowed_object_types(self, predicate_id: str) -> tuple[str, ...]:
        """返回该谓词允许的宾语类型。"""
        return self.predicate(predicate_id).range

    def high_risk_predicates(self) -> tuple[str, ...]:
        """列出全部高风险谓词 ID。"""
        return tuple(pid for pid, p in self.predicates.items() if p.high_risk)

    def prompt_schema_block(self) -> str:
        """Render the ontology as a compact block for the extraction prompt.

        Prompt 里出现的类型/关系清单必须由本体渲染，避免 Prompt 与 Schema 漂移。
        """
        lines = ["实体类型（只能使用这些 ID）："]
        for tid, spec in self.entity_types.items():
            lines.append(f"  {tid} = {spec['display']}（{spec['description']}）")
        lines.append("关系类型（只能使用这些 ID，并满足起止类型约束）：")
        for pid, pred in self.predicates.items():
            risk = "，高风险" if pred.high_risk else ""
            lines.append(
                f"  {pid} = {pred.display}: {'/'.join(pred.domain)} -> {'/'.join(pred.range)}{risk}"
            )
        return "\n".join(lines)


@lru_cache(maxsize=1)
def load_ontology(path: str | Path | None = None) -> Ontology:
    """Load and cache the ontology. Cache key includes the path."""
    target = Path(path) if path else ONTOLOGY_PATH
    with target.open("r", encoding="utf-8") as handle:
        return Ontology(yaml.safe_load(handle))
