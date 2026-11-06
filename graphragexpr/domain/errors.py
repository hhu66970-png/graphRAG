"""Domain exception types."""

from __future__ import annotations


class KGError(Exception):
    """Base class for all knowledge-graph domain errors."""


class OntologyError(KGError):
    """Ontology file is malformed or an unknown type/predicate was used."""


class UnknownEntityTypeError(OntologyError):
    """Entity type is not declared in ontology.yaml.

    Unknown types are rejected rather than mapped to 概念.
    """

    def __init__(self, raw_type: str) -> None:
        super().__init__(f"未声明的实体类型: {raw_type!r}")
        self.raw_type = raw_type


class UnknownPredicateError(OntologyError):
    """Predicate is not declared in ontology.yaml."""

    def __init__(self, raw_predicate: str) -> None:
        super().__init__(f"未声明的关系类型: {raw_predicate!r}")
        self.raw_predicate = raw_predicate


class DomainRangeViolation(OntologyError):
    """Subject or object type is not allowed for this predicate."""

    def __init__(self, predicate: str, subject_type: str, object_type: str) -> None:
        super().__init__(
            f"关系 {predicate} 不允许 {subject_type} -> {object_type}"
        )
        self.predicate = predicate
        self.subject_type = subject_type
        self.object_type = object_type


class EvidenceNotLocatable(KGError):
    """Evidence quote could not be located in the source chunk."""


class RunStateError(KGError):
    """Illegal IngestRun state transition."""


class LLMContractError(KGError):
    """LLM returned output that does not satisfy the required JSON schema."""
