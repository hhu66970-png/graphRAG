"""Fakes for the high-risk acceptance tests.

设计原则
--------
1. **不连数据库。** `FakeHighRiskStore` 在内存里按查询意图返回预设行，
   因此这些测试既不需要 Neo4j，也不可能写进任何真实库。
2. **但真实走完编排。** 假存储只替代最外层的 Cypher 执行，条件单元组装、
   极性解释、候选生成、程序核验、状态判定与渲染全部是生产代码。
3. **记录调用参数。** 假存储会留下每次调用的参数，使「参数化查询、限定 run、
   限定结果数」可以被断言，而不是靠读代码相信。
"""

from __future__ import annotations

from pydantic import BaseModel

from graphragexpr.domain.high_risk import TASK_HIGH_RISK_PREDICATES
from graphragexpr.highrisk.claims import ClaimBatch


def assertion_row(**overrides) -> dict:
    """One assertion row shaped exactly like `storage.high_risk_read` returns."""
    row = {
        "assertion_id": "a-default",
        "fact_key": "f-default",
        "predicate": "HAS_DOSAGE",
        "polarity": "POSITIVE",
        "qualifiers": {},
        "object_value": {},
        "evidence_quote": "",
        "evidence_start": 0,
        "evidence_end": 0,
        "match_mode": "exact",
        "validation_status": "ACCEPTED",
        "is_cross_chunk": False,
        "final_score": 1.0,
        "subject_id": "H:麻黄",
        "subject_name": "麻黄",
        "object_id": None,
        "object_name": None,
        "chunk_ids": ["c-1"],
        "chunk_texts": [],
        "chunk_page_starts": [82],
        "chunk_page_ends": [82],
        "chunk_sections": [],
        "chunk_document_ids": ["d-1"],
    }
    row.update(overrides)
    return row


def dosage_row(assertion_id: str, raw: str, *, quote: str | None = None, **overrides) -> dict:
    """A dosage assertion whose stored value object carries the parsed numbers.

    存量 `DosageValue` 本身含 unit 字段；这里如实写出，使
    「存量解析与原文重解析是否一致」的比对是真的在比对。
    """
    from graphragexpr.highrisk.parsing import parse_dosage_units

    bindings = parse_dosage_units(quote or raw)
    stored: dict = {"raw": raw}
    if bindings:
        stored.update(
            {
                "value_min": bindings[0].value_min,
                "value_max": bindings[0].value_max,
                "unit": bindings[0].unit,
            }
        )
    return assertion_row(
        assertion_id=assertion_id,
        predicate="HAS_DOSAGE",
        object_value=stored,
        evidence_quote=quote if quote is not None else raw,
        chunk_texts=[quote if quote is not None else raw],
        **overrides,
    )


class FakeHighRiskStore:
    """In-memory stand-in for the read paths used by the high-risk service."""

    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows = list(rows or [])
        self.calls: list[dict] = []

    def run(self, cypher: str, **params):
        """测试替身：执行一条 Cypher 并返回假结果。"""
        self.calls.append({"cypher": cypher, **params})
        if "EVIDENCED_BY" not in cypher:
            return []
        # 谓词白名单必须在假存储里同样生效：真实查询带 `predicate IN $predicates`，
        # 若这里不筛，测试就会看到比生产环境更多的证据，
        # 「按子问题只读相关类别」这条约束也就测不出来。
        predicates = set(params.get("predicates") or [])
        candidates = [
            row for row in self.rows if not predicates or row.get("predicate") in predicates
        ]
        if "$subject_ids IS NULL" in cypher:
            subject_ids = params.get("subject_ids")
            if subject_ids:
                candidates = [
                    row for row in candidates if row.get("subject_id") in set(subject_ids)
                ]
            return [dict(row) for row in candidates[: params.get("limit", len(candidates))]]
        # 反向／交集读取：按对象或原文关键词匹配
        object_ids = set(params.get("object_ids") or [])
        keywords = [word for word in (params.get("object_keywords") or []) if word]
        matched = [
            row
            for row in candidates
            if (object_ids and row.get("object_id") in object_ids)
            or any(word in (row.get("evidence_quote") or "") for word in keywords)
        ]
        return [dict(row) for row in matched[: params.get("limit", len(matched))]]

    def close(self) -> None:  # pragma: no cover - parity with Neo4jStore
        """关闭底层 Driver 连接池。"""
        return None


class ScriptedLLM:
    """Deterministic LLMClient returning canned ClaimBatch objects in order.

    最后一次响应会被重复使用，便于测试重试循环。
    """

    def __init__(self, responses: list[ClaimBatch | BaseModel | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete_json(self, *, system: str, user: str, schema, purpose: str):
        """调用 LLM 并校验返回值为指定 JSON Schema。"""
        self.calls.append({"system": system, "user": user, "purpose": purpose})
        if not self._responses:
            return schema(claims=[])
        item = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(item, Exception):
            raise item
        return item


def batch(*claims) -> ClaimBatch:
    """测试夹具：批量写入高风险样例数据。"""
    return ClaimBatch(claims=list(claims))


__all__ = [
    "FakeHighRiskStore",
    "ScriptedLLM",
    "TASK_HIGH_RISK_PREDICATES",
    "assertion_row",
    "batch",
    "dosage_row",
]
