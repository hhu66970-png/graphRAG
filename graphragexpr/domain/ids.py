"""Stable identifier construction (总体设计 §5).

所有 ID 必须可复现：相同输入永远得到相同 ID。
禁止使用数组顺序、Neo4j 内部 elementId 或自增序号作为跨版本外部 ID（SKILL §7.1）。
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid

_WHITESPACE_RE = re.compile(r"\s+")
# 全角→半角常见标点，OCR 常见噪声
_PUNCT_MAP = str.maketrans({
    "，": ",", "。": ".", "；": ";", "：": ":", "（": "(", "）": ")",
    "％": "%", "－": "-", "～": "~", "　": " ",
})


def normalize_text(value: str) -> str:
    """Normalize text for hashing and matching.

    只用于生成稳定 ID 与做规范化匹配；**不得覆盖原始证据文本**（SKILL §7.2）。
    """
    if not value:
        return ""
    text = value.translate(_PUNCT_MAP)
    text = _WHITESPACE_RE.sub("", text)
    return text.strip().lower()


def _sha(*parts: object, length: int) -> str:
    """对规范化文本做短哈希，用于稳定 ID。"""
    joined = "␟".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


def document_id(file_bytes_hash: str, edition: str) -> str:
    """document_id = sha256(file_bytes + edition)[:20]"""
    return _sha(file_bytes_hash, edition, length=20)


def section_id(doc_id: str, section_path: list[str]) -> str:
    """由章节路径生成稳定章节 ID。"""
    return _sha(doc_id, "/".join(section_path), length=20)


def chunk_id(doc_id: str, section_path: list[str], normalized_content: str) -> str:
    """chunk_id = sha256(document_id + section_path + normalized_content)[:24]

    内容变了 chunk_id 就变，因此可直接用作抽取缓存键的一部分。
    """
    return _sha(doc_id, "/".join(section_path), normalized_content, length=24)


def mention_id(chunk: str, start: int, end: int, surface: str) -> str:
    """为一次实体提及生成稳定 ID。"""
    return _sha(chunk, start, end, surface, length=24)


def canonical_id(entity_type: str, key: str) -> str:
    """canonical_id = TYPE + ':' + stable_hash(normalized_key).

    身份只认 canonical_id，不认展示名（SKILL §3.3）。key 通常是规范化后的
    首选名；一旦某实体确定了 canonical_id，改展示名不会改变身份。
    """
    return f"{entity_type}:{_sha(entity_type, normalize_text(key), length=16)}"


def fact_key(
    subject_id: str,
    predicate: str,
    object_ref: str,
    qualifiers: dict | None = None,
    polarity: str = "POSITIVE",
) -> str:
    """同一事实的稳定键；多条证据可指向同一 fact_key（总体设计 §4.1）。"""
    normalized_qualifiers = ";".join(
        f"{k}={normalize_text(str(v))}" for k, v in sorted((qualifiers or {}).items())
    )
    return _sha(subject_id, predicate, object_ref, normalized_qualifiers, polarity, length=24)


def assertion_id(fact: str, chunk: str, span: tuple[int | None, int | None], schema_version: str) -> str:
    """为一条断言生成稳定 ID。"""
    return _sha(fact, chunk, span[0], span[1], schema_version, length=24)


def new_run_id() -> str:
    """时间前缀 + 随机后缀，单调可排序（SKILL §7.1 要求 ULID/UUIDv7 类）。"""
    return f"{int(time.time() * 1000):013d}-{uuid.uuid4().hex[:12]}"
