"""Evidence-preserving extractive map/reduce for comparison questions."""

from __future__ import annotations

import re
from collections import defaultdict

from ..local_retrieval import rank_chunks

_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？；])")


def _best_source_sentence(question: str, entity_name: str, text: str) -> str:
    """从章节文本中抽出最能支撑对比槽的原句。"""
    sentences = [
        part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()
    ]
    if not sentences:
        return text.strip()
    chinese = "".join(re.findall(r"[\u3400-\u9fff]+", question))
    terms = {chinese[index : index + 2] for index in range(max(0, len(chinese) - 1))}

    def score(sentence: str) -> tuple[int, int]:
        """给对比摘要句打分。"""
        overlap = sum(term in sentence for term in terms)
        return (10 * int(entity_name in sentence) + overlap, -len(sentence))

    return max(sentences, key=score)


def reduce_section_summaries(
    question: str,
    rows: list[dict],
    *,
    per_entity: int = 3,
    total_limit: int = 8,
) -> list[dict]:
    """Map source chunks per entity, then reduce them with balanced round-robin merging.

    The output quote is always an exact substring of its source Chunk. This is an
    extractive comparison aid, not an LLM-authored claim.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    entity_order: list[str] = []
    seen_chunks: set[tuple[str, str]] = set()
    for row in rows:
        entity_id = str(row.get("entity_id") or "")
        chunk_id = str(row.get("chunk_id") or "")
        if not entity_id or not chunk_id or (entity_id, chunk_id) in seen_chunks:
            continue
        seen_chunks.add((entity_id, chunk_id))
        if entity_id not in grouped:
            entity_order.append(entity_id)
        grouped[entity_id].append(dict(row))

    mapped: dict[str, list[dict]] = {}
    for entity_id in entity_order:
        candidates = grouped[entity_id]
        entity_name = str(candidates[0].get("entity_name") or "")
        ranked = rank_chunks(
            question,
            candidates,
            exact_terms=[entity_name],
            top_k=per_entity,
        )
        mapped[entity_id] = []
        for row in ranked:
            quote = _best_source_sentence(
                question, entity_name, str(row.get("text") or "")
            )
            mapped[entity_id].append(
                {
                    "evidence_id": f"chunk:{row['chunk_id']}:{entity_id}",
                    "subject_id": entity_id,
                    "predicate": None,
                    "object_id": None,
                    "fact_key": None,
                    "quote": quote,
                    "chunk_ids": [row["chunk_id"]],
                    "page_start": row.get("page_start") or 0,
                    "page_end": row.get("page_end") or row.get("page_start") or 0,
                    "support_score": 1.0,
                    "retrieval_score": float(row.get("score") or 0.0),
                    "qualifiers": {
                        "summary_kind": "extractive_section",
                        "entity_name": entity_name,
                        "section_path": row.get("section_path") or [],
                    },
                }
            )

    reduced: list[dict] = []
    rank = 0
    while len(reduced) < total_limit:
        added = False
        for entity_id in entity_order:
            if rank < len(mapped[entity_id]):
                reduced.append(mapped[entity_id][rank])
                added = True
                if len(reduced) >= total_limit:
                    break
        if not added:
            break
        rank += 1
    return reduced
