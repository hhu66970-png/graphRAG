"""Small-corpus Chinese lexical retrieval used without a real embedding model."""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Iterable


def _tokens(text: str) -> list[str]:
    """Tokenize Chinese as overlapping 2-4 character grams plus Latin words."""
    compact = re.sub(r"\s+", "", text.lower())
    tokens: list[str] = []
    for run in re.findall(r"[\u3400-\u9fff]+", compact):
        for size in (2, 3, 4):
            tokens.extend(run[index:index + size] for index in range(len(run) - size + 1))
    tokens.extend(re.findall(r"[a-z0-9]+", compact))
    return tokens


def rank_chunks(
    question: str,
    chunks: Iterable[dict],
    *,
    exact_terms: Iterable[str] = (),
    top_k: int = 3,
) -> list[dict]:
    """Rank chunk dictionaries with BM25 and bonuses for mentioned graph entities."""
    documents = [dict(chunk) for chunk in chunks]
    if not documents:
        return []

    tokenized = [_tokens(str(document.get("text", ""))) for document in documents]
    question_counts = Counter(_tokens(question))
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))

    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized) or 1.0
    normalized_question = re.sub(r"\s+", "", question.lower())
    mentioned_terms = {
        re.sub(r"\s+", "", term.lower())
        for term in exact_terms
        if len(re.sub(r"\s+", "", term)) >= 2
        and re.sub(r"\s+", "", term.lower()) in normalized_question
    }

    scored: list[dict] = []
    total_documents = len(documents)
    k1 = 1.5
    b = 0.75
    for document, tokens in zip(documents, tokenized, strict=True):
        frequencies = Counter(tokens)
        document_length = len(tokens) or 1
        score = 0.0
        for token, query_frequency in question_counts.items():
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            df = document_frequency[token]
            inverse_document_frequency = math.log(
                1.0 + (total_documents - df + 0.5) / (df + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * document_length / average_length
            )
            score += (
                inverse_document_frequency
                * frequency
                * (k1 + 1.0)
                / denominator
                * min(query_frequency, 2)
            )

        normalized_text = re.sub(r"\s+", "", str(document.get("text", "")).lower())
        matched_terms = [term for term in mentioned_terms if term in normalized_text]
        score += sum(3.0 + len(term) for term in matched_terms)
        ranked = dict(document)
        ranked["score"] = score
        ranked["matched_terms"] = sorted(matched_terms, key=len, reverse=True)
        scored.append(ranked)

    scored.sort(key=lambda item: (-item["score"], item.get("index", 0)))
    positive = [item for item in scored if item["score"] > 0]
    return positive[:top_k] if positive else scored[:top_k]
