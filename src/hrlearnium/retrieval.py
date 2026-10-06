"""Deterministic BM25 search with optional cosine/RRF hybrid ranking.

All passages must already be restricted to the caller's tenant and course.
Scores rank potential evidence; they are not a claim that it answers a question.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from numbers import Real

from .text import meaningful_tokens


@dataclass(frozen=True)
class SearchPassage:
    id: str
    text: str
    chapter_number: int | None
    chapter_title: str
    section_kind: str
    embedding: list[float] | None = None


@dataclass(frozen=True)
class Candidate:
    passage: SearchPassage
    lexical_score: float
    semantic_score: float | None
    rank_score: float


def _unit_vector(vector: list[float], dimensions: int | None = None) -> list[float]:
    if not vector or (dimensions is not None and len(vector) != dimensions):
        raise ValueError("Embedding dimensions do not match or embedding is empty")
    if any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
        for value in vector
    ):
        raise ValueError("Embedding values must be finite numbers")
    # Scaling first avoids overflow for very large but finite input values.
    scale = max(abs(value) for value in vector)
    if not scale:
        raise ValueError("Embedding must have a nonzero norm")
    scaled = [value / scale for value in vector]
    norm = math.sqrt(sum(value * value for value in scaled))
    return [value / norm for value in scaled]


def retrieve(
    question: str,
    passages: list[SearchPassage],
    *,
    query_embedding: list[float] | None = None,
    limit: int = 8,
) -> list[Candidate]:
    """Rank evidence using BM25 alone or BM25/cosine reciprocal rank fusion.

    Missing passage embeddings are allowed. Invalid embeddings are rejected,
    never silently treated as semantic evidence. Without a query embedding,
    stored embeddings are ignored. Ties are resolved by stable passage ID.
    """
    if len({passage.id for passage in passages}) != len(passages):
        raise ValueError("Passage IDs must be unique")
    query_vector = _unit_vector(query_embedding) if query_embedding is not None else None
    passage_vectors: dict[str, list[float]] = {}
    if query_vector is not None:
        passage_vectors = {
            passage.id: _unit_vector(passage.embedding, len(query_vector))
            for passage in passages
            if passage.embedding is not None
        }
    if limit <= 0 or not passages:
        return []
    query_terms = Counter(meaningful_tokens(question))
    if not query_terms and query_vector is None:
        return []
    counts = [
        Counter(meaningful_tokens(passage.chapter_title + "\n" + passage.text))
        for passage in passages
    ]
    lengths = [sum(count.values()) for count in counts]
    avg_length = sum(lengths) / len(lengths) or 1.0
    frequencies = Counter(word for count in counts for word in count)
    lexical: dict[str, float] = {}
    semantic: dict[str, float | None] = {}
    for passage, count, length in zip(passages, counts, lengths, strict=True):
        score = 0.0
        for term, query_frequency in query_terms.items():
            frequency = count[term]
            if frequency:
                document_frequency = frequencies[term]
                inverse_frequency = math.log(
                    1 + (len(passages) - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                normalization = frequency + 1.5 * (1 - 0.75 + 0.75 * length / avg_length)
                score += (
                    inverse_frequency
                    * frequency
                    * 2.5
                    / normalization
                    * (1 + math.log(query_frequency))
                )
        lexical[passage.id] = score
        vector = passage_vectors.get(passage.id)
        semantic[passage.id] = (
            max(-1.0, min(1.0, sum(a * b for a, b in zip(query_vector, vector, strict=True))))
            if query_vector is not None and vector is not None
            else None
        )
    lexical_order = sorted(
        (identifier for identifier, score in lexical.items() if score > 0),
        key=lambda identifier: (-lexical[identifier], identifier),
    )
    if query_vector is None:
        by_id = {passage.id: passage for passage in passages}
        return [
            Candidate(by_id[identifier], lexical[identifier], None, lexical[identifier])
            for identifier in lexical_order[:limit]
        ]
    semantic_order = sorted(
        (identifier for identifier, score in semantic.items() if score is not None and score > 0),
        key=lambda identifier: (-semantic[identifier], identifier),
    )
    fused: Counter[str] = Counter()
    for order in (lexical_order, semantic_order):
        for rank, identifier in enumerate(order, start=1):
            fused[identifier] += 1.0 / (60 + rank)
    results = [
        Candidate(passage, lexical[passage.id], semantic[passage.id], fused[passage.id])
        for passage in passages
        if passage.id in fused
    ]
    return sorted(results, key=lambda item: (-item.rank_score, item.passage.id))[:limit]
