"""Deterministic BM25 retrieval with inspectable scores and citations."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Protocol

from offerdelta.domain.common.errors import ValidationError
from offerdelta.policy.corpus import PolicyChunk, load_policy_corpus

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
MAX_RETRIEVAL_LIMIT: Final = 10
_STOPWORDS: Final = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "before",
        "do",
        "for",
        "i",
        "in",
        "is",
        "of",
        "on",
        "our",
        "the",
        "to",
        "what",
        "when",
        "with",
    }
)


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS)


@dataclass(frozen=True)
class PolicyMatch:
    chunk: PolicyChunk
    score: float
    matched_terms: tuple[str, ...]

    def citation(self, rank: int) -> dict[str, str | int | list[str] | None]:
        payload: dict[str, str | int | list[str] | None] = dict(self.chunk.citation())
        payload.update(
            {
                "rank": rank,
                "score": f"{self.score:.6f}",
                "matched_terms": list(self.matched_terms),
            }
        )
        return payload


class PolicyRetriever(Protocol):
    """Port implemented by sparse retrieval now and a future tenant-aware adapter."""

    corpus_version: str

    def search(self, query: str, *, limit: int = 3) -> tuple[PolicyMatch, ...]: ...


class BM25PolicyRetriever:
    """A small, dependency-free sparse retriever suitable for deterministic tests."""

    def __init__(
        self,
        chunks: Sequence[PolicyChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValidationError("policy retrieval needs at least one chunk")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValidationError("invalid BM25 parameters")
        versions = {chunk.corpus_version for chunk in chunks}
        if len(versions) != 1:
            raise ValidationError("all policy chunks must use one corpus version")
        self.chunks = tuple(chunks)
        self.corpus_version = next(iter(versions))
        self.k1 = k1
        self.b = b
        self._tokens = tuple(tokenize(chunk.searchable_text) for chunk in self.chunks)
        self._frequencies = tuple(Counter(tokens) for tokens in self._tokens)
        self._average_length = sum(map(len, self._tokens)) / len(self._tokens)
        self._document_frequency = Counter(term for tokens in self._tokens for term in set(tokens))

    def search(self, query: str, *, limit: int = 3) -> tuple[PolicyMatch, ...]:
        query_terms = tuple(dict.fromkeys(tokenize(query)))
        if not query_terms:
            return ()
        if not 1 <= limit <= MAX_RETRIEVAL_LIMIT:
            raise ValidationError("policy retrieval limit must be between 1 and 10")
        scored: list[PolicyMatch] = []
        for index, chunk in enumerate(self.chunks):
            frequencies = self._frequencies[index]
            matched = tuple(term for term in query_terms if frequencies[term] > 0)
            if not matched:
                continue
            score = sum(self._term_score(term, index) for term in matched)
            if score > 0:
                scored.append(PolicyMatch(chunk=chunk, score=score, matched_terms=matched))
        scored.sort(key=lambda match: (-match.score, match.chunk.chunk_id))
        return tuple(scored[:limit])

    def _term_score(self, term: str, document_index: int) -> float:
        frequency = self._frequencies[document_index][term]
        document_frequency = self._document_frequency[term]
        document_count = len(self.chunks)
        inverse_document_frequency = math.log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        document_length = len(self._tokens[document_index])
        normalizer = frequency + self.k1 * (
            1 - self.b + self.b * document_length / self._average_length
        )
        return inverse_document_frequency * frequency * (self.k1 + 1) / normalizer


@lru_cache(maxsize=1)
def default_policy_retriever() -> BM25PolicyRetriever:
    return BM25PolicyRetriever(load_policy_corpus())
