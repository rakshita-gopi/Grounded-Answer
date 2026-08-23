"""Deterministic structure-aware retrieval over ingested Parts, Sections, and Clauses."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from grounded_answer.ingestion.models import ParsedClause, ParsedDocument
from grounded_answer.retrieval.base import Retriever
from grounded_answer.retrieval.clause_ids import extract_clause_ids
from grounded_answer.retrieval.models import RetrievalHit, RetrievalQuery

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)


@dataclass(frozen=True, slots=True)
class _IndexedClause:
    clause: ParsedClause
    part_title: str
    section_title: str
    order: int
    tokens: frozenset[str]


def _tokens(text: str) -> frozenset[str]:
    return frozenset(
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if token.lower() not in _STOPWORDS and len(token) > 1
    )


class DeterministicStructureRetriever(Retriever):
    """Lexical Part/Section/Clause retriever used when Ollama embeddings are unset.

    Retrieval walks the ingested Part/Section/Clause tree rather than arbitrary
    chunks. Scoring is deterministic: term overlap, structural title boosts,
    exact clause-id matches, then document order.
    """

    def __init__(self, document: ParsedDocument) -> None:
        self._document = document
        index: list[_IndexedClause] = []
        order = 0
        for part in document.parts:
            for section in part.sections:
                for clause in section.clauses:
                    blob = " ".join(
                        [clause.clause_id, part.title, section.title, clause.title, clause.content]
                    )
                    index.append(
                        _IndexedClause(
                            clause=clause,
                            part_title=part.title,
                            section_title=section.title,
                            order=order,
                            tokens=_tokens(blob),
                        )
                    )
                    order += 1
        self._index = tuple(index)

    def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievalHit]:
        query_tokens = _tokens(query.text)
        mentioned_ids = set(extract_clause_ids(query.text))
        ranked: list[tuple[int, int, _IndexedClause]] = []
        for item in self._index:
            score = len(query_tokens & item.tokens)
            if item.clause.clause_id in mentioned_ids:
                score += 100
            if score <= 0:
                continue
            ranked.append((score, item.order, item))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        selected = ranked[: query.top_k]
        return tuple(_to_hit(item.clause) for _, _, item in selected)


def _to_hit(clause: ParsedClause) -> RetrievalHit:
    return RetrievalHit(
        text=clause.content,
        source=clause.source_document,
        node_id=clause.clause_id,
        title=clause.title,
        clause_id=clause.clause_id,
    )
