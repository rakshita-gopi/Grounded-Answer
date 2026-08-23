"""Deterministic check: is retrieved evidence sufficient to generate an answer?"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from grounded_answer.domain.answer import GroundingStatus
from grounded_answer.domain.evidence import Evidence
from grounded_answer.domain.question import Question
from grounded_answer.retrieval.clause_ids import extract_clause_ids

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
class GroundingAssessment:
    status: GroundingStatus
    evidence: tuple[Evidence, ...]
    reason: str


def _tokens(text: str) -> frozenset[str]:
    return frozenset(
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if token.lower() not in _STOPWORDS and len(token) > 1
    )


def _related(left: str, right: str) -> bool:
    if left == right:
        return True
    if min(len(left), len(right)) >= 5 and (
        left.startswith(right[:5]) or right.startswith(left[:5])
    ):
        return True
    return False


def _clause_supports_question(query_tokens: frozenset[str], item: Evidence) -> bool:
    evidence_tokens = _tokens(f"{item.clause_id} {item.content}")
    for query_token in query_tokens:
        for evidence_token in evidence_tokens:
            if _related(query_token, evidence_token):
                return True
    return False


class GroundingValidator:
    """Decide SUPPORTED vs INSUFFICIENT before the LLM is called.

    Empty or off-topic evidence yields INSUFFICIENT (abstain). Matching clause
    IDs or overlapping policy terms yield SUPPORTED. No extra statuses.
    """

    def assess(
        self,
        question: Question,
        evidence: Sequence[Evidence],
    ) -> GroundingAssessment:
        if not evidence:
            return GroundingAssessment(
                status=GroundingStatus.INSUFFICIENT,
                evidence=(),
                reason="no evidence retrieved",
            )

        mentioned_ids = set(extract_clause_ids(question.text))
        if mentioned_ids:
            matched = tuple(item for item in evidence if item.clause_id in mentioned_ids)
            if matched:
                return GroundingAssessment(
                    status=GroundingStatus.SUPPORTED,
                    evidence=matched,
                    reason="question cites retrieved clauses",
                )

        query_tokens = _tokens(question.text)
        if not query_tokens:
            return GroundingAssessment(
                status=GroundingStatus.INSUFFICIENT,
                evidence=(),
                reason="question has no content tokens",
            )

        supporting = tuple(
            item for item in evidence if _clause_supports_question(query_tokens, item)
        )
        if supporting:
            return GroundingAssessment(
                status=GroundingStatus.SUPPORTED,
                evidence=supporting,
                reason="retrieved clauses overlap the question",
            )
        return GroundingAssessment(
            status=GroundingStatus.INSUFFICIENT,
            evidence=(),
            reason="retrieved clauses do not support the question",
        )
