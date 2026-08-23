"""Score a Stage A evaluation answer against expected.json."""

from __future__ import annotations

from collections.abc import Container, Sequence

from grounded_answer.application.answer_service import INSUFFICIENT_ANSWER
from grounded_answer.domain.answer import Answer, GroundingStatus


def score_answer(
    answer: Answer,
    expected: dict,
    known_clause_ids: Container[str],
) -> dict[str, bool]:
    cited_ids = [citation.clause_id for citation in answer.citations]
    expected_abstain = bool(expected["abstain"])
    required = list(expected.get("required_clause_ids") or [])
    must_contain = list(expected.get("must_contain") or [])

    abstention_ok = (answer.grounding_status is GroundingStatus.INSUFFICIENT) == expected_abstain
    if expected_abstain:
        answer_ok = answer.text.strip() == INSUFFICIENT_ANSWER
        evidence_ok = cited_ids == []
    else:
        answer_ok = all(phrase in answer.text for phrase in must_contain)
        evidence_ok = all(clause_id in cited_ids for clause_id in required)

    citation_ok = all(clause_id in known_clause_ids for clause_id in cited_ids)
    return {
        "answer": answer_ok,
        "evidence": evidence_ok,
        "citation": citation_ok,
        "abstention": abstention_ok,
    }


def percentage(correct: int, total: int) -> str:
    if total == 0:
        return "0%"
    value = 100.0 * correct / total
    if value.is_integer():
        return f"{int(value)}%"
    return f"{value:.1f}%"


def summarize(rows: Sequence[dict[str, bool]]) -> dict[str, str]:
    total = len(rows)
    return {
        "total": str(total),
        "answer": percentage(sum(1 for row in rows if row["answer"]), total),
        "evidence": percentage(sum(1 for row in rows if row["evidence"]), total),
        "citation": percentage(sum(1 for row in rows if row["citation"]), total),
        "abstention": percentage(sum(1 for row in rows if row["abstention"]), total),
    }
