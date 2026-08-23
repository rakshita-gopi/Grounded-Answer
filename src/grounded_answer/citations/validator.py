"""Ensure answer citations point at retrieved evidence and real policy clauses."""

from __future__ import annotations

import re
from collections.abc import Container, Sequence
from dataclasses import dataclass

from grounded_answer.domain.citation import Citation
from grounded_answer.domain.evidence import Evidence
from grounded_answer.retrieval.clause_ids import extract_clause_ids


@dataclass(frozen=True, slots=True)
class CitationValidation:
    text: str
    citations: tuple[Citation, ...]
    invented_clause_ids: tuple[str, ...]
    accepted: bool


class CitationValidator:
    """Reject clause IDs that are not in retrieved evidence or the policy corpus.

    Invented identifiers such as §99.9.9 are dropped. If the model only cited
    invented IDs, the answer is not accepted.
    """

    def __init__(self, known_clause_ids: Container[str] | None = None) -> None:
        self._known_clause_ids = known_clause_ids

    def validate(self, answer_text: str, evidence: Sequence[Evidence]) -> CitationValidation:
        evidence_by_id = {item.clause_id: item for item in evidence}
        cited = extract_clause_ids(answer_text, require_section_sign=True)
        allowed: list[Citation] = []
        invented: list[str] = []
        seen: set[str] = set()
        for clause_id in cited:
            if clause_id in seen:
                continue
            seen.add(clause_id)
            item = evidence_by_id.get(clause_id)
            if item is None:
                invented.append(clause_id)
                continue
            if self._known_clause_ids is not None and clause_id not in self._known_clause_ids:
                invented.append(clause_id)
                continue
            allowed.append(Citation(source_document=item.source, clause_id=clause_id))

        if cited:
            text = _strip_clause_ids(answer_text, invented)
            accepted = bool(allowed)
            citations = tuple(allowed)
        else:
            text = answer_text
            accepted = True
            citations = tuple(
                Citation(source_document=item.source, clause_id=item.clause_id)
                for item in evidence
            )
        return CitationValidation(
            text=text,
            citations=citations,
            invented_clause_ids=tuple(invented),
            accepted=accepted,
        )


def _strip_clause_ids(text: str, clause_ids: Sequence[str]) -> str:
    result = text
    for clause_id in sorted(clause_ids, key=len, reverse=True):
        result = result.replace(clause_id, "")
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r" +([,.;:])", r"\1", result)
    return result.strip()
