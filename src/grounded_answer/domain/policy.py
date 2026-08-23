"""Policy document as represented in the original corpus."""

from __future__ import annotations

from dataclasses import dataclass

from grounded_answer.domain.clause import PolicyClause


@dataclass(frozen=True, slots=True)
class Policy:
    document_id: str
    title: str
    document_type: str
    authority: str
    source_document: str
    clauses: tuple[PolicyClause, ...]
