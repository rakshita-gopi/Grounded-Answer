"""An issued amendment, kept separate from the original policy corpus."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from grounded_answer.domain.policy_change import PolicyChange


@dataclass(frozen=True, slots=True)
class AmendmentParagraph:
    paragraph_id: str
    title: str
    content: str
    source_document: str
    section_number: int


@dataclass(frozen=True, slots=True)
class Amendment:
    amendment_id: str
    source_document: str
    issued_date: date
    effective_date: date
    changes: tuple[PolicyChange, ...]
    paragraphs: tuple[AmendmentParagraph, ...]
