"""Retrieved policy text used as grounding for an answer."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    clause_id: str
    content: str
    source: str
