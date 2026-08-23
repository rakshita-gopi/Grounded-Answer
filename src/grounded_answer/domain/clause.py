"""A single identifiable clause from a policy document."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PolicyClause:
    clause_id: str
    title: str
    content: str
    source_document: str
