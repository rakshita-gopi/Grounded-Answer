"""A pointer from an answer back to a specific policy clause."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Citation:
    source_document: str
    clause_id: str
