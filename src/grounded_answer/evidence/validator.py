"""Deterministic checks on assembled Evidence. Does not generate answers."""

from __future__ import annotations

import re
from collections.abc import Container, Sequence

from grounded_answer.domain.evidence import Evidence

CLAUSE_ID_FORMAT = re.compile(r"^§\d+\.\d+\.\d+$")


class EvidenceValidator:
    def __init__(self, known_clause_ids: Container[str] | None = None) -> None:
        self._known_clause_ids = known_clause_ids

    def validate(self, evidence: Sequence[Evidence]) -> tuple[Evidence, ...]:
        kept: list[Evidence] = []
        seen: set[str] = set()
        for item in evidence:
            if not item.content.strip() or not item.source.strip():
                continue
            if not CLAUSE_ID_FORMAT.match(item.clause_id):
                continue
            if self._known_clause_ids is not None and item.clause_id not in self._known_clause_ids:
                continue
            if item.clause_id in seen:
                continue
            seen.add(item.clause_id)
            kept.append(item)
        return tuple(kept)
