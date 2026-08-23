"""Convert raw retrieval hits into canonical Evidence objects."""

from __future__ import annotations

from collections.abc import Sequence

from grounded_answer.domain.clause import PolicyClause
from grounded_answer.domain.evidence import Evidence
from grounded_answer.retrieval.clause_ids import extract_clause_ids
from grounded_answer.retrieval.models import RetrievalHit


class EvidenceAssembler:
    """Map retriever hits onto ingested policy clauses.

    This layer does not generate answers.
    """

    def __init__(self, clauses: Sequence[PolicyClause]) -> None:
        self._clauses = {clause.clause_id: clause for clause in clauses}

    def assemble(self, hits: Sequence[RetrievalHit]) -> tuple[Evidence, ...]:
        evidence: list[Evidence] = []
        seen: set[str] = set()
        for hit in hits:
            for clause_id in self._clause_ids_for(hit):
                clause = self._clauses.get(clause_id)
                if clause is None or clause.clause_id in seen:
                    continue
                seen.add(clause.clause_id)
                evidence.append(_evidence_from_clause(clause))
        return tuple(evidence)

    def _clause_ids_for(self, hit: RetrievalHit) -> tuple[str, ...]:
        if hit.clause_id:
            return (hit.clause_id,)
        blob = f"{hit.title}\n{hit.text}"
        signed = extract_clause_ids(blob, require_section_sign=True)
        if signed:
            return signed
        return extract_clause_ids(blob, require_section_sign=False)


def _evidence_from_clause(clause: PolicyClause) -> Evidence:
    return Evidence(
        evidence_id=f"{clause.source_document}:{clause.clause_id}",
        clause_id=clause.clause_id,
        content=clause.content,
        source=clause.source_document,
    )
