from collections.abc import Sequence

from grounded_answer.domain.clause import PolicyClause
from grounded_answer.domain.question import Question
from grounded_answer.evidence.assembler import EvidenceAssembler
from grounded_answer.evidence.validator import EvidenceValidator
from grounded_answer.retrieval.base import Retriever
from grounded_answer.retrieval.models import RetrievalHit, RetrievalQuery
from grounded_answer.retrieval.service import RetrievalService


def _clause(clause_id: str, content: str | None = None) -> PolicyClause:
    return PolicyClause(
        clause_id=clause_id,
        title=clause_id,
        content=content or f"canonical content for {clause_id}",
        source_document="policy-manual.md",
    )


class FakeRetriever(Retriever):
    def __init__(self, hits: Sequence[RetrievalHit]) -> None:
        self.hits = tuple(hits)
        self.queries: list[RetrievalQuery] = []

    def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievalHit]:
        self.queries.append(query)
        return self.hits[: query.top_k]


def _sample_hits(*clause_ids: str) -> tuple[RetrievalHit, ...]:
    return tuple(
        RetrievalHit(
            text=f"snippet for {clause_id}",
            source="policy-manual.md",
            node_id=clause_id,
            clause_id=clause_id,
        )
        for clause_id in clause_ids
    )


def test_service_returns_evidence_from_injected_retriever() -> None:
    assembler = EvidenceAssembler([_clause("§2.1.2"), _clause("§2.4.1")])
    retriever = FakeRetriever(_sample_hits("§2.1.2", "§2.4.1"))
    service = RetrievalService(retriever, assembler)

    result = service.retrieve(Question(text="What are the eligibility requirements?"))

    assert [item.clause_id for item in result] == ["§2.1.2", "§2.4.1"]
    assert retriever.queries[0].text == "What are the eligibility requirements?"
    assert result[0].content == "canonical content for §2.1.2"


def test_service_forwards_top_k_to_retriever() -> None:
    assembler = EvidenceAssembler([_clause("§2.1.2"), _clause("§2.4.1"), _clause("§6.6.1")])
    retriever = FakeRetriever(_sample_hits("§2.1.2", "§2.4.1", "§6.6.1"))
    service = RetrievalService(retriever, assembler)

    result = service.retrieve(Question(text="income thresholds"), top_k=2)

    assert retriever.queries[0].top_k == 2
    assert [item.clause_id for item in result] == ["§2.1.2", "§2.4.1"]


def test_service_accepts_any_retriever() -> None:
    assembler = EvidenceAssembler([_clause("§1.4.1")])
    retriever = FakeRetriever(_sample_hits("§1.4.1"))
    service = RetrievalService(retriever, assembler)
    result = service.retrieve(Question(text="What is an applicant?"))
    assert result[0].source == "policy-manual.md"
