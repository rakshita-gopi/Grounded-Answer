from collections.abc import Sequence

from grounded_answer.domain.evidence import Evidence
from grounded_answer.domain.question import Question
from grounded_answer.retrieval.base import Retriever
from grounded_answer.retrieval.models import RetrievalQuery
from grounded_answer.retrieval.service import RetrievalService


class FakeRetriever(Retriever):
    def __init__(self, evidence: Sequence[Evidence]) -> None:
        self.evidence = tuple(evidence)
        self.queries: list[RetrievalQuery] = []

    def retrieve(self, query: RetrievalQuery) -> Sequence[Evidence]:
        self.queries.append(query)
        return self.evidence[: query.top_k]


def _sample_evidence(*clause_ids: str) -> tuple[Evidence, ...]:
    return tuple(
        Evidence(
            evidence_id=f"ev-{index}",
            clause_id=clause_id,
            content=f"content for {clause_id}",
            source="policy-manual.md",
        )
        for index, clause_id in enumerate(clause_ids, start=1)
    )


def test_service_returns_evidence_from_injected_retriever() -> None:
    retriever = FakeRetriever(_sample_evidence("§2.1.2", "§2.4.1"))
    service = RetrievalService(retriever)

    result = service.retrieve(Question(text="What are the eligibility requirements?"))

    assert [item.clause_id for item in result] == ["§2.1.2", "§2.4.1"]
    assert retriever.queries[0].text == "What are the eligibility requirements?"


def test_service_forwards_top_k_to_retriever() -> None:
    retriever = FakeRetriever(_sample_evidence("§2.1.2", "§2.4.1", "§6.6.1"))
    service = RetrievalService(retriever)

    result = service.retrieve(Question(text="income thresholds"), top_k=2)

    assert retriever.queries[0].top_k == 2
    assert [item.clause_id for item in result] == ["§2.1.2", "§2.4.1"]


def test_service_does_not_require_a_pageindex_type() -> None:
    retriever = FakeRetriever(_sample_evidence("§1.4.1"))
    service = RetrievalService(retriever)
    result = service.retrieve(Question(text="What is an applicant?"))
    assert result[0].source == "policy-manual.md"
