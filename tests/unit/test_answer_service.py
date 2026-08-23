from collections.abc import Sequence
from pathlib import Path

from grounded_answer.amendments.service import AmendmentIngestionService
from grounded_answer.application.answer_service import INSUFFICIENT_ANSWER, AnswerService
from grounded_answer.application.query_service import QueryService
from grounded_answer.domain.answer import GroundingStatus
from grounded_answer.domain.clause import PolicyClause
from grounded_answer.domain.question import Question
from grounded_answer.evidence.assembler import EvidenceAssembler
from grounded_answer.ingestion.service import IngestionService
from grounded_answer.llm.provider import StubLLMProvider
from grounded_answer.retrieval.base import Retriever
from grounded_answer.retrieval.models import RetrievalHit, RetrievalQuery
from grounded_answer.retrieval.service import RetrievalService
from grounded_answer.temporal.resolver import PolicyApplicabilityResolver


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

    def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievalHit]:
        return self.hits[: query.top_k]


def _hits(*clause_ids: str) -> tuple[RetrievalHit, ...]:
    return tuple(
        RetrievalHit(
            text=f"snippet for {clause_id}",
            source="policy-manual.md",
            clause_id=clause_id,
        )
        for clause_id in clause_ids
    )


def _answer_service(hits: Sequence[RetrievalHit], llm: StubLLMProvider) -> AnswerService:
    clauses = [_clause(hit.clause_id) for hit in hits if hit.clause_id]
    retrieval = RetrievalService(FakeRetriever(hits), EvidenceAssembler(clauses))
    return AnswerService(QueryService(retrieval), llm)


def test_query_service_returns_evidence_without_using_pageindex() -> None:
    retrieval = RetrievalService(
        FakeRetriever(_hits("§2.1.2")),
        EvidenceAssembler([_clause("§2.1.2")]),
    )
    evidence = QueryService(retrieval).evidence_for(Question(text="eligibility"))
    assert evidence[0].clause_id == "§2.1.2"
    assert evidence[0].content == "canonical content for §2.1.2"


def test_answer_service_generates_from_evidence() -> None:
    llm = StubLLMProvider(text="should not be used")
    eligibility = RetrievalHit(
        text="The conditions of eligibility are listed in this clause.",
        source="policy-manual.md",
        clause_id="§2.1.2",
    )
    resources = RetrievalHit(
        text="Countable resources must not exceed the limit.",
        source="policy-manual.md",
        clause_id="§2.4.1",
    )
    clauses = [
        _clause("§2.1.2", "The conditions of eligibility are listed in this clause."),
        _clause("§2.4.1", "Countable resources must not exceed the limit."),
    ]
    retrieval = RetrievalService(
        FakeRetriever((eligibility, resources)),
        EvidenceAssembler(clauses),
    )
    service = AnswerService(QueryService(retrieval), llm)
    result = service.answer(Question(text="What are the eligibility requirements?"))

    assert result.grounding_status == GroundingStatus.SUPPORTED
    assert "Part 2 of the policy" in result.text
    assert "Okay" not in result.text
    assert "The user is asking" not in result.text
    assert "[§2.1.2]" in result.text
    assert "§99.9.9" not in result.text
    assert [citation.clause_id for citation in result.citations] == ["§2.1.2"]
    assert result.citations[0].source_document == "policy-manual.md"
    assert llm.calls == []


def test_answer_service_abstains_when_evidence_is_missing() -> None:
    llm = StubLLMProvider(text="should not be used")
    service = _answer_service((), llm)
    result = service.answer(Question(text="What is the capital of France?"))

    assert result.grounding_status == GroundingStatus.INSUFFICIENT
    assert result.text == INSUFFICIENT_ANSWER
    assert result.citations == ()
    assert llm.calls == []


def test_answer_service_abstains_when_retrieved_evidence_is_off_topic() -> None:
    llm = StubLLMProvider(text="should not be used")
    service = _answer_service(_hits("§2.1.2"), llm)
    result = service.answer(Question(text="What is the capital of France?"))

    assert result.grounding_status == GroundingStatus.INSUFFICIENT
    assert result.text == INSUFFICIENT_ANSWER
    assert result.citations == ()
    assert llm.calls == []


def test_answer_service_abstains_when_llm_cites_only_invented_clauses() -> None:
    llm = StubLLMProvider(text="The secret rule is in §99.9.9.")
    eligibility = RetrievalHit(
        text="The conditions of eligibility are listed in this clause.",
        source="policy-manual.md",
        clause_id="§2.1.2",
    )
    retrieval = RetrievalService(
        FakeRetriever((eligibility,)),
        EvidenceAssembler(
            [_clause("§2.1.2", "The conditions of eligibility are listed in this clause.")]
        ),
    )
    service = AnswerService(QueryService(retrieval), llm)
    result = service.answer(Question(text="What does this clause list?"))

    assert result.grounding_status == GroundingStatus.INSUFFICIENT
    assert result.text == INSUFFICIENT_ANSWER
    assert result.citations == ()
    assert "§99.9.9" not in result.text
    assert llm.calls


def test_answer_service_strips_reasoning_and_cites_only_supporting_clauses() -> None:
    llm = StubLLMProvider(
        text=(
            "Okay, let me tackle this. The user is asking about listed conditions. "
            "The clause lists the conditions of eligibility."
        )
    )
    eligibility = RetrievalHit(
        text="The conditions of eligibility are listed in this clause.",
        source="policy-manual.md",
        clause_id="§2.1.2",
    )
    resources = RetrievalHit(
        text="Countable resources must not exceed $4,000.",
        source="policy-manual.md",
        clause_id="§2.4.1",
    )
    retrieval = RetrievalService(
        FakeRetriever((eligibility, resources)),
        EvidenceAssembler(
            [
                _clause("§2.1.2", "The conditions of eligibility are listed in this clause."),
                _clause("§2.4.1", "Countable resources must not exceed $4,000."),
            ]
        ),
    )
    service = AnswerService(QueryService(retrieval), llm)
    result = service.answer(Question(text="What does this clause list?"))

    assert result.grounding_status == GroundingStatus.SUPPORTED
    assert "Okay" not in result.text
    assert "The user is asking" not in result.text
    assert "lists the conditions of eligibility" in result.text
    assert "[§2.1.2]" in result.text
    assert "[§2.4.1]" not in result.text
    assert llm.calls


def test_off_topic_does_not_use_missing_date_when_amended_clauses_are_retrieved(
    corpus_dir: Path,
) -> None:
    policy = IngestionService(corpus_dir).load_policy()
    amendment = AmendmentIngestionService().load_amendment()
    hits = (
        RetrievalHit(
            text="The first £120 of monthly earnings is disregarded.",
            source="policy-manual.md",
            clause_id="§6.4.1",
        ),
        RetrievalHit(
            text="The conditions of eligibility are listed in this clause.",
            source="policy-manual.md",
            clause_id="§2.1.2",
        ),
    )
    retrieval = RetrievalService(FakeRetriever(hits), EvidenceAssembler(policy.clauses))
    service = AnswerService(
        QueryService(retrieval),
        StubLLMProvider(text="should not be used"),
        applicability_resolver=PolicyApplicabilityResolver(policy, amendment),
    )
    result = service.answer(Question(text="What is the boiling point of helium?"))
    assert result.grounding_status == GroundingStatus.INSUFFICIENT
    assert result.text == INSUFFICIENT_ANSWER
    assert result.citations == ()
