"""Stage A integration: question → retrieval → evidence → answer."""

from pathlib import Path

from grounded_answer.application.answer_service import INSUFFICIENT_ANSWER
from grounded_answer.application.bootstrap import create_answer_service
from grounded_answer.domain.answer import GroundingStatus
from grounded_answer.domain.question import Question
from grounded_answer.ingestion.service import IngestionService
from grounded_answer.llm.provider import StubLLMProvider


def test_eligibility_question_returns_supported_answer_with_real_clauses(
    corpus_dir: Path,
) -> None:
    policy = IngestionService(corpus_dir).load_policy()
    known_ids = {clause.clause_id for clause in policy.clauses}
    service = create_answer_service(corpus_dir=corpus_dir, environ={}, load_dotenv=False)
    result = service.answer(Question(text="What are the eligibility requirements?"))

    assert result.grounding_status is GroundingStatus.SUPPORTED
    assert result.citations
    assert result.text
    assert "Part 2 of the policy" in result.text
    assert "Okay" not in result.text
    assert "The user is asking" not in result.text
    for citation in result.citations:
        assert citation.clause_id in known_ids
        assert citation.source_document == "policy-manual.md"


def test_resource_limit_question_retrieves_clause_2_4_1(corpus_dir: Path) -> None:
    llm = StubLLMProvider(
        text="A household is not eligible where countable resources exceed $4,000. §2.4.1"
    )
    service = create_answer_service(
        corpus_dir=corpus_dir,
        environ={},
        load_dotenv=False,
        llm=llm,
    )
    result = service.answer(
        Question(text="What is the countable resources limit for a household?")
    )

    assert result.grounding_status is GroundingStatus.SUPPORTED
    cited = [citation.clause_id for citation in result.citations]
    assert "§2.4.1" in cited, cited
    assert "$4,000" in result.text


def test_unsupported_question_abstains(corpus_dir: Path) -> None:
    service = create_answer_service(corpus_dir=corpus_dir, environ={}, load_dotenv=False)
    result = service.answer(Question(text="What is the boiling point of helium?"))

    assert result.grounding_status is GroundingStatus.INSUFFICIENT
    assert result.text == INSUFFICIENT_ANSWER
    assert result.citations == ()


def test_invented_clause_citation_is_rejected(corpus_dir: Path) -> None:
    llm = StubLLMProvider(text="The secret threshold is in §99.9.9.")
    service = create_answer_service(
        corpus_dir=corpus_dir,
        environ={},
        load_dotenv=False,
        llm=llm,
    )
    result = service.answer(Question(text="What does this clause list?"))

    assert result.grounding_status is GroundingStatus.INSUFFICIENT
    assert result.citations == ()
    assert "§99.9.9" not in result.text
