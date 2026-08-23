from pathlib import Path

from grounded_answer.application.answer_service import (
    INSUFFICIENT_ANSWER,
    MISSING_DATE_ANSWER,
)
from grounded_answer.application.bootstrap import create_answer_service
from grounded_answer.domain.answer import GroundingStatus
from grounded_answer.domain.question import Question
from grounded_answer.ingestion.service import IngestionService
from grounded_answer.llm.provider import StubLLMProvider


def test_stage_a_eligibility_still_supported(corpus_dir: Path) -> None:
    policy = IngestionService(corpus_dir).load_policy()
    known_ids = {clause.clause_id for clause in policy.clauses}
    service = create_answer_service(corpus_dir=corpus_dir, environ={}, load_dotenv=False)
    result = service.answer(Question(text="What are the eligibility requirements?"))
    assert result.grounding_status is GroundingStatus.SUPPORTED
    for citation in result.citations:
        assert citation.clause_id in known_ids or citation.clause_id.startswith("¶")


def test_february_determination_uses_original_disregard(corpus_dir: Path) -> None:
    llm = StubLLMProvider(text="The first monthly earnings disregard is $120 per month. §6.4.1")
    service = create_answer_service(
        corpus_dir=corpus_dir,
        environ={},
        load_dotenv=False,
        llm=llm,
    )
    result = service.answer(
        Question(text="For a determination made on 15 February 2026, what is the first monthly earnings disregard?")
    )
    assert result.grounding_status is GroundingStatus.SUPPORTED
    assert any(citation.clause_id == "§6.4.1" for citation in result.citations)
    prompt, context = llm.calls[0]
    evidence_text = " ".join(item.content for item in context.evidence)
    assert "$120 per month" in evidence_text
    assert "$175 per month" not in evidence_text


def test_march_determination_uses_amended_disregard(corpus_dir: Path) -> None:
    llm = StubLLMProvider(text="The first monthly earnings disregard is $175 per month. §6.4.1")
    service = create_answer_service(
        corpus_dir=corpus_dir,
        environ={},
        load_dotenv=False,
        llm=llm,
    )
    result = service.answer(
        Question(text="For a determination made on 15 March 2026, what is the first monthly earnings disregard?")
    )
    assert result.grounding_status is GroundingStatus.SUPPORTED
    prompt, context = llm.calls[0]
    evidence_text = " ".join(item.content for item in context.evidence)
    assert "$175 per month" in evidence_text
    assert "$120 per month" not in evidence_text


def test_missing_date_does_not_guess(corpus_dir: Path) -> None:
    service = create_answer_service(corpus_dir=corpus_dir, environ={}, load_dotenv=False)
    result = service.answer(Question(text="What is the first monthly earnings disregard?"))
    assert result.grounding_status is GroundingStatus.INSUFFICIENT
    assert result.text == MISSING_DATE_ANSWER
    assert result.citations == ()


def test_unsupported_question_still_abstains(corpus_dir: Path) -> None:
    service = create_answer_service(corpus_dir=corpus_dir, environ={}, load_dotenv=False)
    result = service.answer(Question(text="What is the boiling point of helium?"))
    assert result.grounding_status is GroundingStatus.INSUFFICIENT
    assert result.text == INSUFFICIENT_ANSWER


def test_inserted_clause_available_after_effective_date(corpus_dir: Path) -> None:
    llm = StubLLMProvider(
        text="A sanction must not be imposed where the change would have increased the award. §10.5.3A"
    )
    service = create_answer_service(
        corpus_dir=corpus_dir,
        environ={},
        load_dotenv=False,
        llm=llm,
    )
    result = service.answer(
        Question(
            text=(
                "For a determination made on 15 March 2026, may a sanction be imposed "
                "for a failure to report a change of circumstances that would have increased the award?"
            )
        )
    )
    assert result.grounding_status is GroundingStatus.SUPPORTED
    assert any(citation.clause_id == "§10.5.3A" for citation in result.citations)
