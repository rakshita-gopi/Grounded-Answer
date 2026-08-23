from grounded_answer.domain.evidence import Evidence
from grounded_answer.domain.question import Question
from grounded_answer.grounding.validator import GroundingValidator
from grounded_answer.domain.answer import GroundingStatus


def _evidence(clause_id: str, content: str) -> Evidence:
    return Evidence(
        evidence_id=f"policy-manual.md:{clause_id}",
        clause_id=clause_id,
        content=content,
        source="policy-manual.md",
    )


def test_empty_evidence_is_insufficient() -> None:
    assessment = GroundingValidator().assess(Question(text="What are the eligibility conditions?"), ())
    assert assessment.status is GroundingStatus.INSUFFICIENT
    assert assessment.evidence == ()


def test_overlapping_policy_terms_are_supported() -> None:
    evidence = (
        _evidence(
            "§2.1.2",
            "The conditions are that the person is eligible and satisfies each requirement.",
        ),
    )
    assessment = GroundingValidator().assess(
        Question(text="What are the eligibility requirements?"),
        evidence,
    )
    assert assessment.status is GroundingStatus.SUPPORTED
    assert assessment.evidence[0].clause_id == "§2.1.2"


def test_cited_clause_id_is_supported_even_without_term_overlap() -> None:
    evidence = (_evidence("§6.6.1", "The thresholds are listed in the table."),)
    assessment = GroundingValidator().assess(
        Question(text="What does §6.6.1 say?"),
        evidence,
    )
    assert assessment.status is GroundingStatus.SUPPORTED
    assert assessment.evidence[0].clause_id == "§6.6.1"


def test_off_topic_retrieved_clauses_are_insufficient() -> None:
    evidence = (
        _evidence(
            "§2.1.2",
            "The conditions are that the person is eligible for assistance.",
        ),
    )
    assessment = GroundingValidator().assess(
        Question(text="What is the capital of France?"),
        evidence,
    )
    assert assessment.status is GroundingStatus.INSUFFICIENT
    assert assessment.evidence == ()
