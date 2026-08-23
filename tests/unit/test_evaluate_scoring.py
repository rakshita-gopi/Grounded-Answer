import sys
from pathlib import Path

from grounded_answer.application.answer_service import INSUFFICIENT_ANSWER
from grounded_answer.domain.answer import Answer, GroundingStatus
from grounded_answer.domain.citation import Citation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation" / "scripts"))
from scoring import percentage, score_answer, summarize


def _answer(text: str, clause_ids: tuple[str, ...], status: GroundingStatus) -> Answer:
    return Answer(
        text=text,
        citations=tuple(
            Citation(source_document="policy-manual.md", clause_id=clause_id)
            for clause_id in clause_ids
        ),
        grounding_status=status,
    )


def test_score_supported_answer() -> None:
    result = score_answer(
        _answer("The limit is $4,000 under the resources rule.", ("§2.4.1",), GroundingStatus.SUPPORTED),
        {
            "abstain": False,
            "required_clause_ids": ["§2.4.1"],
            "must_contain": ["$4,000"],
        },
        known_clause_ids={"§2.4.1", "§6.6.1"},
    )
    assert result == {
        "answer": True,
        "evidence": True,
        "citation": True,
        "abstention": True,
    }


def test_score_abstention() -> None:
    result = score_answer(
        _answer(INSUFFICIENT_ANSWER, (), GroundingStatus.INSUFFICIENT),
        {"abstain": True, "required_clause_ids": [], "must_contain": []},
        known_clause_ids={"§2.4.1"},
    )
    assert result["answer"] is True
    assert result["evidence"] is True
    assert result["citation"] is True
    assert result["abstention"] is True


def test_score_invented_citation_fails() -> None:
    result = score_answer(
        _answer("See §99.9.9.", ("§99.9.9",), GroundingStatus.SUPPORTED),
        {"abstain": False, "required_clause_ids": ["§2.4.1"], "must_contain": ["$4,000"]},
        known_clause_ids={"§2.4.1"},
    )
    assert result["citation"] is False
    assert result["evidence"] is False
    assert result["answer"] is False


def test_percentage_and_summarize() -> None:
    assert percentage(2, 4) == "50%"
    assert percentage(1, 3) == "33.3%"
    summary = summarize(
        [
            {"answer": True, "evidence": True, "citation": True, "abstention": True},
            {"answer": False, "evidence": True, "citation": True, "abstention": False},
        ]
    )
    assert summary["total"] == "2"
    assert summary["answer"] == "50%"
    assert summary["evidence"] == "100%"
    assert summary["abstention"] == "50%"
