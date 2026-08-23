"""Stage B generation cases. Live Ollama tests skip when qwen3:4b is not reachable."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from grounded_answer.application.answer_service import (
    INSUFFICIENT_ANSWER,
    MISSING_DATE_ANSWER,
)
from grounded_answer.application.bootstrap import create_answer_service
from grounded_answer.domain.answer import GroundingStatus
from grounded_answer.domain.question import Question
from grounded_answer.llm.ollama import OllamaLLMProvider
from grounded_answer.llm.provider import LLMUnavailableError, StubLLMProvider

FEB_EARNINGS = (
    "For a determination made on 15 February 2026, what is the first monthly earnings disregard?"
)
MAR_EARNINGS = (
    "For a determination made on 15 March 2026, what is the first monthly earnings disregard?"
)
FEB_CHANGE = (
    "The claimant's circumstances changed on 20 February 2026. "
    "How many calendar days does the recipient have to report the change?"
)
MAR_CHANGE = (
    "The claimant's circumstances changed on 5 March 2026. "
    "How many calendar days does the recipient have to report the change?"
)
SPANNING = (
    "The claim runs from 20 February 2026 to 10 March 2026. "
    "What earnings disregard figures apply, and how is the award treated?"
)


def _live_ollama_llm() -> bool:
    try:
        OllamaLLMProvider.from_environ(os.environ).ping()
    except LLMUnavailableError:
        return False
    return os.environ.get("LLM_PROVIDER", "").strip().lower() == "ollama"


def test_february_answer_contains_original_disregard(corpus_dir: Path) -> None:
    llm = StubLLMProvider(text="The first monthly earnings disregard is $120 per month. §6.4.1")
    service = create_answer_service(
        corpus_dir=corpus_dir, environ={}, load_dotenv=False, llm=llm
    )
    result = service.answer(Question(text=FEB_EARNINGS))
    assert result.grounding_status is GroundingStatus.SUPPORTED
    assert "$120" in result.text
    assert any(citation.clause_id == "§6.4.1" for citation in result.citations)
    assert llm.calls == []


def test_march_answer_contains_amended_disregard(corpus_dir: Path) -> None:
    llm = StubLLMProvider(text="The first monthly earnings disregard is $175 per month. §6.4.1")
    service = create_answer_service(
        corpus_dir=corpus_dir, environ={}, load_dotenv=False, llm=llm
    )
    result = service.answer(Question(text=MAR_EARNINGS))
    assert result.grounding_status is GroundingStatus.SUPPORTED
    assert "$175" in result.text
    assert any(citation.clause_id == "§6.4.1" for citation in result.citations)
    assert llm.calls == []


def test_february_change_answer_contains_ten_days(corpus_dir: Path) -> None:
    llm = StubLLMProvider(text="The recipient has 10 calendar days to report the change. §4.3.2")
    service = create_answer_service(
        corpus_dir=corpus_dir, environ={}, load_dotenv=False, llm=llm
    )
    result = service.answer(Question(text=FEB_CHANGE))
    assert result.grounding_status is GroundingStatus.SUPPORTED
    assert "10 calendar days" in result.text
    assert "14 calendar days" not in result.text
    assert any(citation.clause_id == "§4.3.2" for citation in result.citations)
    assert llm.calls == []


def test_march_change_answer_contains_fourteen_days(corpus_dir: Path) -> None:
    llm = StubLLMProvider(text="The recipient has 14 calendar days to report the change. §4.3.2")
    service = create_answer_service(
        corpus_dir=corpus_dir, environ={}, load_dotenv=False, llm=llm
    )
    result = service.answer(Question(text=MAR_CHANGE))
    assert result.grounding_status is GroundingStatus.SUPPORTED
    assert "14 calendar days" in result.text
    assert "10 calendar days" not in result.text
    assert llm.calls == []


def test_spanning_claim_answer_contains_both_disregards(corpus_dir: Path) -> None:
    llm = StubLLMProvider(
        text="The award uses $120 then $175, apportioned under §7.4.3. §6.4.1"
    )
    service = create_answer_service(
        corpus_dir=corpus_dir, environ={}, load_dotenv=False, llm=llm
    )
    result = service.answer(Question(text=SPANNING))
    assert result.grounding_status is GroundingStatus.SUPPORTED
    assert "$120" in result.text
    assert "$175" in result.text
    cited = {citation.clause_id for citation in result.citations}
    assert "§6.4.1" in cited
    assert "§7.4.3" in cited
    assert "§6.6.1" not in cited
    assert llm.calls == []


def test_first_sanction_after_amendment_does_not_call_llm(corpus_dir: Path) -> None:
    llm = StubLLMProvider(text="should not be used")
    service = create_answer_service(
        corpus_dir=corpus_dir, environ={}, load_dotenv=False, llm=llm
    )
    result = service.answer(
        Question(
            text="For a determination made on 2 March 2026, what is the reduction for a first sanction?"
        )
    )
    assert result.grounding_status is GroundingStatus.SUPPORTED
    assert result.text.startswith("The reduction for a first sanction is 15 per cent.")
    assert "[§10.5.2]" in result.text
    assert "Okay" not in result.text
    assert llm.calls == []


def test_missing_date_does_not_call_llm(corpus_dir: Path) -> None:
    llm = StubLLMProvider(text="should not be used")
    service = create_answer_service(
        corpus_dir=corpus_dir, environ={}, load_dotenv=False, llm=llm
    )
    result = service.answer(Question(text="What is the first monthly earnings disregard?"))
    assert result.grounding_status is GroundingStatus.INSUFFICIENT
    assert result.text == MISSING_DATE_ANSWER
    assert result.citations == ()
    assert llm.calls == []


def test_helium_does_not_call_llm(corpus_dir: Path) -> None:
    llm = StubLLMProvider(text="should not be used")
    service = create_answer_service(
        corpus_dir=corpus_dir, environ={}, load_dotenv=False, llm=llm
    )
    result = service.answer(Question(text="What is the boiling point of helium?"))
    assert result.grounding_status is GroundingStatus.INSUFFICIENT
    assert result.text == INSUFFICIENT_ANSWER
    assert result.citations == ()
    assert llm.calls == []


@pytest.mark.skipif(not _live_ollama_llm(), reason="Ollama qwen3:4b is not available")
def test_live_ollama_february_contains_120(corpus_dir: Path) -> None:
    service = create_answer_service(
        corpus_dir=corpus_dir, environ=dict(os.environ), load_dotenv=False
    )
    result = service.answer(Question(text=FEB_EARNINGS))
    assert result.grounding_status is GroundingStatus.SUPPORTED
    assert "$120" in result.text
    assert any(citation.clause_id == "§6.4.1" for citation in result.citations)


@pytest.mark.skipif(not _live_ollama_llm(), reason="Ollama qwen3:4b is not available")
def test_live_ollama_march_contains_175(corpus_dir: Path) -> None:
    service = create_answer_service(
        corpus_dir=corpus_dir, environ=dict(os.environ), load_dotenv=False
    )
    result = service.answer(Question(text=MAR_EARNINGS))
    assert result.grounding_status is GroundingStatus.SUPPORTED
    assert "$175" in result.text
    assert any(citation.clause_id == "§6.4.1" for citation in result.citations)


@pytest.mark.skipif(not _live_ollama_llm(), reason="Ollama qwen3:4b is not available")
def test_live_ollama_february_change_contains_ten_days(corpus_dir: Path) -> None:
    service = create_answer_service(
        corpus_dir=corpus_dir, environ=dict(os.environ), load_dotenv=False
    )
    result = service.answer(Question(text=FEB_CHANGE))
    assert result.grounding_status is GroundingStatus.SUPPORTED
    assert "10 calendar days" in result.text


@pytest.mark.skipif(not _live_ollama_llm(), reason="Ollama qwen3:4b is not available")
def test_live_ollama_march_change_contains_fourteen_days(corpus_dir: Path) -> None:
    service = create_answer_service(
        corpus_dir=corpus_dir, environ=dict(os.environ), load_dotenv=False
    )
    result = service.answer(Question(text=MAR_CHANGE))
    assert result.grounding_status is GroundingStatus.SUPPORTED
    assert "14 calendar days" in result.text


@pytest.mark.skipif(not _live_ollama_llm(), reason="Ollama qwen3:4b is not available")
def test_live_ollama_spanning_contains_both_amounts(corpus_dir: Path) -> None:
    service = create_answer_service(
        corpus_dir=corpus_dir, environ=dict(os.environ), load_dotenv=False
    )
    result = service.answer(Question(text=SPANNING))
    assert result.grounding_status is GroundingStatus.SUPPORTED
    assert "$120" in result.text
    assert "$175" in result.text


@pytest.mark.skipif(not _live_ollama_llm(), reason="Ollama qwen3:4b is not available")
def test_live_ollama_missing_date_still_abstains(corpus_dir: Path) -> None:
    service = create_answer_service(
        corpus_dir=corpus_dir, environ=dict(os.environ), load_dotenv=False
    )
    result = service.answer(Question(text="What is the first monthly earnings disregard?"))
    assert result.grounding_status is GroundingStatus.INSUFFICIENT
    assert result.text == MISSING_DATE_ANSWER


@pytest.mark.skipif(not _live_ollama_llm(), reason="Ollama qwen3:4b is not available")
def test_live_ollama_helium_still_abstains(corpus_dir: Path) -> None:
    service = create_answer_service(
        corpus_dir=corpus_dir, environ=dict(os.environ), load_dotenv=False
    )
    result = service.answer(Question(text="What is the boiling point of helium?"))
    assert result.grounding_status is GroundingStatus.INSUFFICIENT
    assert result.text == INSUFFICIENT_ANSWER
