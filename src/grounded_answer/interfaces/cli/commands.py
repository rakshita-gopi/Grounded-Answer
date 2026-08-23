"""CLI commands for asking grounded policy questions."""

from datetime import date

from grounded_answer.application.answer_service import AnswerService
from grounded_answer.application.bootstrap import create_answer_service
from grounded_answer.domain.answer import Answer, GroundingStatus
from grounded_answer.domain.question import Question
from grounded_answer.domain.temporal import TemporalContext

RULE = "-" * 24


def format_answer(answer: Answer) -> str:
    if answer.grounding_status is GroundingStatus.INSUFFICIENT:
        return (
            f"GROUNDING\n{RULE}\n"
            f"{answer.grounding_status.value}\n"
            f"{answer.text}\n"
        )
    evidence = "\n".join(citation.clause_id for citation in answer.citations)
    return (
        f"ANSWER\n{RULE}\n"
        f"{answer.text}\n"
        f"EVIDENCE\n{RULE}\n"
        f"{evidence}\n"
        f"GROUNDING\n{RULE}\n"
        f"{answer.grounding_status.value}\n"
    )


def ask(
    question: str,
    service: AnswerService | None = None,
    *,
    determination_date: date | None = None,
    change_of_circumstances_date: date | None = None,
    claim_start_date: date | None = None,
    claim_end_date: date | None = None,
) -> str:
    answer_service = service or create_answer_service(load_dotenv=True)
    temporal = TemporalContext(
        determination_date=determination_date,
        claim_start_date=claim_start_date,
        claim_end_date=claim_end_date,
        change_of_circumstances_date=change_of_circumstances_date,
    )
    return format_answer(
        answer_service.answer(Question(text=question, temporal=temporal))
    )
