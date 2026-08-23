"""CLI commands for asking grounded policy questions."""

from grounded_answer.application.answer_service import AnswerService
from grounded_answer.application.bootstrap import create_answer_service
from grounded_answer.domain.answer import Answer, GroundingStatus
from grounded_answer.domain.question import Question

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


def ask(question: str, service: AnswerService | None = None) -> str:
    answer_service = service or create_answer_service(load_dotenv=True)
    return format_answer(answer_service.answer(Question(text=question)))
