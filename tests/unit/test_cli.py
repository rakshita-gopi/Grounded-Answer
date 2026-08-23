from grounded_answer.application.bootstrap import create_answer_service
from grounded_answer.application.answer_service import INSUFFICIENT_ANSWER
from grounded_answer.domain.answer import Answer, GroundingStatus
from grounded_answer.domain.citation import Citation
from grounded_answer.interfaces.cli.commands import ask, format_answer
from grounded_answer.interfaces.cli.main import build_parser, main


def test_format_supported_answer() -> None:
    output = format_answer(
        Answer(
            text="Eligibility is assessed at household level.",
            citations=(
                Citation(source_document="policy-manual.md", clause_id="§2.1.2"),
                Citation(source_document="policy-manual.md", clause_id="§2.4.1"),
            ),
            grounding_status=GroundingStatus.SUPPORTED,
        )
    )
    assert output.startswith("ANSWER")
    assert "§2.1.2" in output
    assert "§2.4.1" in output
    assert "GROUNDING" in output
    assert "SUPPORTED" in output
    assert "EVIDENCE" in output


def test_format_insufficient_answer() -> None:
    output = format_answer(
        Answer(
            text=INSUFFICIENT_ANSWER,
            citations=(),
            grounding_status=GroundingStatus.INSUFFICIENT,
        )
    )
    assert "GROUNDING" in output
    assert "INSUFFICIENT" in output
    assert INSUFFICIENT_ANSWER in output
    assert "EVIDENCE" not in output


def test_parser_accepts_ask_command() -> None:
    args = build_parser().parse_args(["ask", "What are the eligibility requirements?"])
    assert args.command == "ask"
    assert args.question == "What are the eligibility requirements?"
    assert args.determination_date is None


def test_parser_accepts_determination_date() -> None:
    args = build_parser().parse_args(
        ["ask", "What is the earnings disregard?", "--determination-date", "2026-03-15"]
    )
    assert args.determination_date.isoformat() == "2026-03-15"



def test_main_prints_insufficient_for_unsupported_question(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        "grounded_answer.interfaces.cli.main.ask",
        lambda question, service=None, **kwargs: format_answer(
            Answer(
                text=INSUFFICIENT_ANSWER,
                citations=(),
                grounding_status=GroundingStatus.INSUFFICIENT,
            )
        ),
    )
    code = main(["ask", "What is the capital of France?"])
    captured = capsys.readouterr()
    assert code == 0
    assert "INSUFFICIENT" in captured.out
    assert INSUFFICIENT_ANSWER in captured.out


def test_ask_unsupported_question_against_corpus() -> None:
    service = create_answer_service(environ={}, load_dotenv=False)
    output = ask("What is the boiling point of helium?", service=service)
    assert "INSUFFICIENT" in output
    assert INSUFFICIENT_ANSWER in output
