"""Run the Stage A original evaluation dataset and print measured scores."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation" / "scripts"))

from scoring import score_answer, summarize  # noqa: E402

from grounded_answer.application.bootstrap import create_answer_service  # noqa: E402
from grounded_answer.domain.question import Question  # noqa: E402
from grounded_answer.ingestion.service import IngestionService  # noqa: E402


def load_dataset(root: Path) -> tuple[list[dict], dict]:
    original = root / "evaluation" / "original"
    questions = json.loads((original / "questions.json").read_text(encoding="utf-8"))
    expected = json.loads((original / "expected.json").read_text(encoding="utf-8"))
    return questions["questions"], expected


def run(root: Path | None = None) -> dict[str, str]:
    root = root or ROOT
    items, expected = load_dataset(root)
    known_ids = {
        clause.clause_id
        for clause in IngestionService(root / "data" / "policy").load_policy().clauses
    }
    service = create_answer_service(
        corpus_dir=root / "data" / "policy",
        load_dotenv=True,
    )
    rows = []
    for item in items:
        answer = service.answer(Question(text=item["question"]))
        rows.append(score_answer(answer, expected[item["id"]], known_ids))
    return summarize(rows)


def render(summary: dict[str, str]) -> str:
    return (
        f"Total questions: {summary['total']}\n"
        f"Answer correctness: {summary['answer']}\n"
        f"Evidence correctness: {summary['evidence']}\n"
        f"Citation correctness: {summary['citation']}\n"
        f"Abstention correctness: {summary['abstention']}\n"
    )


def main() -> int:
    sys.stdout.write(render(run()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
