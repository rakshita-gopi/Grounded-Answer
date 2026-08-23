"""Run Stage A original and Stage B surprise evaluation datasets."""

from __future__ import annotations

import argparse
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
from grounded_answer.amendments.service import AmendmentIngestionService  # noqa: E402


DATASETS = {
    "original": ROOT / "evaluation" / "original",
    "surprise": ROOT / "evaluation" / "surprise",
}


def load_dataset(folder: Path) -> tuple[list[dict], dict]:
    questions = json.loads((folder / "questions.json").read_text(encoding="utf-8"))
    expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
    return questions["questions"], expected


def known_clause_ids(root: Path) -> set[str]:
    ids = {
        clause.clause_id
        for clause in IngestionService(root / "data" / "policy").load_policy().clauses
    }
    amendment = AmendmentIngestionService(root / "data" / "amendments").load_amendment()
    ids.update(paragraph.paragraph_id for paragraph in amendment.paragraphs)
    ids.update(change.target_clause for change in amendment.changes)
    return ids


def run_dataset(name: str, root: Path) -> dict[str, str]:
    items, expected = load_dataset(DATASETS[name])
    known_ids = known_clause_ids(root)
    service = create_answer_service(
        corpus_dir=root / "data" / "policy",
        load_dotenv=True,
    )
    rows = []
    for item in items:
        answer = service.answer(Question(text=item["question"]))
        rows.append(score_answer(answer, expected[item["id"]], known_ids))
    return summarize(rows)


def render(name: str, summary: dict[str, str]) -> str:
    return (
        f"Dataset: {name}\n"
        f"Total questions: {summary['total']}\n"
        f"Answer correctness: {summary['answer']}\n"
        f"Evidence correctness: {summary['evidence']}\n"
        f"Citation correctness: {summary['citation']}\n"
        f"Abstention correctness: {summary['abstention']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Grounded Answer evaluation datasets.")
    parser.add_argument(
        "--dataset",
        choices=("original", "surprise", "all"),
        default="all",
        help="Which evaluation dataset to run.",
    )
    args = parser.parse_args(argv)
    names = ("original", "surprise") if args.dataset == "all" else (args.dataset,)
    chunks = [render(name, run_dataset(name, ROOT)) for name in names]
    sys.stdout.write("\n".join(chunks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
