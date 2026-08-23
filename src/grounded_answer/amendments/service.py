"""Load the surprise-challenge amendment without modifying the original policy."""

from __future__ import annotations

from pathlib import Path

from grounded_answer.amendments.parser import AmendmentParseError, parse_amendment
from grounded_answer.domain.amendment import Amendment
from grounded_answer.ingestion.parser import load_policy_text

DEFAULT_AMENDMENTS_DIR = Path(__file__).resolve().parents[3] / "data" / "amendments"
AMENDMENT_FILENAME = "Amendment No. 2026-01.md"


class AmendmentIngestionError(RuntimeError):
    """Raised when the amendment corpus cannot be loaded."""


class AmendmentIngestionService:
    def __init__(self, amendments_dir: Path | None = None) -> None:
        self.amendments_dir = amendments_dir or DEFAULT_AMENDMENTS_DIR

    def load_amendment(self) -> Amendment:
        path = self.amendments_dir / AMENDMENT_FILENAME
        if not path.exists():
            raise AmendmentIngestionError("The amendment corpus could not be loaded.")
        try:
            text = load_policy_text(path)
            issued, effective, paragraphs, changes = parse_amendment(
                text,
                source_document=path.name,
            )
        except (OSError, AmendmentParseError) as exc:
            raise AmendmentIngestionError("The amendment corpus could not be loaded.") from exc
        return Amendment(
            amendment_id="2026-01",
            source_document=path.name,
            issued_date=issued,
            effective_date=effective,
            changes=changes,
            paragraphs=paragraphs,
        )
