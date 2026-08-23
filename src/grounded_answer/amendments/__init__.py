"""Amendment corpus ingestion, kept separate from the original policy manual."""

from grounded_answer.amendments.parser import AmendmentParseError, parse_amendment
from grounded_answer.amendments.service import (
    AmendmentIngestionError,
    AmendmentIngestionService,
)

__all__ = [
    "AmendmentIngestionError",
    "AmendmentIngestionService",
    "AmendmentParseError",
    "parse_amendment",
]
