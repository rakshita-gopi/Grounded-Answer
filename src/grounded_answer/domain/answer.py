"""A grounded answer, including citations and whether evidence supports it."""

from dataclasses import dataclass
from enum import Enum

from grounded_answer.domain.citation import Citation


class GroundingStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    grounding_status: GroundingStatus
