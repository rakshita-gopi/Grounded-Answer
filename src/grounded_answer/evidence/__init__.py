"""Evidence assembly pipeline: raw hits -> canonical Evidence."""

from grounded_answer.evidence.assembler import EvidenceAssembler
from grounded_answer.evidence.validator import EvidenceValidator

__all__ = [
    "EvidenceAssembler",
    "EvidenceValidator",
]
