"""Application-facing retrieval that assembles canonical Evidence."""

from grounded_answer.domain.evidence import Evidence
from grounded_answer.domain.question import Question
from grounded_answer.evidence.assembler import EvidenceAssembler
from grounded_answer.evidence.validator import EvidenceValidator
from grounded_answer.retrieval.base import Retriever
from grounded_answer.retrieval.models import RetrievalQuery


class RetrievalService:
    def __init__(
        self,
        retriever: Retriever,
        assembler: EvidenceAssembler,
        validator: EvidenceValidator | None = None,
    ) -> None:
        self._retriever = retriever
        self._assembler = assembler
        self._validator = validator or EvidenceValidator()

    def retrieve(self, question: Question, top_k: int = 8) -> tuple[Evidence, ...]:
        query = RetrievalQuery(text=question.text, top_k=top_k)
        hits = self._retriever.retrieve(query)
        assembled = self._assembler.assemble(hits)
        validated = self._validator.validate(assembled)
        return validated
