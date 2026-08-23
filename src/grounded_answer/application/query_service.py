"""Look up policy evidence for a question.

This service coordinates retrieval. It does not depend on PageIndex or any
other concrete retriever.
"""

from grounded_answer.domain.evidence import Evidence
from grounded_answer.domain.question import Question
from grounded_answer.retrieval.service import RetrievalService


class QueryService:
    def __init__(self, retrieval_service: RetrievalService) -> None:
        self._retrieval_service = retrieval_service

    def evidence_for(self, question: Question, top_k: int = 8) -> tuple[Evidence, ...]:
        return self._retrieval_service.retrieve(question, top_k=top_k)
