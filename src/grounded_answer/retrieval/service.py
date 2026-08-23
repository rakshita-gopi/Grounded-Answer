"""Application-facing retrieval that delegates to a Retriever port."""

from grounded_answer.domain.evidence import Evidence
from grounded_answer.domain.question import Question
from grounded_answer.retrieval.base import Retriever
from grounded_answer.retrieval.models import RetrievalQuery


class RetrievalService:
    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    def retrieve(self, question: Question, top_k: int = 8) -> tuple[Evidence, ...]:
        query = RetrievalQuery(text=question.text, top_k=top_k)
        return tuple(self._retriever.retrieve(query))
