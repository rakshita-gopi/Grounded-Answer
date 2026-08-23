"""Retrieval abstraction used by the rest of the application."""

from grounded_answer.retrieval.base import Retriever
from grounded_answer.retrieval.models import RetrievalQuery
from grounded_answer.retrieval.service import RetrievalService

__all__ = [
    "RetrievalQuery",
    "RetrievalService",
    "Retriever",
]
