"""Retrieval abstraction used by the rest of the application."""

from grounded_answer.retrieval.base import Retriever
from grounded_answer.retrieval.factory import create_retriever
from grounded_answer.retrieval.local_fallback import DeterministicStructureRetriever
from grounded_answer.retrieval.models import RetrievalQuery
from grounded_answer.retrieval.pageindex_adapter import (
    PageIndexRetriever,
    PageIndexUnavailableError,
)
from grounded_answer.retrieval.service import RetrievalService

__all__ = [
    "DeterministicStructureRetriever",
    "PageIndexRetriever",
    "PageIndexUnavailableError",
    "RetrievalQuery",
    "RetrievalService",
    "Retriever",
    "create_retriever",
]
