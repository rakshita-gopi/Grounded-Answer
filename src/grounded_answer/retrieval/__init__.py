"""Retrieval abstraction used by the rest of the application."""

from grounded_answer.retrieval.base import Retriever
from grounded_answer.retrieval.factory import create_retriever
from grounded_answer.retrieval.local_fallback import DeterministicStructureRetriever
from grounded_answer.retrieval.models import RetrievalHit, RetrievalQuery
from grounded_answer.retrieval.pageindex_adapter import (
    PageIndexRetriever,
    PageIndexUnavailableError,
)

__all__ = [
    "DeterministicStructureRetriever",
    "PageIndexRetriever",
    "PageIndexUnavailableError",
    "RetrievalHit",
    "RetrievalQuery",
    "RetrievalService",
    "Retriever",
    "create_retriever",
]


def __getattr__(name: str):
    if name == "RetrievalService":
        from grounded_answer.retrieval.service import RetrievalService

        return RetrievalService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
