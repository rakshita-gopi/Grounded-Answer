"""Retrieval query sent to a Retriever implementation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    text: str
    top_k: int = 8
