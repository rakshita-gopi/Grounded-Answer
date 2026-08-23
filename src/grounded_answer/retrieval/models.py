"""Retrieval query and raw hits returned by a Retriever implementation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    text: str
    top_k: int = 8


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """Raw retrieval result. Not yet canonical Evidence."""

    text: str
    source: str
    node_id: str = ""
    title: str = ""
    clause_id: str | None = None
