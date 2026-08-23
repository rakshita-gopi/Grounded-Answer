"""Merge hits from the original policy retriever and the amendment retriever."""

from __future__ import annotations

from collections.abc import Sequence

from grounded_answer.retrieval.base import Retriever
from grounded_answer.retrieval.models import RetrievalHit, RetrievalQuery


class CompositeRetriever(Retriever):
    """Retrieve from multiple corpora through the existing Retriever port."""

    def __init__(self, *retrievers: Retriever) -> None:
        self._retrievers = retrievers

    def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievalHit]:
        merged: list[RetrievalHit] = []
        seen: set[tuple[str, str]] = set()
        child_query = RetrievalQuery(text=query.text, top_k=max(query.top_k, 8))
        for retriever in self._retrievers:
            for hit in retriever.retrieve(child_query):
                key = (hit.clause_id or hit.node_id, hit.source)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(hit)
        limit = max(query.top_k * 2, 12)
        return tuple(merged[:limit])
