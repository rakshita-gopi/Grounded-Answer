"""Semantic retrieval over an EmbeddingIndex. Scoring stays inside this adapter."""

from __future__ import annotations

from collections.abc import Sequence

from grounded_answer.embeddings.base import EmbeddingProvider
from grounded_answer.embeddings.index import EmbeddingIndex
from grounded_answer.retrieval.base import Retriever
from grounded_answer.retrieval.clause_ids import extract_clause_ids
from grounded_answer.retrieval.models import RetrievalHit, RetrievalQuery


class EmbeddingRetriever(Retriever):
    """Rank indexed clauses by cosine similarity, with a clause-id boost."""

    def __init__(self, index: EmbeddingIndex, provider: EmbeddingProvider) -> None:
        self._index = index
        self._provider = provider

    def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievalHit]:
        if not self._index.items:
            return ()
        query_vector = self._provider.embed_query(query.text)
        mentioned = set(extract_clause_ids(query.text))
        ranked: list[tuple[float, int, RetrievalHit]] = []
        for order, item in enumerate(self._index.items):
            score = _dot(query_vector, item.embedding)
            if item.clause_id in mentioned:
                score += 10.0
            ranked.append(
                (
                    score,
                    order,
                    RetrievalHit(
                        text=item.text,
                        source=item.source,
                        node_id=item.clause_id,
                        title=item.title,
                        clause_id=item.clause_id,
                    ),
                )
            )
        ranked.sort(key=lambda row: (-row[0], row[1]))
        return tuple(hit for _, _, hit in ranked[: query.top_k])


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    limit = min(len(left), len(right))
    return sum(left[index] * right[index] for index in range(limit))
