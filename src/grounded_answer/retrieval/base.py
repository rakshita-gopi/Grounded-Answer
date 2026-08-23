"""Retriever port. Application code depends on this interface, not on a vendor."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from grounded_answer.retrieval.models import RetrievalHit, RetrievalQuery


class Retriever(ABC):
    """Return raw retrieval hits for a query.

    Concrete adapters (PageIndex, local fallback, test doubles) implement this
    method. Canonical Evidence is produced by the evidence assembly layer, not
    by the adapter.
    """

    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievalHit]:
        raise NotImplementedError
