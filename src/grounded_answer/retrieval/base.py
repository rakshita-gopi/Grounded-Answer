"""Retriever port. Application code depends on this interface, not on a vendor."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from grounded_answer.domain.evidence import Evidence
from grounded_answer.retrieval.models import RetrievalQuery


class Retriever(ABC):
    """Return policy evidence for a query.

    Concrete adapters (PageIndex, local fallback, test doubles) implement this
    method. Callers must not depend on adapter internals.
    """

    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> Sequence[Evidence]:
        raise NotImplementedError
