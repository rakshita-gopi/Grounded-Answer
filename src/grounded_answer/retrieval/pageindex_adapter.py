"""PageIndex adapter. Isolated behind the Retriever port."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from grounded_answer.retrieval.base import Retriever
from grounded_answer.retrieval.models import RetrievalHit, RetrievalQuery


class PageIndexUnavailableError(RuntimeError):
    """Raised when the PageIndex SDK or its required configuration is missing."""


@dataclass(frozen=True, slots=True)
class PageIndexNode:
    node_id: str
    title: str
    text: str


class PageIndexGateway(Protocol):
    def retrieve_nodes(self, query: str, top_k: int) -> Sequence[PageIndexNode]:
        """Return structure-aware nodes for a query."""


class PageIndexRetriever(Retriever):
    """Return raw PageIndex tree nodes as retrieval hits."""

    def __init__(
        self,
        gateway: PageIndexGateway,
        source_document: str = "policy-manual.md",
    ) -> None:
        self._gateway = gateway
        self._source_document = source_document

    def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievalHit]:
        nodes = self._gateway.retrieve_nodes(query.text, query.top_k)
        return tuple(
            RetrievalHit(
                text=node.text,
                source=self._source_document,
                node_id=node.node_id,
                title=node.title,
            )
            for node in nodes
        )


class SdkPageIndexGateway:
    """Thin wrapper around the PageIndex SDK client.

    The SDK requires cloud or LLM credentials. Callers should catch
    PageIndexUnavailableError and use the local fallback.
    """

    def __init__(self, client: Any, doc_id: str) -> None:
        self._client = client
        self._doc_id = doc_id

    def retrieve_nodes(self, query: str, top_k: int) -> Sequence[PageIndexNode]:
        client = self._client
        if hasattr(client, "submit_query") and hasattr(client, "get_retrieval"):
            submitted = client.submit_query(self._doc_id, query)
            retrieval_id = submitted["retrieval_id"] if isinstance(submitted, dict) else submitted
            result = client.get_retrieval(retrieval_id)
            return _nodes_from_sdk_result(result, top_k)
        raise PageIndexUnavailableError(
            "The installed PageIndex SDK does not expose a node-retrieval API."
        )


def build_pageindex_retriever(
    api_key: str,
    doc_id: str,
    source_document: str = "policy-manual.md",
) -> PageIndexRetriever:
    try:
        from pageindex import PageIndexClient
    except ImportError as exc:
        raise PageIndexUnavailableError(
            "The pageindex package is not installed. Use the local fallback, "
            "or install it with: pip install pageindex"
        ) from exc
    client = PageIndexClient(api_key=api_key)
    return PageIndexRetriever(
        SdkPageIndexGateway(client, doc_id=doc_id),
        source_document=source_document,
    )


def _nodes_from_sdk_result(result: Any, top_k: int) -> tuple[PageIndexNode, ...]:
    rows: list[Any]
    if isinstance(result, dict):
        rows = result.get("nodes") or result.get("results") or result.get("pages") or []
    elif isinstance(result, list):
        rows = result
    else:
        rows = []
    nodes: list[PageIndexNode] = []
    for index, row in enumerate(rows[:top_k]):
        if isinstance(row, PageIndexNode):
            nodes.append(row)
            continue
        if not isinstance(row, dict):
            continue
        nodes.append(
            PageIndexNode(
                node_id=str(row.get("node_id") or row.get("id") or index),
                title=str(row.get("title") or ""),
                text=str(row.get("text") or row.get("content") or ""),
            )
        )
    return tuple(nodes)
