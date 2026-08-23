"""PageIndex adapter. Isolated behind the Retriever port."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from grounded_answer.domain.evidence import Evidence
from grounded_answer.retrieval.base import Retriever
from grounded_answer.retrieval.clause_ids import extract_clause_ids
from grounded_answer.retrieval.models import RetrievalQuery


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
    """Maps PageIndex tree nodes onto canonical Evidence objects."""

    def __init__(
        self,
        gateway: PageIndexGateway,
        source_document: str = "policy-manual.md",
    ) -> None:
        self._gateway = gateway
        self._source_document = source_document

    def retrieve(self, query: RetrievalQuery) -> Sequence[Evidence]:
        nodes = self._gateway.retrieve_nodes(query.text, query.top_k)
        evidence: list[Evidence] = []
        seen: set[str] = set()
        for node in nodes:
            for item in _evidence_from_node(node, self._source_document):
                if item.evidence_id in seen:
                    continue
                seen.add(item.evidence_id)
                evidence.append(item)
                if len(evidence) >= query.top_k:
                    return tuple(evidence)
        return tuple(evidence)


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


def _evidence_from_node(node: PageIndexNode, source_document: str) -> tuple[Evidence, ...]:
    blob = f"{node.title}\n{node.text}"
    clause_ids = extract_clause_ids(blob, require_section_sign=True)
    if not clause_ids:
        clause_ids = extract_clause_ids(blob, require_section_sign=False)
    if not clause_ids:
        return (
            Evidence(
                evidence_id=f"{source_document}:node:{node.node_id}",
                clause_id=node.node_id,
                content=node.text or node.title,
                source=source_document,
            ),
        )
    return tuple(
        Evidence(
            evidence_id=f"{source_document}:{clause_id}",
            clause_id=clause_id,
            content=node.text or node.title,
            source=source_document,
        )
        for clause_id in clause_ids
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
