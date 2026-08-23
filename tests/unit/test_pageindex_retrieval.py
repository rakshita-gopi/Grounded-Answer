from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

from grounded_answer.ingestion.parser import load_policy_text, parse_policy_manual
from grounded_answer.retrieval.factory import create_retriever
from grounded_answer.retrieval.local_fallback import DeterministicStructureRetriever
from grounded_answer.retrieval.models import RetrievalQuery
from grounded_answer.retrieval.pageindex_adapter import (
    PageIndexNode,
    PageIndexRetriever,
    PageIndexUnavailableError,
    build_pageindex_retriever,
)
from grounded_answer.retrieval.service import RetrievalService
from grounded_answer.domain.question import Question


class FakePageIndexGateway:
    def __init__(self, nodes: Sequence[PageIndexNode]) -> None:
        self.nodes = tuple(nodes)
        self.queries: list[tuple[str, int]] = []

    def retrieve_nodes(self, query: str, top_k: int) -> Sequence[PageIndexNode]:
        self.queries.append((query, top_k))
        return self.nodes[:top_k]


def test_pageindex_adapter_maps_nodes_to_clause_evidence() -> None:
    gateway = FakePageIndexGateway(
        [
            PageIndexNode(
                node_id="n1",
                title="Income thresholds",
                text="§6.6.1 A household is not eligible where countable income exceeds the applicable threshold.",
            )
        ]
    )
    retriever = PageIndexRetriever(gateway)
    result = retriever.retrieve(RetrievalQuery(text="income thresholds", top_k=5))
    assert result[0].clause_id == "§6.6.1"
    assert result[0].source == "policy-manual.md"
    assert gateway.queries == [("income thresholds", 5)]


def test_pageindex_adapter_respects_top_k() -> None:
    gateway = FakePageIndexGateway(
        [
            PageIndexNode(node_id="1", title="A", text="§2.1.2 conditions"),
            PageIndexNode(node_id="2", title="B", text="§2.4.1 resources"),
            PageIndexNode(node_id="3", title="C", text="§6.6.1 thresholds"),
        ]
    )
    retriever = PageIndexRetriever(gateway)
    result = retriever.retrieve(RetrievalQuery(text="eligibility", top_k=2))
    assert [item.clause_id for item in result] == ["§2.1.2", "§2.4.1"]


def test_local_fallback_finds_resource_limit(sample_policy_path: Path) -> None:
    document = parse_policy_manual(load_policy_text(sample_policy_path), "sample_policy.md")
    retriever = DeterministicStructureRetriever(document)
    result = retriever.retrieve(RetrievalQuery(text="countable resources exceed $4,000"))
    assert result
    assert result[0].clause_id == "§2.4.1"
    assert "$4,000" in result[0].content


def test_local_fallback_honours_explicit_clause_id(corpus_dir: Path) -> None:
    document = parse_policy_manual(
        load_policy_text(corpus_dir / "policy-manual.md"),
        "policy-manual.md",
    )
    retriever = DeterministicStructureRetriever(document)
    result = retriever.retrieve(RetrievalQuery(text="See §6.6.1 for the thresholds"))
    assert result[0].clause_id == "§6.6.1"


def test_local_fallback_is_deterministic(corpus_dir: Path) -> None:
    document = parse_policy_manual(
        load_policy_text(corpus_dir / "policy-manual.md"),
        "policy-manual.md",
    )
    retriever = DeterministicStructureRetriever(document)
    query = RetrievalQuery(text="eligibility conditions household")
    first = retriever.retrieve(query)
    second = retriever.retrieve(query)
    assert [item.clause_id for item in first] == [item.clause_id for item in second]


def test_factory_uses_local_fallback_without_pageindex_credentials(corpus_dir: Path) -> None:
    retriever = create_retriever(corpus_dir=corpus_dir, environ={})
    assert isinstance(retriever, DeterministicStructureRetriever)
    service = RetrievalService(retriever)
    result = service.retrieve(Question(text="income thresholds"), top_k=5)
    assert any(item.clause_id == "§6.6.1" for item in result)


def test_factory_falls_back_when_pageindex_is_unavailable(corpus_dir: Path) -> None:
    with patch(
        "grounded_answer.retrieval.factory.build_pageindex_retriever",
        side_effect=PageIndexUnavailableError("unavailable"),
    ):
        retriever = create_retriever(
            corpus_dir=corpus_dir,
            environ={"PAGEINDEX_API_KEY": "pi-test", "PAGEINDEX_DOC_ID": "doc-1"},
        )
    assert isinstance(retriever, DeterministicStructureRetriever)


def test_build_pageindex_retriever_raises_when_sdk_missing() -> None:
    try:
        import pageindex  # noqa: F401
    except ImportError:
        raised = False
        try:
            build_pageindex_retriever(api_key="pi-test", doc_id="doc-1")
        except PageIndexUnavailableError:
            raised = True
        assert raised
    else:
        # SDK is present in this environment; constructing the adapter is enough.
        retriever = build_pageindex_retriever(api_key="pi-test", doc_id="doc-1")
        assert isinstance(retriever, PageIndexRetriever)
