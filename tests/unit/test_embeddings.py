from pathlib import Path

from grounded_answer.embeddings.base import EmbeddingProvider
from grounded_answer.embeddings.index import (
    IndexIncompatibleError,
    build_index,
    ensure_index,
    save_index,
)
from grounded_answer.ingestion.parser import load_policy_text, parse_policy_manual
from grounded_answer.retrieval.embedding_retriever import EmbeddingRetriever
from grounded_answer.retrieval.models import RetrievalQuery


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "fake-model") -> None:
        self._model = model
        self.embed_document_calls = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return self._model

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_documents(self, texts):
        self.embed_document_calls += 1
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            1.0 if "resource" in lowered or "4,000" in lowered else 0.1,
            1.0 if "threshold" in lowered or "income" in lowered else 0.1,
            1.0 if "eligib" in lowered else 0.1,
            1.0 if "§2.4.1" in lowered else 0.0,
        ]


def _sample_document(sample_policy_path: Path):
    return parse_policy_manual(load_policy_text(sample_policy_path), "sample_policy.md")


def test_build_index_records_model_and_dimension(sample_policy_path: Path) -> None:
    document = _sample_document(sample_policy_path)
    index = build_index((document,), FakeEmbeddingProvider())
    assert index.metadata.provider == "fake"
    assert index.metadata.model == "fake-model"
    assert index.metadata.dimension == 4
    assert index.items


def test_index_rejects_different_model(sample_policy_path: Path, tmp_path: Path) -> None:
    document = _sample_document(sample_policy_path)
    index = build_index((document,), FakeEmbeddingProvider("model-a"))
    save_index(tmp_path / "index.json", index)
    try:
        ensure_index((document,), FakeEmbeddingProvider("model-b"), tmp_path)
        rebuilt = True
    except IndexIncompatibleError:
        rebuilt = False
    # ensure_index rebuilds rather than serving stale vectors
    assert rebuilt is True
    loaded_ok = ensure_index((document,), FakeEmbeddingProvider("model-b"), tmp_path)
    assert loaded_ok.metadata.model == "model-b"


def test_embedding_retriever_ranks_resource_clause(sample_policy_path: Path) -> None:
    document = _sample_document(sample_policy_path)
    provider = FakeEmbeddingProvider()
    index = build_index((document,), provider)
    hits = EmbeddingRetriever(index, provider).retrieve(
        RetrievalQuery(text="countable resources exceed $4,000", top_k=3)
    )
    assert hits
    assert hits[0].clause_id == "§2.4.1"


def test_ensure_index_reuses_compatible_file_without_reembedding(
    sample_policy_path: Path, tmp_path: Path
) -> None:
    document = _sample_document(sample_policy_path)
    provider = FakeEmbeddingProvider()
    first = ensure_index((document,), provider, tmp_path)
    assert provider.embed_document_calls == 1
    assert first.items
    second = ensure_index((document,), provider, tmp_path)
    assert provider.embed_document_calls == 1
    assert [item.clause_id for item in second.items] == [item.clause_id for item in first.items]
