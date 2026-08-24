"""Persistent clause embedding index with model/dimension metadata."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from grounded_answer.embeddings.base import EmbeddingProvider
from grounded_answer.embeddings.config import DEFAULT_INDEX_VERSION
from grounded_answer.ingestion.models import ParsedClause, ParsedDocument

INDEX_FILENAME = "index.json"


class IndexIncompatibleError(RuntimeError):
    """Raised when stored vectors were built with a different embedding model."""


@dataclass(frozen=True, slots=True)
class IndexedVector:
    clause_id: str
    source: str
    title: str
    text: str
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class IndexMetadata:
    provider: str
    model: str
    dimension: int
    index_version: int
    source_fingerprints: dict[str, str]


@dataclass(frozen=True, slots=True)
class EmbeddingIndex:
    metadata: IndexMetadata
    items: tuple[IndexedVector, ...]

    def validate(self, provider: EmbeddingProvider) -> None:
        if self.metadata.provider != provider.provider_name:
            raise IndexIncompatibleError(
                f"ERROR: The stored index used embedding provider "
                f"{self.metadata.provider!r}, but {provider.provider_name!r} is configured. "
                "The index must be rebuilt."
            )
        if self.metadata.model != provider.model_name:
            raise IndexIncompatibleError(
                f"ERROR: The stored index was built with {self.metadata.model!r}, "
                f"but {provider.model_name!r} is configured. The index must be rebuilt."
            )
        if self.metadata.index_version != DEFAULT_INDEX_VERSION:
            raise IndexIncompatibleError(
                "ERROR: The stored index version is incompatible and must be rebuilt."
            )
        if self.items and len(self.items[0].embedding) != self.metadata.dimension:
            raise IndexIncompatibleError(
                "ERROR: Stored embedding dimension does not match index metadata."
            )


def index_path(index_dir: Path) -> Path:
    return index_dir / INDEX_FILENAME


def document_fingerprint(document: ParsedDocument) -> str:
    digest = hashlib.sha256()
    digest.update(document.source_document.encode("utf-8"))
    for clause in document.clauses:
        digest.update(clause.clause_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(clause.content.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def fingerprints_for(documents: Sequence[ParsedDocument]) -> dict[str, str]:
    return {document.source_document: document_fingerprint(document) for document in documents}


def load_index(path: Path) -> EmbeddingIndex:
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = payload["metadata"]
    items = tuple(
        IndexedVector(
            clause_id=row["clause_id"],
            source=row["source"],
            title=row.get("title") or "",
            text=row["text"],
            embedding=tuple(float(value) for value in row["embedding"]),
        )
        for row in payload["items"]
    )
    metadata = IndexMetadata(
        provider=str(meta["provider"]),
        model=str(meta["model"]),
        dimension=int(meta["dimension"]),
        index_version=int(meta["index_version"]),
        source_fingerprints=dict(meta.get("source_fingerprints") or {}),
    )
    return EmbeddingIndex(metadata=metadata, items=items)


def save_index(path: Path, index: EmbeddingIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "provider": index.metadata.provider,
            "model": index.metadata.model,
            "dimension": index.metadata.dimension,
            "index_version": index.metadata.index_version,
            "source_fingerprints": index.metadata.source_fingerprints,
        },
        "items": [
            {
                "clause_id": item.clause_id,
                "source": item.source,
                "title": item.title,
                "text": item.text,
                "embedding": list(item.embedding),
            }
            for item in index.items
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_index(
    documents: Sequence[ParsedDocument],
    provider: EmbeddingProvider,
) -> EmbeddingIndex:
    clauses = _flatten(documents)
    texts = [_document_text(clause) for clause in clauses]
    sys.stderr.write(
        f"Embedding {len(clauses)} clauses with {provider.model_name} "
        f"({provider.provider_name})...\n"
    )
    sys.stderr.flush()
    vectors = provider.embed_documents(texts)
    if len(vectors) != len(clauses):
        raise IndexIncompatibleError("ERROR: Embedding provider returned the wrong number of vectors.")
    dimension = len(vectors[0]) if vectors else 0
    items = tuple(
        IndexedVector(
            clause_id=clause.clause_id,
            source=clause.source_document,
            title=clause.title,
            text=clause.content,
            embedding=tuple(vector),
        )
        for clause, vector in zip(clauses, vectors)
    )
    metadata = IndexMetadata(
        provider=provider.provider_name,
        model=provider.model_name,
        dimension=dimension,
        index_version=DEFAULT_INDEX_VERSION,
        source_fingerprints=fingerprints_for(documents),
    )
    return EmbeddingIndex(metadata=metadata, items=items)


def ensure_index(
    documents: Sequence[ParsedDocument],
    provider: EmbeddingProvider,
    index_dir: Path,
) -> EmbeddingIndex:
    path = index_path(index_dir)
    expected = fingerprints_for(documents)
    label = ", ".join(document.source_document for document in documents) or str(index_dir)
    if path.exists():
        try:
            loaded = load_index(path)
            loaded.validate(provider)
            if loaded.metadata.source_fingerprints == expected and loaded.items:
                sys.stderr.write(
                    f"Using existing retrieval index for {label} "
                    f"({len(loaded.items)} clauses). Skipping re-embed.\n"
                )
                sys.stderr.flush()
                return loaded
            reason = "source documents changed"
        except (OSError, KeyError, TypeError, ValueError, IndexIncompatibleError) as exc:
            reason = str(exc)
        sys.stderr.write(f"Rebuilding embedding index for {label} ({reason}).\n")
        sys.stderr.flush()
    else:
        sys.stderr.write(
            f"Building retrieval index for {label}. "
            "This runs once and is reused from the index volume.\n"
        )
        sys.stderr.flush()
    index = build_index(documents, provider)
    save_index(path, index)
    sys.stderr.write(f"Retrieval index saved to {path} ({len(index.items)} clauses).\n")
    sys.stderr.flush()
    return index


def _flatten(documents: Sequence[ParsedDocument]) -> tuple[ParsedClause, ...]:
    clauses: list[ParsedClause] = []
    seen: set[tuple[str, str]] = set()
    for document in documents:
        for clause in document.clauses:
            key = (clause.source_document, clause.clause_id)
            if key in seen:
                continue
            seen.add(key)
            clauses.append(clause)
    return tuple(clauses)


def _document_text(clause: ParsedClause) -> str:
    return f"{clause.clause_id} {clause.title}\n{clause.content}"
