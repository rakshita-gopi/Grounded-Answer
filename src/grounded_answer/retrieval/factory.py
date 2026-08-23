"""Choose a Retriever implementation without exposing Ollama to callers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from grounded_answer.embeddings.config import embeddings_requested
from grounded_answer.embeddings.index import ensure_index
from grounded_answer.embeddings.ollama import OllamaEmbeddingProvider
from grounded_answer.ingestion.models import ParsedDocument
from grounded_answer.ingestion.parser import load_policy_text, parse_policy_manual
from grounded_answer.ingestion.service import DEFAULT_CORPUS_DIR
from grounded_answer.retrieval.base import Retriever
from grounded_answer.retrieval.composite import CompositeRetriever
from grounded_answer.retrieval.embedding_retriever import EmbeddingRetriever
from grounded_answer.retrieval.local_fallback import DeterministicStructureRetriever

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"
DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "index"


def create_retriever(
    corpus_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
    *,
    load_dotenv: bool = False,
    amendment_document: ParsedDocument | None = None,
) -> Retriever:
    """Return Ollama embeddings when configured, otherwise the lexical fallback.

    `load_dotenv` is off by default so tests control configuration explicitly.
    Ollama types stay inside the embedding adapter.
    """
    env = dict(environ) if environ is not None else dict(os.environ)
    if load_dotenv:
        env = {**_read_env_file(DEFAULT_ENV_PATH), **env}

    policy_document = _load_policy_document(corpus_dir)
    policy_retriever = _local_or_embedding_retriever(policy_document, env, "policy.json")

    if amendment_document is None:
        return policy_retriever
    amendment_retriever = _local_or_embedding_retriever(
        amendment_document, env, "amendment.json"
    )
    return CompositeRetriever(policy_retriever, amendment_retriever)


def _local_or_embedding_retriever(
    document: ParsedDocument,
    env: Mapping[str, str],
    filename: str,
) -> Retriever:
    if not embeddings_requested(env):
        return DeterministicStructureRetriever(document)
    provider = OllamaEmbeddingProvider.from_environ(env)
    provider.ping()
    index_dir = Path(env.get("INDEX_DIR", "").strip() or DEFAULT_INDEX_DIR) / filename.replace(
        ".json", ""
    )
    index = ensure_index((document,), provider, index_dir)
    return EmbeddingRetriever(index, provider)


def _load_policy_document(corpus_dir: Path | None) -> ParsedDocument:
    source_path = (corpus_dir or DEFAULT_CORPUS_DIR) / "policy-manual.md"
    return parse_policy_manual(
        load_policy_text(source_path),
        source_document=source_path.name,
    )


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values
