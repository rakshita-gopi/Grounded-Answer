"""Choose a Retriever implementation without exposing PageIndex to callers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from grounded_answer.ingestion.models import ParsedDocument
from grounded_answer.ingestion.parser import load_policy_text, parse_policy_manual
from grounded_answer.ingestion.service import DEFAULT_CORPUS_DIR
from grounded_answer.retrieval.base import Retriever
from grounded_answer.retrieval.composite import CompositeRetriever
from grounded_answer.retrieval.local_fallback import DeterministicStructureRetriever
from grounded_answer.retrieval.pageindex_adapter import (
    PageIndexUnavailableError,
    build_pageindex_retriever,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"


def create_retriever(
    corpus_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
    *,
    load_dotenv: bool = False,
    amendment_document: ParsedDocument | None = None,
) -> Retriever:
    """Return a PageIndex retriever when configured, otherwise the local fallback.

    `load_dotenv` is off by default so tests and callers control configuration
    explicitly. The CLI can turn it on later. An optional amendment document is
    searched through the same Retriever port, not through PageIndex internals.
    """
    env = dict(environ) if environ is not None else dict(os.environ)
    if load_dotenv:
        env = {**_read_env_file(DEFAULT_ENV_PATH), **env}

    api_key = env.get("PAGEINDEX_API_KEY", "").strip()
    doc_id = env.get("PAGEINDEX_DOC_ID", "").strip()
    policy_retriever: Retriever | None = None
    if api_key and doc_id:
        try:
            policy_retriever = build_pageindex_retriever(api_key=api_key, doc_id=doc_id)
        except PageIndexUnavailableError:
            policy_retriever = None
    if policy_retriever is None:
        policy_retriever = _local_policy_retriever(corpus_dir)

    if amendment_document is None:
        return policy_retriever
    amendment_retriever = DeterministicStructureRetriever(amendment_document)
    return CompositeRetriever(policy_retriever, amendment_retriever)


def _local_policy_retriever(corpus_dir: Path | None) -> Retriever:
    source_path = (corpus_dir or DEFAULT_CORPUS_DIR) / "policy-manual.md"
    document = parse_policy_manual(
        load_policy_text(source_path),
        source_document=source_path.name,
    )
    return DeterministicStructureRetriever(document)


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
