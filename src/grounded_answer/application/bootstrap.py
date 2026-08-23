"""Wire application services without exposing adapters to the CLI."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from grounded_answer.application.answer_service import AnswerService
from grounded_answer.application.query_service import QueryService
from grounded_answer.citations.validator import CitationValidator
from grounded_answer.evidence.assembler import EvidenceAssembler
from grounded_answer.evidence.validator import EvidenceValidator
from grounded_answer.ingestion.service import DEFAULT_CORPUS_DIR, IngestionService
from grounded_answer.llm.base import LLMProvider
from grounded_answer.llm.provider import create_llm_provider
from grounded_answer.retrieval.factory import create_retriever
from grounded_answer.retrieval.service import RetrievalService

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_PATH = _REPO_ROOT / ".env"


def create_answer_service(
    corpus_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
    *,
    load_dotenv: bool = False,
    llm: LLMProvider | None = None,
) -> AnswerService:
    env = dict(environ) if environ is not None else dict(os.environ)
    if load_dotenv:
        env = {**_read_env_file(_ENV_PATH), **env}

    corpus = corpus_dir or DEFAULT_CORPUS_DIR
    policy = IngestionService(corpus).load_policy()
    known_ids = {clause.clause_id for clause in policy.clauses}
    retriever = create_retriever(corpus_dir=corpus, environ=env, load_dotenv=False)
    retrieval = RetrievalService(
        retriever,
        EvidenceAssembler(policy.clauses),
        EvidenceValidator(known_clause_ids=known_ids),
    )
    return AnswerService(
        QueryService(retrieval),
        llm or create_llm_provider(env),
        citation_validator=CitationValidator(known_clause_ids=known_ids),
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
