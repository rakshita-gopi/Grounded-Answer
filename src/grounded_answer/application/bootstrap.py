"""Wire application services without exposing adapters to the CLI."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from grounded_answer.amendments.parser import amendment_as_parsed_document
from grounded_answer.amendments.service import (
    AmendmentIngestionService,
    DEFAULT_AMENDMENTS_DIR,
)
from grounded_answer.application.answer_service import AnswerService
from grounded_answer.application.query_service import QueryService
from grounded_answer.citations.validator import CitationValidator
from grounded_answer.domain.clause import PolicyClause
from grounded_answer.domain.policy_change import ChangeType
from grounded_answer.evidence.assembler import EvidenceAssembler
from grounded_answer.evidence.validator import EvidenceValidator
from grounded_answer.ingestion.service import DEFAULT_CORPUS_DIR, IngestionService
from grounded_answer.llm.base import LLMProvider
from grounded_answer.llm.provider import create_llm_provider
from grounded_answer.retrieval.factory import create_retriever
from grounded_answer.retrieval.service import RetrievalService
from grounded_answer.temporal.resolver import PolicyApplicabilityResolver

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_PATH = _REPO_ROOT / ".env"


def create_answer_service(
    corpus_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
    *,
    load_dotenv: bool = False,
    llm: LLMProvider | None = None,
    amendments_dir: Path | None = None,
    load_amendments: bool = True,
) -> AnswerService:
    env = dict(environ) if environ is not None else dict(os.environ)
    if load_dotenv:
        env = {**_read_env_file(_ENV_PATH), **env}

    corpus = corpus_dir or DEFAULT_CORPUS_DIR
    policy = IngestionService(corpus).load_policy()
    amendment = None
    amendment_document = None
    extra_clauses: list[PolicyClause] = []
    if load_amendments:
        amendment = AmendmentIngestionService(
            amendments_dir or DEFAULT_AMENDMENTS_DIR
        ).load_amendment()
        inserted = tuple(
            (change.target_clause, change.target_clause, change.new_rule)
            for change in amendment.changes
            if change.change_type is ChangeType.INSERT
        )
        amendment_document = amendment_as_parsed_document(
            amendment.paragraphs,
            inserted_clauses=inserted,
            source_document=amendment.source_document,
        )
        extra_clauses = [
            PolicyClause(
                clause_id=clause.clause_id,
                title=clause.title,
                content=clause.content,
                source_document=clause.source_document,
            )
            for clause in amendment_document.clauses
        ]

    known_ids = {clause.clause_id for clause in policy.clauses}
    known_ids.update(clause.clause_id for clause in extra_clauses)
    retriever = create_retriever(
        corpus_dir=corpus,
        environ=env,
        load_dotenv=False,
        amendment_document=amendment_document,
    )
    retrieval = RetrievalService(
        retriever,
        EvidenceAssembler((*policy.clauses, *extra_clauses)),
        EvidenceValidator(known_clause_ids=known_ids),
    )
    resolver = (
        PolicyApplicabilityResolver(policy, amendment) if amendment is not None else None
    )
    return AnswerService(
        QueryService(retrieval),
        llm or create_llm_provider(env),
        citation_validator=CitationValidator(known_clause_ids=known_ids),
        applicability_resolver=resolver,
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
