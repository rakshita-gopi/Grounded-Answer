"""Load the original policy corpus into domain objects."""

from __future__ import annotations

import json
from pathlib import Path

from grounded_answer.domain.clause import PolicyClause
from grounded_answer.domain.policy import Policy
from grounded_answer.ingestion.parser import load_policy_text, parse_policy_manual

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[3] / "data" / "policy"


class IngestionService:
    def __init__(self, corpus_dir: Path | None = None) -> None:
        self.corpus_dir = corpus_dir or DEFAULT_CORPUS_DIR

    def load_policy(self) -> Policy:
        manifest_path = self.corpus_dir / "manifest.json"
        source_path = self.corpus_dir / "policy-manual.md"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        text = load_policy_text(source_path)
        parsed = parse_policy_manual(text, source_document=source_path.name)
        clauses = tuple(
            PolicyClause(
                clause_id=clause.clause_id,
                title=clause.title,
                content=clause.content,
                source_document=clause.source_document,
            )
            for clause in parsed.clauses
        )
        return Policy(
            document_id=manifest["document_id"],
            title=manifest["title"],
            document_type=manifest["type"],
            authority=manifest["authority"],
            source_document=source_path.name,
            clauses=clauses,
        )
