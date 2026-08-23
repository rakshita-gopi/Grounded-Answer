"""Structure-aware parser for the Household Support Program policy manual."""

from __future__ import annotations

import re
from pathlib import Path

from grounded_answer.ingestion.models import (
    ParsedClause,
    ParsedDocument,
    ParsedPart,
    ParsedSection,
)

PART_RE = re.compile(r"^# Part (\d+)\s+[—–-]\s+(.+)$")
SECTION_RE = re.compile(r"^## (\d+\.\d+)\s+(.+)$")
CLAUSE_RE = re.compile(r"^\*\*(\d+\.\d+\.\d+)(?:\s+([^*]+))?\*\*\s*(.*)$")
END_RE = re.compile(r"^\*End of consolidated text")


class PolicyParseError(ValueError):
    """Raised when the document does not contain the expected Part/Section/Clause structure."""


def load_policy_text(path: Path) -> str:
    """Load a policy markdown file as UTF-8 text."""
    return path.read_text(encoding="utf-8-sig")


def parse_policy_manual(
    text: str,
    source_document: str = "policy-manual.md",
) -> ParsedDocument:
    """Parse policy markdown into Parts, Sections, and Clauses.

    Clause identifiers follow the manual's own convention (§4.3.2 = Part 4,
    section 3, paragraph 2). Lettered items such as (a) remain inside the
    parent clause rather than becoming separate identifiers.
    """
    parts: list[ParsedPart] = []
    current_part_number: int | None = None
    current_part_title: str | None = None
    current_sections: list[tuple[str, str, list[ParsedClause]]] = []

    current_section_id: str | None = None
    current_section_title: str | None = None
    current_section_clauses: list[ParsedClause] = []

    clause_id: str | None = None
    clause_title: str | None = None
    clause_body: list[str] = []
    seen_ids: set[str] = set()

    def flush_clause() -> None:
        nonlocal clause_id, clause_title, clause_body
        if clause_id is None:
            return
        if current_part_number is None or current_section_id is None:
            raise PolicyParseError(f"Clause {clause_id} appears outside a part or section.")
        content = "\n".join(clause_body).strip()
        if not content:
            raise PolicyParseError(f"Clause {clause_id} has no content.")
        if clause_id in seen_ids:
            raise PolicyParseError(f"Duplicate clause identifier: {clause_id}")
        seen_ids.add(clause_id)
        current_section_clauses.append(
            ParsedClause(
                clause_id=clause_id,
                title=clause_title or current_section_title or "",
                content=content,
                part_number=current_part_number,
                section_identifier=current_section_id,
                source_document=source_document,
            )
        )
        clause_id = None
        clause_title = None
        clause_body = []

    def flush_section() -> None:
        nonlocal current_section_id, current_section_title, current_section_clauses
        flush_clause()
        if current_section_id is None:
            return
        current_sections.append(
            (current_section_id, current_section_title or "", list(current_section_clauses))
        )
        current_section_id = None
        current_section_title = None
        current_section_clauses = []

    def flush_part() -> None:
        nonlocal current_part_number, current_part_title, current_sections
        flush_section()
        if current_part_number is None:
            return
        sections = tuple(
            ParsedSection(
                identifier=section_id,
                title=section_title,
                clauses=tuple(section_clauses),
            )
            for section_id, section_title, section_clauses in current_sections
        )
        parts.append(
            ParsedPart(
                number=current_part_number,
                title=current_part_title or "",
                sections=sections,
            )
        )
        current_part_number = None
        current_part_title = None
        current_sections = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if END_RE.match(line):
            break
        if line.strip() == "---":
            continue

        part_match = PART_RE.match(line)
        if part_match:
            flush_part()
            current_part_number = int(part_match.group(1))
            current_part_title = part_match.group(2).strip()
            continue

        section_match = SECTION_RE.match(line)
        if section_match:
            flush_section()
            current_section_id = section_match.group(1)
            current_section_title = section_match.group(2).strip()
            continue

        clause_match = CLAUSE_RE.match(line)
        if clause_match:
            flush_clause()
            number = clause_match.group(1)
            term = (clause_match.group(2) or "").strip()
            remainder = clause_match.group(3) or ""
            clause_id = f"§{number}"
            clause_title = term or current_section_title
            if remainder:
                clause_body.append(remainder)
            continue

        if clause_id is not None:
            clause_body.append(line)

    flush_part()

    clauses = tuple(
        clause
        for part in parts
        for section in part.sections
        for clause in section.clauses
    )
    if not clauses:
        raise PolicyParseError("No policy clauses were found.")

    return ParsedDocument(
        source_document=source_document,
        parts=tuple(parts),
        clauses=clauses,
    )
