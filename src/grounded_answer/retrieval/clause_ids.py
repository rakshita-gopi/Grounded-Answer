"""Extract policy clause identifiers from query or retrieved text."""

from __future__ import annotations

import re

CLAUSE_ID_RE = re.compile(r"§\s*(\d+\.\d+\.\d+)")
OPTIONAL_SECTION_SIGN_RE = re.compile(r"§?\s*(\d+\.\d+\.\d+)")


def format_clause_id(number: str) -> str:
    return f"§{number}"


def extract_clause_ids(text: str, *, require_section_sign: bool = False) -> tuple[str, ...]:
    pattern = CLAUSE_ID_RE if require_section_sign else OPTIONAL_SECTION_SIGN_RE
    found: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(text):
        clause_id = format_clause_id(match.group(1))
        if clause_id not in seen:
            seen.add(clause_id)
            found.append(clause_id)
    return tuple(found)
