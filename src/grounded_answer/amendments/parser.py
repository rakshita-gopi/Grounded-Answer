"""Parse Amendment No. 2026-01 markdown into structured paragraphs and changes."""

from __future__ import annotations

import re
from datetime import date

from grounded_answer.domain.amendment import AmendmentParagraph
from grounded_answer.domain.policy_change import ApplicabilityBasis, ChangeType, PolicyChange
from grounded_answer.ingestion.models import ParsedClause, ParsedDocument, ParsedPart, ParsedSection

ISSUED_RE = re.compile(r"\*\*Issued:\*\*\s*(.+)$", re.IGNORECASE)
EFFECTIVE_RE = re.compile(r"\*\*Effective:\*\*\s*(.+)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^## (\d+)\.\s+(.+)$")
PARAGRAPH_RE = re.compile(r"^\*\*(\d+\.\d+)\*\*\s*(.*)$")
SUBSTITUTE_RE = re.compile(
    r'In\s+(§\d+\.\d+\.\d+)(?:\([a-z]\))?\s*,\s*for\s+"([^"]+)"'
    r'(?:\s+\([^)]+\))?\s+substitute\s+"?\*\*(.+?)\*\*"?',
    re.IGNORECASE | re.DOTALL,
)
INSERT_RE = re.compile(
    r"After\s+(§\d+\.\d+\.\d+)\s*,\s*insert",
    re.IGNORECASE,
)
INSERTED_CLAUSE_RE = re.compile(
    r">\s*\*\*(\d+\.\d+\.\d+[A-Za-z]?)\*\*\s*(.+)",
    re.DOTALL,
)
MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
LONG_DATE_RE = re.compile(
    r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
    re.IGNORECASE,
)


class AmendmentParseError(ValueError):
    """Raised when the amendment cannot be parsed as structured policy changes."""


def parse_long_date(text: str) -> date:
    match = LONG_DATE_RE.search(text.strip())
    if not match:
        raise AmendmentParseError(f"Could not parse date: {text!r}")
    return date(int(match.group(3)), MONTHS[match.group(2).lower()], int(match.group(1)))


def parse_amendment(
    text: str,
    source_document: str = "Amendment No. 2026-01.md",
) -> tuple[date, date, tuple[AmendmentParagraph, ...], tuple[PolicyChange, ...]]:
    issued_date: date | None = None
    effective_date: date | None = None
    current_section: int | None = None
    current_title = ""
    paragraph_id: str | None = None
    paragraph_body: list[str] = []
    paragraphs: list[AmendmentParagraph] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_id, paragraph_body
        if paragraph_id is None or current_section is None:
            paragraph_id = None
            paragraph_body = []
            return
        content = "\n".join(paragraph_body).strip()
        if not content:
            raise AmendmentParseError(f"Amendment paragraph {paragraph_id} has no content.")
        paragraphs.append(
            AmendmentParagraph(
                paragraph_id=f"¶{paragraph_id}",
                title=current_title,
                content=content,
                source_document=source_document,
                section_number=current_section,
            )
        )
        paragraph_id = None
        paragraph_body = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        issued_match = ISSUED_RE.search(line)
        if issued_match:
            issued_date = parse_long_date(issued_match.group(1))
            continue
        effective_match = EFFECTIVE_RE.search(line)
        if effective_match:
            effective_date = parse_long_date(effective_match.group(1))
            continue
        section_match = SECTION_RE.match(line)
        if section_match:
            flush_paragraph()
            current_section = int(section_match.group(1))
            current_title = section_match.group(2).strip()
            continue
        paragraph_match = PARAGRAPH_RE.match(line)
        if paragraph_match:
            flush_paragraph()
            paragraph_id = paragraph_match.group(1)
            remainder = paragraph_match.group(2) or ""
            paragraph_body = [remainder] if remainder else []
            continue
        if paragraph_id is not None:
            paragraph_body.append(line)

    flush_paragraph()
    if issued_date is None or effective_date is None:
        raise AmendmentParseError("Amendment issued date and effective date are required.")
    if not paragraphs:
        raise AmendmentParseError("No amendment paragraphs were found.")

    determination_sections, change_sections = _applicability_sections(paragraphs)
    changes = tuple(
        change
        for paragraph in paragraphs
        for change in _changes_from_paragraph(
            paragraph,
            source_document,
            determination_sections,
            change_sections,
        )
    )
    if not changes:
        raise AmendmentParseError("No policy changes were found in the amendment.")
    return issued_date, effective_date, tuple(paragraphs), changes


def amendment_as_parsed_document(
    paragraphs: tuple[AmendmentParagraph, ...],
    inserted_clauses: tuple[tuple[str, str, str], ...] = (),
    source_document: str = "Amendment No. 2026-01.md",
) -> ParsedDocument:
    """Index amendment paragraphs (and inserted clauses) for structure-aware retrieval."""
    clauses: list[ParsedClause] = []
    for paragraph in paragraphs:
        clauses.append(
            ParsedClause(
                clause_id=paragraph.paragraph_id,
                title=paragraph.title,
                content=paragraph.content,
                part_number=paragraph.section_number,
                section_identifier=str(paragraph.section_number),
                source_document=paragraph.source_document,
            )
        )
    for clause_id, title, content in inserted_clauses:
        clauses.append(
            ParsedClause(
                clause_id=clause_id,
                title=title,
                content=content,
                part_number=10,
                section_identifier="10.5",
                source_document=source_document,
            )
        )
    if not clauses:
        raise AmendmentParseError("Amendment document has no retrievable clauses.")
    section = ParsedSection(identifier="amendment", title="Amendment", clauses=tuple(clauses))
    part = ParsedPart(number=0, title="Amendment", sections=(section,))
    return ParsedDocument(
        source_document=source_document,
        parts=(part,),
        clauses=tuple(clauses),
    )


_PARAGRAPH_REF_RE = re.compile(r"paragraphs?\s+([\d, and]+)", re.IGNORECASE)


def _paragraph_numbers_referred(text: str) -> set[int]:
    match = _PARAGRAPH_REF_RE.search(text)
    if not match:
        return set()
    return {int(value) for value in re.findall(r"\d+", match.group(1))}


def _applicability_sections(
    paragraphs: tuple[AmendmentParagraph, ...] | list[AmendmentParagraph],
) -> tuple[frozenset[int], frozenset[int]]:
    determination: set[int] = set()
    change_of_circumstances: set[int] = set()
    for paragraph in paragraphs:
        if paragraph.section_number != 5:
            continue
        numbers = _paragraph_numbers_referred(paragraph.content)
        lowered = paragraph.content.lower()
        if "determination" in lowered:
            determination.update(numbers or {1, 3, 4})
        if "change of circumstances" in lowered:
            change_of_circumstances.update(numbers or {2})
    if not determination:
        determination = {1, 3, 4}
    if not change_of_circumstances:
        change_of_circumstances = {2}
    return frozenset(determination), frozenset(change_of_circumstances)


def _basis_for_section(
    section_number: int,
    determination_sections: frozenset[int],
    change_sections: frozenset[int],
) -> ApplicabilityBasis | None:
    if section_number in change_sections:
        return ApplicabilityBasis.CHANGE_OF_CIRCUMSTANCES_DATE
    if section_number in determination_sections:
        return ApplicabilityBasis.DETERMINATION_DATE
    return None


def _changes_from_paragraph(
    paragraph: AmendmentParagraph,
    source_document: str,
    determination_sections: frozenset[int],
    change_sections: frozenset[int],
) -> tuple[PolicyChange, ...]:
    basis = _basis_for_section(paragraph.section_number, determination_sections, change_sections)
    if basis is None:
        return ()
    content = paragraph.content
    changes: list[PolicyChange] = []

    insert_match = INSERT_RE.search(content)
    inserted = INSERTED_CLAUSE_RE.search(content)
    if insert_match and inserted:
        clause_number = inserted.group(1)
        changes.append(
            PolicyChange(
                change_id=paragraph.paragraph_id,
                target_clause=f"§{clause_number}",
                change_type=ChangeType.INSERT,
                previous_rule="",
                new_rule=_strip_md(inserted.group(2)),
                applicability=basis,
                source=source_document,
                section_number=paragraph.section_number,
            )
        )
        return tuple(changes)

    table = _extract_markdown_table(content)
    target_match = re.search(r"§(\d+\.\d+\.\d+)", content)
    if table and target_match and "substitute the following" in content.lower():
        changes.append(
            PolicyChange(
                change_id=paragraph.paragraph_id,
                target_clause=f"§{target_match.group(1)}",
                change_type=ChangeType.REPLACE,
                previous_rule="",
                new_rule=table,
                applicability=basis,
                source=source_document,
                section_number=paragraph.section_number,
            )
        )
        return tuple(changes)

    for match in SUBSTITUTE_RE.finditer(content):
        changes.append(
            PolicyChange(
                change_id=paragraph.paragraph_id,
                target_clause=match.group(1),
                change_type=ChangeType.SUBSTITUTE,
                previous_rule=match.group(2),
                new_rule=_strip_md(match.group(3)),
                applicability=basis,
                source=source_document,
                section_number=paragraph.section_number,
            )
        )
    return tuple(changes)


def _extract_markdown_table(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    start = None
    for index, line in enumerate(lines):
        if line.startswith("|"):
            start = index
            break
    if start is None:
        return ""
    end = start
    for index in range(start, len(lines)):
        if lines[index].startswith("|"):
            end = index
        elif lines[index].strip():
            break
    return "\n".join(lines[start : end + 1]).strip()


def _strip_md(value: str) -> str:
    cleaned = value.strip().strip(".").strip()
    cleaned = cleaned.replace("**", "")
    return cleaned.strip().strip('"')
