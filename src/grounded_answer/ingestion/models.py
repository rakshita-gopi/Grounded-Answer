"""Structured representations produced while ingesting a policy document."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParsedClause:
    clause_id: str
    title: str
    content: str
    part_number: int
    section_identifier: str
    source_document: str


@dataclass(frozen=True, slots=True)
class ParsedSection:
    identifier: str
    title: str
    clauses: tuple[ParsedClause, ...]


@dataclass(frozen=True, slots=True)
class ParsedPart:
    number: int
    title: str
    sections: tuple[ParsedSection, ...]


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    source_document: str
    parts: tuple[ParsedPart, ...]
    clauses: tuple[ParsedClause, ...]
