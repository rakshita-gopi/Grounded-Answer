"""A user question to be answered from the policy corpus."""

from __future__ import annotations

from dataclasses import dataclass

from grounded_answer.domain.temporal import TemporalContext


@dataclass(frozen=True, slots=True)
class Question:
    text: str
    temporal: TemporalContext | None = None
