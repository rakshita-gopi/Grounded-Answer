"""Temporal fields that may be required to choose an applicable policy version."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class TemporalContext:
    determination_date: date | None = None
    claim_start_date: date | None = None
    claim_end_date: date | None = None
    change_of_circumstances_date: date | None = None

    def is_empty(self) -> bool:
        return (
            self.determination_date is None
            and self.claim_start_date is None
            and self.claim_end_date is None
            and self.change_of_circumstances_date is None
        )
