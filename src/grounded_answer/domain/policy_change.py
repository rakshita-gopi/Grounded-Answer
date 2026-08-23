"""A single change made by an amendment to a policy clause."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChangeType(str, Enum):
    SUBSTITUTE = "SUBSTITUTE"
    REPLACE = "REPLACE"
    INSERT = "INSERT"


class ApplicabilityBasis(str, Enum):
    DETERMINATION_DATE = "determination_date"
    CHANGE_OF_CIRCUMSTANCES_DATE = "change_of_circumstances_date"
    ORIGINAL = "original"
    CROSS_BOUNDARY = "cross_boundary"


@dataclass(frozen=True, slots=True)
class PolicyChange:
    change_id: str
    target_clause: str
    change_type: ChangeType
    previous_rule: str
    new_rule: str
    applicability: ApplicabilityBasis
    source: str
    section_number: int
