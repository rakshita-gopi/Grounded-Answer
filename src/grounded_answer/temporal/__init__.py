"""Temporal policy resolution: extract dates, then choose the applicable rule."""

from grounded_answer.temporal.extract import extract_temporal_context, merge_temporal
from grounded_answer.temporal.resolver import (
    ApplicablePolicyRule,
    PolicyApplicabilityResolver,
    ResolutionResult,
)

__all__ = [
    "ApplicablePolicyRule",
    "PolicyApplicabilityResolver",
    "ResolutionResult",
    "extract_temporal_context",
    "merge_temporal",
]
