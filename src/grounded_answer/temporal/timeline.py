"""Split a claim period around an amendment effective date."""

from __future__ import annotations

from datetime import date, timedelta


def format_period(start: date, end: date) -> str:
    return f"{start.isoformat()}/{end.isoformat()}"


def split_claim_period(
    start: date,
    end: date,
    boundary: date,
) -> tuple[tuple[date, date, str], ...]:
    if end < start:
        raise ValueError("Claim end date must not precede claim start date.")
    if end < boundary:
        return ((start, end, "original"),)
    if start >= boundary:
        return ((start, end, "amended"),)
    original_end = boundary - timedelta(days=1)
    return (
        (start, original_end, "original"),
        (boundary, end, "amended"),
    )
