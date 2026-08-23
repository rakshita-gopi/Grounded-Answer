from datetime import date

from grounded_answer.domain.temporal import TemporalContext
from grounded_answer.temporal.extract import extract_temporal_context, merge_temporal


def test_extracts_determination_date() -> None:
    temporal = extract_temporal_context(
        "What happens for a determination made after March 1, 2026?"
    )
    assert temporal.determination_date == date(2026, 3, 1)


def test_extracts_change_of_circumstances_date() -> None:
    temporal = extract_temporal_context(
        "The claimant's circumstances changed on March 5, 2026. Which rule applies?"
    )
    assert temporal.change_of_circumstances_date == date(2026, 3, 5)


def test_extracts_february_claim_period() -> None:
    temporal = extract_temporal_context(
        "What was the applicable allowance in February 2026?"
    )
    assert temporal.claim_start_date == date(2026, 2, 1)
    assert temporal.claim_end_date == date(2026, 2, 28)
    assert temporal.determination_date is None


def test_extracts_claim_range() -> None:
    temporal = extract_temporal_context(
        "The claim runs from 20 February 2026 to 10 March 2026."
    )
    assert temporal.claim_start_date == date(2026, 2, 20)
    assert temporal.claim_end_date == date(2026, 3, 10)


def test_does_not_invent_a_date_when_none_is_present() -> None:
    temporal = extract_temporal_context("What is the first monthly earnings disregard?")
    assert temporal.is_empty()


def test_explicit_cli_dates_override_extracted_gaps() -> None:
    extracted = extract_temporal_context("What is the earnings disregard?")
    merged = merge_temporal(
        extracted,
        TemporalContext(determination_date=date(2026, 3, 15)),
    )
    assert merged.determination_date == date(2026, 3, 15)
