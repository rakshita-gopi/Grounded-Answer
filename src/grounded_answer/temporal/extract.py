"""Extract temporal context from a question. Does not invent missing dates."""

from __future__ import annotations

import calendar
import re
from datetime import date

from grounded_answer.domain.temporal import TemporalContext

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
MONTH_ALT = "|".join(MONTHS)
ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
DMY_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_ALT})\s+(\d{{4}})\b",
    re.IGNORECASE,
)
MDY_RE = re.compile(
    rf"\b({MONTH_ALT})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b",
    re.IGNORECASE,
)
MONTH_YEAR_RE = re.compile(rf"\b({MONTH_ALT})\s+(\d{{4}})\b", re.IGNORECASE)
RANGE_RE = re.compile(
    r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:[.?,;]|$)",
    re.IGNORECASE,
)
DETERMINATION_HINT = re.compile(
    r"\b(determination|determined|determining|decision made)\b",
    re.IGNORECASE,
)
CHANGE_HINT = re.compile(
    r"\b(change of circumstances|circumstances changed|change occurring|"
    r"reporting period|must report|to report)\b",
    re.IGNORECASE,
)
CLAIM_HINT = re.compile(
    r"\b(claim|award period|period spanning|relates to a period|in force)\b",
    re.IGNORECASE,
)


def last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def parse_date_token(text: str) -> date | None:
    text = text.strip().strip(".,;:")
    iso = ISO_RE.fullmatch(text) or ISO_RE.search(text)
    if iso and ISO_RE.fullmatch(text.strip()):
        return _safe_date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
    dmy = DMY_RE.fullmatch(text) or DMY_RE.search(text)
    if dmy and DMY_RE.fullmatch(text.strip()):
        return _safe_date(int(dmy.group(3)), MONTHS[dmy.group(2).lower()], int(dmy.group(1)))
    mdy = MDY_RE.fullmatch(text) or MDY_RE.search(text)
    if mdy and MDY_RE.fullmatch(text.strip()):
        return _safe_date(int(mdy.group(3)), MONTHS[mdy.group(1).lower()], int(mdy.group(2)))
    return None


def extract_temporal_context(text: str) -> TemporalContext:
    range_match = RANGE_RE.search(text)
    if range_match:
        start = parse_date_token(range_match.group(1))
        end = parse_date_token(range_match.group(2))
        if start and end:
            return TemporalContext(claim_start_date=start, claim_end_date=end)

    dated = _dated_mentions(text)
    month_periods = _month_year_periods(text, dated)

    determination: date | None = None
    change: date | None = None
    claim_start: date | None = None
    claim_end: date | None = None

    for start, end, kind in dated:
        if kind == "determination":
            determination = start
        elif kind == "change":
            change = start
        elif kind == "claim":
            claim_start = start
            claim_end = end
        elif kind == "unspecified":
            if DETERMINATION_HINT.search(text):
                determination = determination or start
            elif CHANGE_HINT.search(text):
                change = change or start
            elif CLAIM_HINT.search(text):
                claim_start = claim_start or start
                claim_end = claim_end or end
            else:
                claim_start = claim_start or start
                claim_end = claim_end or end
                if DETERMINATION_HINT.search(text) is None and start == end:
                    determination = determination or start

    for start, end in month_periods:
        if CHANGE_HINT.search(text) and change is None:
            change = start
        elif DETERMINATION_HINT.search(text) and determination is None:
            determination = start
        else:
            claim_start = claim_start or start
            claim_end = claim_end or end

    return TemporalContext(
        determination_date=determination,
        claim_start_date=claim_start,
        claim_end_date=claim_end,
        change_of_circumstances_date=change,
    )


def merge_temporal(
    extracted: TemporalContext,
    explicit: TemporalContext | None,
) -> TemporalContext:
    if explicit is None or explicit.is_empty():
        return extracted
    return TemporalContext(
        determination_date=explicit.determination_date or extracted.determination_date,
        claim_start_date=explicit.claim_start_date or extracted.claim_start_date,
        claim_end_date=explicit.claim_end_date or extracted.claim_end_date,
        change_of_circumstances_date=(
            explicit.change_of_circumstances_date or extracted.change_of_circumstances_date
        ),
    )


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _window_kind(prefix: str) -> str:
    if DETERMINATION_HINT.search(prefix):
        return "determination"
    if CHANGE_HINT.search(prefix):
        return "change"
    if CLAIM_HINT.search(prefix):
        return "claim"
    return "unspecified"


def _dated_mentions(text: str) -> list[tuple[date, date, str]]:
    found: list[tuple[date, date, str]] = []
    occupied: list[tuple[int, int]] = []

    def add(match: re.Match[str], parsed: date | None) -> None:
        if parsed is None:
            return
        occupied.append((match.start(), match.end()))
        prefix = text[max(0, match.start() - 80) : match.start()]
        found.append((parsed, parsed, _window_kind(prefix)))

    for match in ISO_RE.finditer(text):
        add(match, _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3))))
    for match in DMY_RE.finditer(text):
        add(match, _safe_date(int(match.group(3)), MONTHS[match.group(2).lower()], int(match.group(1))))
    for match in MDY_RE.finditer(text):
        add(match, _safe_date(int(match.group(3)), MONTHS[match.group(1).lower()], int(match.group(2))))
    return found


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(not (end <= left or start >= right) for left, right in occupied)


def _month_year_periods(text: str, dated: list[tuple[date, date, str]]) -> list[tuple[date, date]]:
    occupied_dates = {item[0] for item in dated}
    periods: list[tuple[date, date]] = []
    for match in MONTH_YEAR_RE.finditer(text):
        year = int(match.group(2))
        month = MONTHS[match.group(1).lower()]
        start = date(year, month, 1)
        end = date(year, month, last_day_of_month(year, month))
        # Skip "1 March 2026" already captured as a full date.
        surrounding = text[max(0, match.start() - 3) : match.end()]
        if DMY_RE.search(surrounding) or MDY_RE.search(surrounding):
            continue
        if start in occupied_dates:
            continue
        periods.append((start, end))
    return periods
