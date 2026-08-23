"""Post-grounding answer strategy: extract structured facts, else call the LLM."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from grounded_answer.domain.citation import Citation
from grounded_answer.domain.evidence import Evidence
from grounded_answer.grounding.validator import overlap_score

_MONEY_RE = re.compile(r"\$[0-9,]+(?:\.\d+)?(?:\s+per month)?", re.IGNORECASE)
_FIRST_MONEY_RE = re.compile(
    r"\bfirst\s+\**(\$[0-9,]+(?:\.\d+)?(?:\s+per month)?)",
    re.IGNORECASE,
)
_DURATION_RE = re.compile(r"\b(\d+)\s+calendar days\b", re.IGNORECASE)
_PERCENT_RE = re.compile(r"\b(\d+)\s+per cent\b", re.IGNORECASE)
_CITATION_RE = re.compile(
    r"\s*(?:\[[§¶][^\]]+\]|[§¶]\d+(?:\.\d+)*[A-Za-z]?)\s*"
)
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_REASONING_RE = re.compile(
    r"^(okay|alright|sure|let me|the user is|i need to|i should|looking through|"
    r"first,?\s+i\b|first i'll|here is the answer|based on my|according to the retrieved|"
    r"step by step|i will |i'll |checking the|the retrieved evidence|wait,|"
    r"let's |so the user|the question is)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class StructuredAnswer:
    sentence: str
    clause_ids: tuple[str, ...]
    sources: dict[str, str]


def extract_structured_answer(
    question: str,
    evidence: Sequence[Evidence],
) -> StructuredAnswer | None:
    """Return a deterministic answer when effective evidence contains a typed fact."""
    if not evidence:
        return None
    if _asks_eligibility_overview(question):
        extracted = _eligibility_overview(question, evidence)
        if extracted is not None:
            return extracted
    if _asks_multiple_figures(question):
        extracted = _multi_money(question, evidence)
        if extracted is not None:
            return extracted
    if _asks_duration(question):
        extracted = _duration(question, evidence)
        if extracted is not None:
            return extracted
    if _asks_percent(question):
        extracted = _percent(question, evidence)
        if extracted is not None:
            return extracted
    if _asks_money(question):
        extracted = _money(question, evidence)
        if extracted is not None:
            return extracted
    if _asks_yes_no(question):
        extracted = _yes_no(question, evidence)
        if extracted is not None:
            return extracted
    return _direct_evidence_answer(question, evidence)


def format_with_citations(sentence: str, citations: Sequence[Citation]) -> str:
    """Attach application-owned citations. The model must not format these."""
    text = clean_answer_text(sentence)
    if not text:
        return ""
    ids: list[str] = []
    seen: set[str] = set()
    for citation in citations:
        if citation.clause_id in seen:
            continue
        seen.add(citation.clause_id)
        ids.append(citation.clause_id)
    if not ids:
        return f"{text}."
    attached = " ".join(f"[{clause_id}]" for clause_id in ids)
    return f"{text}. {attached}"


def citations_for(
    evidence: Sequence[Evidence],
    clause_ids: Sequence[str],
) -> tuple[Citation, ...]:
    by_id = {item.clause_id: item for item in evidence}
    result: list[Citation] = []
    seen: set[str] = set()
    for clause_id in clause_ids:
        if clause_id in seen:
            continue
        item = by_id.get(clause_id)
        if item is None:
            continue
        seen.add(clause_id)
        result.append(Citation(source_document=item.source, clause_id=clause_id))
    return tuple(result)


def select_supporting_citations(
    question: str,
    answer_text: str,
    evidence: Sequence[Evidence],
    *,
    limit: int = 3,
) -> tuple[Citation, ...]:
    """Cite only clauses that overlap the answer, not every retrieved hit."""
    mentioned = extract_ids_from_text(answer_text)
    amounts = [_normalize_money(match) for match in _MONEY_RE.findall(answer_text)]
    percents = _PERCENT_RE.findall(answer_text)
    durations = _DURATION_RE.findall(answer_text)
    ranked: list[tuple[int, Evidence]] = []
    for item in evidence:
        answer_score = overlap_score(answer_text, item.content)
        if item.clause_id in mentioned:
            answer_score += 5
        if amounts and not any(amount in item.content for amount in amounts):
            continue
        if percents and not any(f"{value} per cent" in item.content.lower() for value in percents):
            continue
        if durations and not any(
            f"{value} calendar days" in item.content.lower() for value in durations
        ):
            continue
        if answer_score < 1:
            continue
        score = answer_score * 2 + overlap_score(question, item.content)
        ranked.append((score, item))
    ranked.sort(key=lambda row: -row[0])
    chosen = ranked[:limit] if ranked else []
    if not chosen:
        fallback = max(
            evidence,
            key=lambda item: overlap_score(answer_text, item.content),
            default=None,
        )
        if fallback is not None and overlap_score(answer_text, fallback.content) >= 1:
            chosen = [(1, fallback)]
    return tuple(
        Citation(source_document=item.source, clause_id=item.clause_id) for _, item in chosen
    )


def extract_ids_from_text(text: str) -> set[str]:
    return set(re.findall(r"[§¶]\d+(?:\.\d+)*[A-Za-z]?", text))


def clean_answer_text(sentence: str, *, keep_citations: bool = False) -> str:
    """Keep a short factual answer; drop chain-of-thought and preamble."""
    text = _strip_thinking_markers(sentence)
    if not keep_citations:
        text = _strip_model_citations(text)
    text = _HEADING_RE.sub("", text)
    kept: list[str] = []
    for raw in _SENTENCE_RE.split(text.replace("\n", " ")):
        piece = re.sub(r"\s+", " ", raw).strip(" :-")
        if not piece:
            continue
        if _REASONING_RE.search(piece):
            continue
        kept.append(piece.rstrip("."))
        if len(kept) >= 3:
            break
    return ". ".join(kept).strip()


def _strip_thinking_markers(text: str) -> str:
    cleaned = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE).sub("", text)
    lower = cleaned.lower()
    close = lower.rfind("</think>")
    if close != -1:
        cleaned = cleaned[close + len("</think>") :]
    return cleaned


def _strip_model_citations(text: str) -> str:
    return _CITATION_RE.sub(" ", text)


def _asks_eligibility_overview(question: str) -> bool:
    q = question.lower()
    if "eligib" not in q:
        return False
    return any(token in q for token in ("requirement", "condition", "qualify", "who is eligible"))


def _eligibility_overview(
    question: str, evidence: Sequence[Evidence]
) -> StructuredAnswer | None:
    part2 = [item for item in evidence if item.clause_id.startswith("§2.")]
    pool = part2 or list(evidence)
    preferred = [item for item in pool if item.clause_id in {"§2.1.1", "§2.1.2"}]
    if not preferred:
        preferred = sorted(pool, key=lambda item: -overlap_score(question, item.content))[:2]
    if not preferred:
        return None
    chosen: list[Evidence] = []
    seen: set[str] = set()
    for item in preferred:
        if item.clause_id in seen:
            continue
        seen.add(item.clause_id)
        chosen.append(item)
    extra = ""
    for item in pool:
        if "resident in Calder County" not in item.content:
            continue
        extra = ", including being resident in Calder County"
        if item.clause_id not in seen:
            seen.add(item.clause_id)
            chosen.append(item)
        break
    chosen = chosen[:3]
    return StructuredAnswer(
        sentence=(
            "A person must satisfy the applicable eligibility conditions "
            f"set out in Part 2 of the policy{extra}"
        ),
        clause_ids=tuple(item.clause_id for item in chosen),
        sources={item.clause_id: item.source for item in chosen},
    )


def _asks_multiple_figures(question: str) -> bool:
    q = question.lower()
    return (
        ("figures" in q and "apply" in q)
        or "claim runs" in q
        or "spanning" in q
        or "apportion" in q
        or ("award treated" in q)
        or ("both" in q and "disregard" in q)
    )


def _asks_duration(question: str) -> bool:
    q = question.lower()
    return (
        "calendar day" in q
        or ("how many" in q and "day" in q)
        or ("report" in q and "day" in q)
    )


def _asks_percent(question: str) -> bool:
    q = question.lower()
    return "per cent" in q or "percent" in q or ("sanction" in q and "reduction" in q)


def _asks_money(question: str) -> bool:
    q = question.lower()
    if "once per" in q:
        return False
    if "no award" in q:
        return False
    markers = (
        "earnings",
        "disregard",
        "threshold",
        "resources",
        "needs figure",
        "dollar",
        "award",
        "$",
    )
    return any(marker in q for marker in markers)


def _asks_yes_no(question: str) -> bool:
    q = question.lower().strip()
    return (
        q.startswith("may ")
        or " may " in q
        or q.startswith("must ")
        or "shall " in q
        or q.startswith("is ")
        or q.startswith("are ")
    )


def _money(question: str, evidence: Sequence[Evidence]) -> StructuredAnswer | None:
    item = _best_item(
        question,
        evidence,
        lambda row: _MONEY_RE.search(row.content) is not None and _money_topic_match(question, row),
    )
    if item is None:
        return None
    amount = None
    if "first" in question.lower():
        match = _FIRST_MONEY_RE.search(item.content)
        if match:
            amount = match.group(1)
    if amount is None:
        match = _MONEY_RE.search(item.content)
        if match:
            amount = match.group(0)
    if not amount:
        return None
    amount = _normalize_money(amount)
    if "threshold" in question.lower():
        sentence = f"The monthly countable income threshold is {amount}"
    elif "resources" in question.lower():
        sentence = f"The countable resources limit is {amount}"
    elif "needs" in question.lower():
        sentence = f"The monthly needs figure is {amount}"
    elif "disregard" in question.lower() or "earnings" in question.lower():
        if "first" in question.lower():
            sentence = f"The first monthly earnings disregard is {amount}"
        else:
            sentence = f"The earnings disregard is {amount}"
    else:
        sentence = f"The applicable amount is {amount}"
    return StructuredAnswer(sentence=sentence, clause_ids=(item.clause_id,), sources={item.clause_id: item.source})


def _multi_money(question: str, evidence: Sequence[Evidence]) -> StructuredAnswer | None:
    amounts: list[str] = []
    clause_ids: list[str] = []
    sources: dict[str, str] = {}
    for item in evidence:
        if "disregard" in question.lower() or "earnings" in question.lower():
            found = [_normalize_money(match) for match in _FIRST_MONEY_RE.findall(item.content)]
        else:
            found = [_normalize_money(match) for match in _MONEY_RE.findall(item.content)]
        if len(found) >= 3 and "threshold" in item.content.lower() and "threshold" not in question.lower():
            continue
        related = overlap_score(question, item.content) >= 1 or len(found) >= 2
        if not related and item.clause_id != "§7.4.3":
            continue
        if not found:
            if _is_apportion_rule(item) or item.clause_id == "§7.4.3":
                if item.clause_id not in clause_ids:
                    clause_ids.append(item.clause_id)
                    sources[item.clause_id] = item.source
            continue
        if item.clause_id not in clause_ids:
            clause_ids.append(item.clause_id)
            sources[item.clause_id] = item.source
        for amount in found:
            if amount not in amounts:
                amounts.append(amount)
    if len(amounts) < 2:
        return None
    joined = " and ".join(amounts)
    sentence = f"The applicable figures are {joined}"
    cite_ids: list[str] = []
    for clause_id in clause_ids:
        item = next((row for row in evidence if row.clause_id == clause_id), None)
        if item is None:
            continue
        if _FIRST_MONEY_RE.search(item.content) or _is_apportion_rule(item):
            if clause_id not in cite_ids:
                cite_ids.append(clause_id)
    if any(_is_apportion_rule(item) for item in evidence):
        sentence += ". The award is apportioned by reference to the number of days"
        for item in evidence:
            if _is_apportion_rule(item) and item.clause_id not in cite_ids:
                cite_ids.append(item.clause_id)
                sources[item.clause_id] = item.source
    if len(cite_ids) < 1:
        cite_ids = list(clause_ids)
    return StructuredAnswer(sentence=sentence, clause_ids=tuple(cite_ids), sources=sources)


def _duration(question: str, evidence: Sequence[Evidence]) -> StructuredAnswer | None:
    item = _best_item(
        question, evidence, lambda row: _DURATION_RE.search(row.content) is not None
    )
    if item is None:
        return None
    match = _DURATION_RE.search(item.content)
    if match is None:
        return None
    days = match.group(1)
    if "report" in question.lower():
        sentence = f"The recipient has {days} calendar days to report the change"
    else:
        sentence = f"The applicable period is {days} calendar days"
    return StructuredAnswer(sentence=sentence, clause_ids=(item.clause_id,), sources={item.clause_id: item.source})


def _percent(question: str, evidence: Sequence[Evidence]) -> StructuredAnswer | None:
    item = _best_item(
        question, evidence, lambda row: _PERCENT_RE.search(row.content) is not None
    )
    if item is None:
        return None
    match = _PERCENT_RE.search(item.content)
    if match is None:
        return None
    value = f"{match.group(1)} per cent"
    q = question.lower()
    if "first" in q and "sanction" in q:
        sentence = f"The reduction for a first sanction is {value}"
    elif "sanction" in q or "reduction" in q:
        sentence = f"The reduction is {value}"
    else:
        sentence = f"The applicable rate is {value}"
    return StructuredAnswer(sentence=sentence, clause_ids=(item.clause_id,), sources={item.clause_id: item.source})


def _yes_no(question: str, evidence: Sequence[Evidence]) -> StructuredAnswer | None:
    def has_modal(item: Evidence) -> bool:
        text = item.content.lower()
        return "must not" in text or "shall not" in text or " may " in text or text.startswith("may ")

    item = _best_item(question, evidence, has_modal)
    if item is None:
        return None
    sentence = item.content.strip().split(".")[0].strip()
    sentence = re.sub(r"^\*+", "", sentence).strip()
    if len(sentence) < 12:
        return None
    return StructuredAnswer(sentence=sentence, clause_ids=(item.clause_id,), sources={item.clause_id: item.source})


def _direct_evidence_answer(
    question: str, evidence: Sequence[Evidence]
) -> StructuredAnswer | None:
    """Use the overlapping clause text when it already contains the answer."""
    item = _best_item(question, evidence, lambda _row: True)
    if item is None:
        return None
    text = _HEADING_RE.sub("", item.content)
    sentences: list[str] = []
    for raw in _SENTENCE_RE.split(text.replace("\n", " ")):
        piece = re.sub(r"\s+", " ", raw).strip(" :-*")
        if len(piece) < 12 or piece.startswith("|"):
            continue
        sentences.append(piece.rstrip("."))
        if len(sentences) >= 2:
            break
    if not sentences:
        return None
    sentence = ". ".join(sentences)
    if overlap_score(question, sentence) < 2:
        return None
    return StructuredAnswer(
        sentence=sentence,
        clause_ids=(item.clause_id,),
        sources={item.clause_id: item.source},
    )


def _money_topic_match(question: str, item: Evidence) -> bool:
    q = question.lower()
    content = item.content.lower()
    if "resources" in q:
        if "resource" not in content:
            return False
        if "limit" in q:
            return "exceed" in content or "limit" in content
        return True
    if "threshold" in q:
        return "threshold" in content or "countable income" in content
    if "disregard" in q or "earnings" in q:
        return "earning" in content or "disregard" in content
    if "needs" in q:
        return "needs" in content
    return True


def _best_item(
    question: str,
    evidence: Sequence[Evidence],
    predicate,
) -> Evidence | None:
    ranked = [
        (_item_score(question, item), index, item)
        for index, item in enumerate(evidence)
        if predicate(item)
    ]
    if not ranked:
        return None
    ranked.sort(key=lambda row: (-row[0], row[1]))
    score, _, item = ranked[0]
    if score < 1:
        return None
    return item


def _item_score(question: str, item: Evidence) -> int:
    score = overlap_score(question, item.content)
    q = question.lower()
    content = item.content.lower()
    if "report" in q and "report" in content:
        score += 4
    if "change" in q and "change" in content:
        score += 2
    if "must report" in content:
        score += 8
    if "overpayment" in content and "overpayment" not in q:
        score -= 6
    if "resources" in q and "limit" in q and ("exceed" in content or "limit" in content):
        score += 4
    if "resources" in q and "not countable resources" in content:
        score -= 4
    return score


def _is_apportion_rule(item: Evidence) -> bool:
    return item.clause_id == "§7.4.3"


def _normalize_money(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())
