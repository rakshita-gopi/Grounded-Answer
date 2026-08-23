from grounded_answer.domain.citation import Citation
from grounded_answer.domain.evidence import Evidence
from grounded_answer.generation.strategy import (
    clean_answer_text,
    extract_structured_answer,
    format_with_citations,
    select_supporting_citations,
)


def _item(clause_id: str, content: str) -> Evidence:
    return Evidence(
        evidence_id=f"policy:{clause_id}",
        clause_id=clause_id,
        content=content,
        source="policy-manual.md",
    )


def test_resource_limit_prefers_exceed_clause_over_exclusions() -> None:
    evidence = (
        _item(
            "§2.4.2",
            "The following are not countable resources — the surrender value of a life insurance policy where that value does not exceed $1,500.",
        ),
        _item("§2.4.1", "A household is not eligible where the total countable resources of the household exceed $4,000."),
    )
    result = extract_structured_answer(
        "What is the countable resources limit for a household?",
        evidence,
    )
    assert result is not None
    assert "$4,000" in result.sentence
    assert "$1,500" not in result.sentence
    assert result.clause_ids == ("§2.4.1",)


def test_extracts_first_money_amount_from_effective_clause() -> None:
    evidence = (
        _item("§6.4.1", "the first $175 per month of household earnings from employment"),
        _item("§2.4.1", "countable resources must not exceed $4,000"),
    )
    result = extract_structured_answer(
        "what is the first monthly earnings disregard?",
        evidence,
    )
    assert result is not None
    assert "$175 per month" in result.sentence
    assert "$4,000" not in result.sentence
    assert result.clause_ids == ("§6.4.1",)


def test_extracts_duration_from_effective_clause() -> None:
    evidence = (
        _item(
            "§4.3.2",
            "A recipient must report any change within 14 calendar days of the change occurring.",
        ),
    )
    result = extract_structured_answer(
        "How many calendar days does the recipient have to report the change?",
        evidence,
    )
    assert result is not None
    assert "14 calendar days" in result.sentence
    assert result.clause_ids == ("§4.3.2",)


def test_extracts_multiple_money_amounts_and_apportionment() -> None:
    evidence = (
        _item(
            "§6.4.1",
            "Period before: the first $120 per month of household earnings. "
            "Period after: the first $175 per month of household earnings. "
            "a care allowance, to the extent of $200 per month.",
        ),
        _item("§7.4.3", "An award is apportioned by reference to the number of days."),
        _item(
            "§6.6.1",
            "Applicable figures differ across 1 March 2026. "
            "The award is apportioned under §7.4.3 by reference to the number of days. "
            "A household is not eligible where countable income exceeds the applicable threshold. $1,180 $1,225",
        ),
    )
    result = extract_structured_answer(
        "What earnings disregard figures apply, and how is the award treated?",
        evidence,
    )
    assert result is not None
    assert "$120 per month" in result.sentence
    assert "$175 per month" in result.sentence
    assert "$200" not in result.sentence
    assert "apportioned" in result.sentence
    assert "§6.4.1" in result.clause_ids
    assert "§7.4.3" in result.clause_ids
    assert "§6.6.1" not in result.clause_ids


def test_application_appends_validated_citations() -> None:
    text = format_with_citations(
        "The first monthly earnings disregard is $175 per month",
        (Citation(source_document="policy-manual.md", clause_id="§6.4.1"),),
    )
    assert text == "The first monthly earnings disregard is $175 per month. [§6.4.1]"


def test_format_strips_model_citations_before_appending() -> None:
    text = format_with_citations(
        "The disregard is $120 per month. §99.9.9",
        (Citation(source_document="policy-manual.md", clause_id="§6.4.1"),),
    )
    assert "§99.9.9" not in text
    assert text.endswith("[§6.4.1]")


def test_eligibility_overview_stays_concise_and_cites_part_2() -> None:
    evidence = (
        _item("§2.1.1", "A person is eligible if that person satisfies each of the conditions in §2.1.2."),
        _item(
            "§2.1.2",
            "The conditions are that the person is resident in Calder County and satisfies Part 3.",
        ),
        _item("§6.4.1", "the first $175 per month of household earnings from employment"),
    )
    result = extract_structured_answer("What are the eligibility requirements?", evidence)
    assert result is not None
    assert "Part 2 of the policy" in result.sentence
    assert "resident in Calder County" in result.sentence
    assert "Okay" not in result.sentence
    assert "§2.1.2" in result.clause_ids
    assert "§6.4.1" not in result.clause_ids


def test_first_sanction_wording_and_percent() -> None:
    evidence = (
        _item(
            "§10.5.2",
            "A sanction is a reduction of the monthly award by 15 per cent for a period of 4 weeks for a first sanction.",
        ),
        _item("§6.4.1", "the first $175 per month of household earnings from employment"),
    )
    result = extract_structured_answer(
        "For a determination made on 2 March 2026, what is the reduction for a first sanction?",
        evidence,
    )
    assert result is not None
    assert result.sentence == "The reduction for a first sanction is 15 per cent"
    assert result.clause_ids == ("§10.5.2",)


def test_clean_answer_text_drops_reasoning_preamble() -> None:
    text = clean_answer_text(
        "Okay, let me tackle this. The user is asking about the limit. "
        "The countable resources limit is $4,000."
    )
    assert text == "The countable resources limit is $4,000"
    assert "Okay" not in text
    assert "The user is asking" not in text


def test_select_supporting_citations_ignores_unrelated_hits() -> None:
    evidence = (
        _item("§2.4.1", "countable resources must not exceed $4,000"),
        _item("§6.6.1", "A household is not eligible where countable income exceeds the applicable threshold."),
        _item("§7.4.3", "An award is apportioned by reference to the number of days."),
    )
    citations = select_supporting_citations(
        "What is the countable resources limit for a household?",
        "The countable resources limit is $4,000",
        evidence,
    )
    assert [citation.clause_id for citation in citations] == ["§2.4.1"]
