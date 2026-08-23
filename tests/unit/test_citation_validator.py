from grounded_answer.citations.validator import CitationValidator
from grounded_answer.domain.evidence import Evidence


def _evidence(clause_id: str, content: str = "policy text") -> Evidence:
    return Evidence(
        evidence_id=f"policy-manual.md:{clause_id}",
        clause_id=clause_id,
        content=content,
        source="policy-manual.md",
    )


def test_keeps_citations_that_exist_in_evidence() -> None:
    result = CitationValidator().validate(
        "A person is eligible under §2.1.2.",
        [_evidence("§2.1.2")],
    )
    assert result.accepted
    assert [citation.clause_id for citation in result.citations] == ["§2.1.2"]
    assert result.invented_clause_ids == ()
    assert "§2.1.2" in result.text


def test_drops_invented_clause_ids() -> None:
    result = CitationValidator().validate(
        "See §2.1.2 and §99.9.9.",
        [_evidence("§2.1.2")],
    )
    assert result.accepted
    assert [citation.clause_id for citation in result.citations] == ["§2.1.2"]
    assert result.invented_clause_ids == ("§99.9.9",)
    assert "§99.9.9" not in result.text
    assert "§2.1.2" in result.text


def test_rejects_answer_that_only_cites_invented_ids() -> None:
    result = CitationValidator().validate(
        "The rule is in §99.9.9.",
        [_evidence("§2.1.2")],
    )
    assert not result.accepted
    assert result.citations == ()
    assert result.invented_clause_ids == ("§99.9.9",)


def test_uses_evidence_citations_when_answer_has_no_clause_ids() -> None:
    result = CitationValidator().validate(
        "The household resource limit applies.",
        [_evidence("§2.4.1")],
    )
    assert result.accepted
    assert [citation.clause_id for citation in result.citations] == ["§2.4.1"]


def test_known_corpus_rejects_id_not_in_policy() -> None:
    result = CitationValidator(known_clause_ids={"§2.1.2"}).validate(
        "See §2.1.2 and §2.4.1.",
        [_evidence("§2.1.2"), _evidence("§2.4.1")],
    )
    assert [citation.clause_id for citation in result.citations] == ["§2.1.2"]
    assert "§2.4.1" in result.invented_clause_ids
