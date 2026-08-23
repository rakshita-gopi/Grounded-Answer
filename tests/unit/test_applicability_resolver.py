from datetime import date
from pathlib import Path

from grounded_answer.amendments.service import AmendmentIngestionService
from grounded_answer.domain.evidence import Evidence
from grounded_answer.domain.temporal import TemporalContext
from grounded_answer.ingestion.service import IngestionService
from grounded_answer.temporal.resolver import PolicyApplicabilityResolver


def _resolver(repo_root: Path) -> PolicyApplicabilityResolver:
    policy = IngestionService(repo_root / "data" / "policy").load_policy()
    amendment = AmendmentIngestionService(repo_root / "data" / "amendments").load_amendment()
    return PolicyApplicabilityResolver(policy, amendment)


def _hit(clause_id: str, content: str, source: str = "policy-manual.md") -> Evidence:
    return Evidence(
        evidence_id=f"{source}:{clause_id}",
        clause_id=clause_id,
        content=content,
        source=source,
    )


def test_february_determination_keeps_original_earnings(repo_root: Path) -> None:
    resolver = _resolver(repo_root)
    original = resolver.clause_content("§6.4.1")
    result = resolver.resolve(
        [_hit("§6.4.1", original)],
        TemporalContext(determination_date=date(2026, 2, 15)),
    )
    item = next(row for row in result.evidence if row.clause_id == "§6.4.1")
    assert "$120 per month" in item.content
    assert "$175 per month" not in item.content
    assert result.unresolved_targets == ()


def test_march_determination_applies_amended_earnings(repo_root: Path) -> None:
    resolver = _resolver(repo_root)
    original = resolver.clause_content("§6.4.1")
    result = resolver.resolve(
        [_hit("§6.4.1", original)],
        TemporalContext(determination_date=date(2026, 3, 15)),
    )
    item = next(row for row in result.evidence if row.clause_id == "§6.4.1")
    assert "$175 per month" in item.content
    assert "$120 per month" not in item.content


def test_later_determination_of_earlier_period_uses_new_figures(repo_root: Path) -> None:
    resolver = _resolver(repo_root)
    original = resolver.clause_content("§6.4.1")
    result = resolver.resolve(
        [_hit("§6.4.1", original)],
        TemporalContext(
            determination_date=date(2026, 3, 15),
            claim_start_date=date(2026, 2, 1),
            claim_end_date=date(2026, 2, 28),
        ),
    )
    item = next(row for row in result.evidence if row.clause_id == "§6.4.1")
    assert "$175 per month" in item.content


def test_reporting_depends_on_change_date_not_determination(repo_root: Path) -> None:
    resolver = _resolver(repo_root)
    original = resolver.clause_content("§4.3.2")
    before = resolver.resolve(
        [_hit("§4.3.2", original)],
        TemporalContext(
            determination_date=date(2026, 3, 20),
            change_of_circumstances_date=date(2026, 2, 10),
        ),
    )
    after = resolver.resolve(
        [_hit("§4.3.2", original)],
        TemporalContext(change_of_circumstances_date=date(2026, 3, 5)),
    )
    before_text = next(row.content for row in before.evidence if row.clause_id == "§4.3.2")
    after_text = next(row.content for row in after.evidence if row.clause_id == "§4.3.2")
    assert "10 calendar days" in before_text
    assert "14 calendar days" in after_text


def test_cross_boundary_claim_segments_figures(repo_root: Path) -> None:
    resolver = _resolver(repo_root)
    original = resolver.clause_content("§6.4.1")
    result = resolver.resolve(
        [_hit("§6.4.1", original)],
        TemporalContext(claim_start_date=date(2026, 2, 20), claim_end_date=date(2026, 3, 10)),
    )
    item = next(row for row in result.evidence if row.clause_id == "§6.4.1")
    assert "$120 per month" in item.content
    assert "$175 per month" in item.content
    assert any(row.clause_id == "§7.4.3" for row in result.evidence)
    assert item.applicability_basis == "cross_boundary"


def test_missing_date_leaves_amended_clause_unresolved(repo_root: Path) -> None:
    resolver = _resolver(repo_root)
    original = resolver.clause_content("§6.4.1")
    result = resolver.resolve([_hit("§6.4.1", original)], TemporalContext())
    assert "§6.4.1" in result.unresolved_targets
    assert all(row.clause_id != "§6.4.1" for row in result.evidence)


def test_insert_is_omitted_before_effective_date(repo_root: Path) -> None:
    resolver = _resolver(repo_root)
    result = resolver.resolve(
        [_hit("§10.5.2", resolver.clause_content("§10.5.2"))],
        TemporalContext(determination_date=date(2026, 2, 15)),
    )
    assert all(row.clause_id != "§10.5.3A" for row in result.evidence)
