from pathlib import Path

from grounded_answer.amendments.parser import parse_amendment
from grounded_answer.amendments.service import AmendmentIngestionService
from grounded_answer.domain.policy_change import ApplicabilityBasis, ChangeType
from grounded_answer.ingestion.parser import load_policy_text


def test_loads_amendment_metadata_and_keeps_original_policy_separate(
    repo_root: Path,
) -> None:
    amendment = AmendmentIngestionService(repo_root / "data" / "amendments").load_amendment()
    assert amendment.amendment_id == "2026-01"
    assert amendment.source_document == "Amendment No. 2026-01.md"
    assert amendment.issued_date.isoformat() == "2026-02-12"
    assert amendment.effective_date.isoformat() == "2026-03-01"
    policy_text = (repo_root / "data" / "policy" / "policy-manual.md").read_text(encoding="utf-8")
    assert "$120 per month" in policy_text
    assert "$175 per month" not in policy_text


def test_parses_substitutions_insert_and_table(repo_root: Path) -> None:
    text = load_policy_text(repo_root / "data" / "amendments" / "Amendment No. 2026-01.md")
    issued, effective, paragraphs, changes = parse_amendment(text)
    assert issued.isoformat() == "2026-02-12"
    assert effective.isoformat() == "2026-03-01"
    ids = {paragraph.paragraph_id for paragraph in paragraphs}
    assert {"¶1.1", "¶2.1", "¶2.2", "¶3.1", "¶4.1", "¶4.2", "¶5.1", "¶5.2", "¶5.3"} <= ids

    by_target = {change.target_clause: change for change in changes}
    assert by_target["§6.4.1"].change_type is ChangeType.SUBSTITUTE
    assert by_target["§6.4.1"].previous_rule == "$120 per month"
    assert by_target["§6.4.1"].new_rule == "$175 per month"
    assert by_target["§6.4.1"].applicability is ApplicabilityBasis.DETERMINATION_DATE

    assert by_target["§4.3.2"].previous_rule == "10 calendar days"
    assert by_target["§4.3.2"].new_rule == "14 calendar days"
    assert by_target["§4.3.2"].applicability is ApplicabilityBasis.CHANGE_OF_CIRCUMSTANCES_DATE

    assert by_target["§9.1.4"].previous_rule == "30 calendar days"
    assert by_target["§9.1.4"].new_rule == "14 calendar days"

    assert by_target["§10.5.2"].previous_rule == "20 per cent"
    assert by_target["§10.5.2"].new_rule == "15 per cent"

    assert by_target["§6.6.1"].change_type is ChangeType.REPLACE
    assert "$1,225" in by_target["§6.6.1"].new_rule

    inserted = by_target["§10.5.3A"]
    assert inserted.change_type is ChangeType.INSERT
    assert "would have increased the award" in inserted.new_rule
