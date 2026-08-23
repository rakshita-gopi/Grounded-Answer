from grounded_answer.domain.clause import PolicyClause
from grounded_answer.domain.evidence import Evidence
from grounded_answer.evidence.assembler import EvidenceAssembler
from grounded_answer.evidence.validator import EvidenceValidator
from grounded_answer.retrieval.models import RetrievalHit


def _clause(clause_id: str, content: str) -> PolicyClause:
    return PolicyClause(
        clause_id=clause_id,
        title="Income thresholds",
        content=content,
        source_document="policy-manual.md",
    )


def test_assembler_maps_pageindex_hit_to_canonical_clause() -> None:
    full_text = (
        "A household is not eligible where countable income exceeds the applicable "
        "threshold. The thresholds are — $1,180"
    )
    assembler = EvidenceAssembler([_clause("§6.6.1", full_text)])
    hits = (
        RetrievalHit(
            text="§6.6.1 snippet about income",
            source="pageindex",
            node_id="n1",
            title="Income thresholds",
        ),
    )

    evidence = assembler.assemble(hits)

    assert len(evidence) == 1
    assert evidence[0].clause_id == "§6.6.1"
    assert evidence[0].content == full_text
    assert evidence[0].source == "policy-manual.md"


def test_assembler_uses_hit_clause_id_when_present() -> None:
    assembler = EvidenceAssembler([_clause("§2.4.1", "resources exceed $4,000")])
    hits = (
        RetrievalHit(
            text="partial node text",
            source="policy-manual.md",
            clause_id="§2.4.1",
        ),
    )
    evidence = assembler.assemble(hits)
    assert evidence[0].clause_id == "§2.4.1"
    assert evidence[0].content == "resources exceed $4,000"


def test_assembler_drops_unknown_clause_ids() -> None:
    assembler = EvidenceAssembler([_clause("§2.1.2", "eligibility conditions")])
    hits = (
        RetrievalHit(text="§99.9.9 invented clause", source="policy-manual.md"),
        RetrievalHit(text="§2.1.2 real clause", source="policy-manual.md"),
    )
    evidence = assembler.assemble(hits)
    assert [item.clause_id for item in evidence] == ["§2.1.2"]


def test_assembler_deduplicates_clause_ids() -> None:
    assembler = EvidenceAssembler([_clause("§2.1.2", "conditions")])
    hits = (
        RetrievalHit(text="§2.1.2 first", source="policy-manual.md"),
        RetrievalHit(text="§2.1.2 again", source="policy-manual.md"),
    )
    evidence = assembler.assemble(hits)
    assert len(evidence) == 1


def test_validator_keeps_well_formed_evidence() -> None:
    validator = EvidenceValidator(known_clause_ids={"§6.6.1"})
    kept = validator.validate(
        [
            Evidence(
                evidence_id="policy-manual.md:§6.6.1",
                clause_id="§6.6.1",
                content="thresholds",
                source="policy-manual.md",
            )
        ]
    )
    assert kept[0].clause_id == "§6.6.1"


def test_validator_drops_invented_and_empty_evidence() -> None:
    validator = EvidenceValidator(known_clause_ids={"§2.1.2"})
    kept = validator.validate(
        [
            Evidence(
                evidence_id="bad",
                clause_id="§99.9.9",
                content="invented",
                source="policy-manual.md",
            ),
            Evidence(
                evidence_id="empty",
                clause_id="§2.1.2",
                content="   ",
                source="policy-manual.md",
            ),
            Evidence(
                evidence_id="ok",
                clause_id="§2.1.2",
                content="The conditions are that the person —",
                source="policy-manual.md",
            ),
        ]
    )
    assert [item.clause_id for item in kept] == ["§2.1.2"]
