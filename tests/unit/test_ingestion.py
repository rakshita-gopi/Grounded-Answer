from pathlib import Path

from grounded_answer.ingestion.parser import load_policy_text, parse_policy_manual
from grounded_answer.ingestion.service import IngestionService


def _by_id(clauses):
    return {clause.clause_id: clause for clause in clauses}


class TestDocumentLoading:
    def test_loads_utf8_policy_manual(self, corpus_dir: Path) -> None:
        path = corpus_dir / "policy-manual.md"
        text = load_policy_text(path)
        assert "Household Support Program" in text
        assert "§4.3.2" in text

    def test_service_loads_manifest_and_source(self, corpus_dir: Path) -> None:
        policy = IngestionService(corpus_dir).load_policy()
        assert policy.document_id == "policy-manual"
        assert policy.title == "Policy Manual"
        assert policy.document_type == "policy"
        assert policy.authority == "primary"
        assert policy.source_document == "policy-manual.md"
        assert policy.clauses


class TestClauseDetection:
    def test_sample_recognises_parts_sections_and_clauses(self, sample_policy_path: Path) -> None:
        parsed = parse_policy_manual(load_policy_text(sample_policy_path), "sample_policy.md")
        assert [part.number for part in parsed.parts] == [1, 2, 6]
        assert [section.identifier for part in parsed.parts for section in part.sections] == [
            "1.1",
            "1.4",
            "2.4",
            "6.6",
        ]
        assert [clause.clause_id for clause in parsed.clauses] == [
            "§1.1.1",
            "§1.1.2",
            "§1.4.1",
            "§2.4.1",
            "§6.6.1",
        ]

    def test_lettered_items_are_not_separate_clauses(self, sample_policy_path: Path) -> None:
        parsed = parse_policy_manual(load_policy_text(sample_policy_path), "sample_policy.md")
        assert "§(a)" not in {clause.clause_id for clause in parsed.clauses}
        body = _by_id(parsed.clauses)["§1.1.2"].content
        assert "(a) is resident in Calder County;" in body
        assert "(b) is aged 18 or over." in body

    def test_preamble_is_not_treated_as_a_clause(self, sample_policy_path: Path) -> None:
        parsed = parse_policy_manual(load_policy_text(sample_policy_path), "sample_policy.md")
        combined = " ".join(clause.content for clause in parsed.clauses)
        assert "Preamble text is not a numbered clause." not in combined


class TestClauseIds:
    def test_sample_ids_use_section_sign_prefix(self, sample_policy_path: Path) -> None:
        parsed = parse_policy_manual(load_policy_text(sample_policy_path), "sample_policy.md")
        assert all(clause.clause_id.startswith("§") for clause in parsed.clauses)

    def test_original_manual_includes_specified_clause_ids(self, corpus_dir: Path) -> None:
        policy = IngestionService(corpus_dir).load_policy()
        ids = {clause.clause_id for clause in policy.clauses}
        assert "§2.1.2" in ids
        assert "§2.4.1" in ids
        assert "§6.6.1" in ids

    def test_original_manual_clause_ids_are_unique(self, corpus_dir: Path) -> None:
        policy = IngestionService(corpus_dir).load_policy()
        ids = [clause.clause_id for clause in policy.clauses]
        assert ids == sorted(ids, key=ids.index)
        assert len(ids) == len(set(ids))

    def test_definition_clause_keeps_two_digit_paragraph_number(self, corpus_dir: Path) -> None:
        policy = IngestionService(corpus_dir).load_policy()
        ids = {clause.clause_id for clause in policy.clauses}
        assert "§1.4.10" in ids
        assert "§1.4.11" in ids


class TestClauseTextPreservation:
    def test_sample_preserves_clause_wording_and_table(self, sample_policy_path: Path) -> None:
        parsed = parse_policy_manual(load_policy_text(sample_policy_path), "sample_policy.md")
        clauses = _by_id(parsed.clauses)
        assert clauses["§2.4.1"].content.startswith(
            "A household is not eligible where the total countable resources of the household exceed $4,000."
        )
        assert "$1,180" in clauses["§6.6.1"].content
        assert "| Household size | Monthly threshold |" in clauses["§6.6.1"].content
        assert clauses["§1.4.1"].title == "Applicant"
        assert "a person who has submitted an application." in clauses["§1.4.1"].content

    def test_original_manual_preserves_key_figures_and_lists(self, corpus_dir: Path) -> None:
        policy = IngestionService(corpus_dir).load_policy()
        clauses = _by_id(policy.clauses)
        assert "$4,000" in clauses["§2.4.1"].content
        assert "10 calendar days" in clauses["§4.3.2"].content
        assert "(e) is not excluded under Part 4; and" in clauses["§2.1.2"].content
        assert "$1,180" in clauses["§6.6.1"].content
        assert "each additional member" in clauses["§6.6.1"].content
        assert clauses["§1.4.1"].title == "Applicant"


class TestDeterminism:
    def test_repeated_parse_is_identical(self, corpus_dir: Path) -> None:
        first = IngestionService(corpus_dir).load_policy()
        second = IngestionService(corpus_dir).load_policy()
        assert first == second
        assert [clause.clause_id for clause in first.clauses] == [
            clause.clause_id for clause in second.clauses
        ]
