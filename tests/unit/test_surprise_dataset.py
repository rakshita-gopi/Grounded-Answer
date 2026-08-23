import json
from pathlib import Path

from grounded_answer.amendments.service import AmendmentIngestionService
from grounded_answer.ingestion.service import IngestionService


def test_surprise_dataset_ids_match_and_expected_facts_exist(repo_root: Path, corpus_dir: Path) -> None:
    questions = json.loads(
        (repo_root / "evaluation" / "surprise" / "questions.json").read_text(encoding="utf-8")
    )
    expected = json.loads(
        (repo_root / "evaluation" / "surprise" / "expected.json").read_text(encoding="utf-8")
    )
    ids = [item["id"] for item in questions["questions"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == set(expected)

    types = {item["type"] for item in questions["questions"]}
    assert {
        "original-rule-before-amendment",
        "new-rule-after-amendment",
        "determination-date",
        "change-of-circumstances",
        "transitional",
        "cross-boundary",
        "missing-date",
        "unsupported",
    } <= types

    policy = IngestionService(corpus_dir).load_policy()
    amendment = AmendmentIngestionService(repo_root / "data" / "amendments").load_amendment()
    known_ids = {clause.clause_id for clause in policy.clauses}
    known_ids.update(change.target_clause for change in amendment.changes)
    original_text = "\n".join(clause.content for clause in policy.clauses)
    amendment_text = "\n".join(paragraph.content for paragraph in amendment.paragraphs)

    for item in questions["questions"]:
        record = expected[item["id"]]
        if record["abstain"]:
            assert record["grounding"] == "INSUFFICIENT"
            continue
        assert record["required_clause_ids"]
        for clause_id in record["required_clause_ids"]:
            assert clause_id in known_ids
        for phrase in record["must_contain"]:
            assert phrase in original_text or phrase in amendment_text or phrase in {
                "$175",
                "$1,225",
                "14 calendar days",
                "15 per cent",
                "must not be imposed",
            }
