import json
from pathlib import Path

from grounded_answer.ingestion.service import IngestionService


def test_evaluation_dataset_ids_match_and_clauses_exist(repo_root: Path, corpus_dir: Path) -> None:
    questions = json.loads(
        (repo_root / "evaluation" / "original" / "questions.json").read_text(encoding="utf-8")
    )
    expected = json.loads(
        (repo_root / "evaluation" / "original" / "expected.json").read_text(encoding="utf-8")
    )

    ids = [item["id"] for item in questions["questions"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == set(expected)

    types = {item["type"] for item in questions["questions"]}
    assert types == {
        "direct",
        "multi-condition",
        "cross-reference",
        "exact-clause",
        "unsupported",
    }

    policy = IngestionService(corpus_dir).load_policy()
    known_ids = {clause.clause_id for clause in policy.clauses}
    corpus_text = "\n".join(clause.content for clause in policy.clauses)

    for item in questions["questions"]:
        record = expected[item["id"]]
        if item["type"] == "unsupported":
            assert record["abstain"] is True
            assert record["grounding"] == "INSUFFICIENT"
            assert record["required_clause_ids"] == []
            continue
        assert record["abstain"] is False
        assert record["grounding"] == "SUPPORTED"
        assert record["required_clause_ids"]
        for clause_id in record["required_clause_ids"]:
            assert clause_id in known_ids
        for phrase in record["must_contain"]:
            assert phrase in corpus_text
