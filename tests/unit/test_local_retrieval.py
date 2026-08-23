from pathlib import Path

from grounded_answer.domain.question import Question
from grounded_answer.evidence.assembler import EvidenceAssembler
from grounded_answer.evidence.validator import EvidenceValidator
from grounded_answer.ingestion.parser import load_policy_text, parse_policy_manual
from grounded_answer.ingestion.service import IngestionService
from grounded_answer.retrieval.factory import create_retriever
from grounded_answer.retrieval.local_fallback import DeterministicStructureRetriever
from grounded_answer.retrieval.models import RetrievalQuery
from grounded_answer.retrieval.service import RetrievalService


def test_local_fallback_finds_resource_limit(sample_policy_path: Path) -> None:
    document = parse_policy_manual(load_policy_text(sample_policy_path), "sample_policy.md")
    retriever = DeterministicStructureRetriever(document)
    result = retriever.retrieve(RetrievalQuery(text="countable resources exceed $4,000"))
    assert result
    assert result[0].clause_id == "§2.4.1"
    assert "$4,000" in result[0].text


def test_local_fallback_honours_explicit_clause_id(corpus_dir: Path) -> None:
    document = parse_policy_manual(
        load_policy_text(corpus_dir / "policy-manual.md"),
        "policy-manual.md",
    )
    retriever = DeterministicStructureRetriever(document)
    result = retriever.retrieve(RetrievalQuery(text="See §6.6.1 for the thresholds"))
    assert result[0].clause_id == "§6.6.1"


def test_local_fallback_is_deterministic(corpus_dir: Path) -> None:
    document = parse_policy_manual(
        load_policy_text(corpus_dir / "policy-manual.md"),
        "policy-manual.md",
    )
    retriever = DeterministicStructureRetriever(document)
    query = RetrievalQuery(text="eligibility conditions household")
    first = retriever.retrieve(query)
    second = retriever.retrieve(query)
    assert [item.clause_id for item in first] == [item.clause_id for item in second]


def test_factory_uses_lexical_retriever_without_ollama(corpus_dir: Path) -> None:
    retriever = create_retriever(corpus_dir=corpus_dir, environ={})
    assert isinstance(retriever, DeterministicStructureRetriever)
    policy = IngestionService(corpus_dir).load_policy()
    service = RetrievalService(
        retriever,
        EvidenceAssembler(policy.clauses),
        EvidenceValidator(known_clause_ids={clause.clause_id for clause in policy.clauses}),
    )
    result = service.retrieve(Question(text="income thresholds"), top_k=5)
    assert any(item.clause_id == "§6.6.1" for item in result)
