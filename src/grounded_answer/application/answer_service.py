"""Coordinate retrieval, evidence, and LLM generation into a grounded Answer."""

from grounded_answer.application.query_service import QueryService
from grounded_answer.domain.answer import Answer, GroundingStatus
from grounded_answer.domain.citation import Citation
from grounded_answer.domain.evidence import Evidence
from grounded_answer.domain.question import Question
from grounded_answer.llm.base import LLMContext, LLMProvider
from grounded_answer.llm.prompts import build_generation_prompt

INSUFFICIENT_ANSWER = "I don't know based on the supplied policy manual."


class AnswerService:
    def __init__(self, query_service: QueryService, llm: LLMProvider) -> None:
        self._query_service = query_service
        self._llm = llm

    def answer(self, question: Question, top_k: int = 8) -> Answer:
        evidence = self._query_service.evidence_for(question, top_k=top_k)
        if not evidence:
            return Answer(
                text=INSUFFICIENT_ANSWER,
                citations=(),
                grounding_status=GroundingStatus.INSUFFICIENT,
            )
        context = LLMContext(question=question.text, evidence=evidence)
        prompt = build_generation_prompt(question.text, evidence)
        response = self._llm.generate(prompt, context)
        return Answer(
            text=response.text,
            citations=_citations_from_evidence(evidence),
            grounding_status=GroundingStatus.SUPPORTED,
        )


def _citations_from_evidence(evidence: tuple[Evidence, ...]) -> tuple[Citation, ...]:
    seen: set[str] = set()
    citations: list[Citation] = []
    for item in evidence:
        if item.clause_id in seen:
            continue
        seen.add(item.clause_id)
        citations.append(
            Citation(source_document=item.source, clause_id=item.clause_id)
        )
    return tuple(citations)
