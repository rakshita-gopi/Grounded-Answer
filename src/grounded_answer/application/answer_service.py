"""Coordinate retrieval, grounding, and LLM generation into a grounded Answer."""

from grounded_answer.application.query_service import QueryService
from grounded_answer.citations.validator import CitationValidator
from grounded_answer.domain.answer import Answer, GroundingStatus
from grounded_answer.domain.question import Question
from grounded_answer.grounding.validator import GroundingValidator
from grounded_answer.llm.base import LLMContext, LLMProvider
from grounded_answer.llm.prompts import build_generation_prompt

INSUFFICIENT_ANSWER = "I don't know based on the supplied policy manual."


class AnswerService:
    def __init__(
        self,
        query_service: QueryService,
        llm: LLMProvider,
        grounding_validator: GroundingValidator | None = None,
        citation_validator: CitationValidator | None = None,
    ) -> None:
        self._query_service = query_service
        self._llm = llm
        self._grounding_validator = grounding_validator or GroundingValidator()
        self._citation_validator = citation_validator or CitationValidator()

    def answer(self, question: Question, top_k: int = 8) -> Answer:
        retrieved = self._query_service.evidence_for(question, top_k=top_k)
        assessment = self._grounding_validator.assess(question, retrieved)
        if assessment.status is GroundingStatus.INSUFFICIENT:
            return Answer(
                text=INSUFFICIENT_ANSWER,
                citations=(),
                grounding_status=GroundingStatus.INSUFFICIENT,
            )
        evidence = assessment.evidence
        context = LLMContext(question=question.text, evidence=evidence)
        prompt = build_generation_prompt(question.text, evidence)
        response = self._llm.generate(prompt, context)
        citation_check = self._citation_validator.validate(response.text, evidence)
        if not citation_check.accepted:
            return Answer(
                text=INSUFFICIENT_ANSWER,
                citations=(),
                grounding_status=GroundingStatus.INSUFFICIENT,
            )
        return Answer(
            text=citation_check.text,
            citations=citation_check.citations,
            grounding_status=GroundingStatus.SUPPORTED,
        )
