"""Coordinate retrieval, grounding, and LLM generation into a grounded Answer."""

from grounded_answer.application.query_service import QueryService
from grounded_answer.citations.validator import CitationValidator
from grounded_answer.domain.answer import Answer, GroundingStatus
from grounded_answer.domain.question import Question
from grounded_answer.generation.strategy import (
    citations_for,
    clean_answer_text,
    extract_structured_answer,
    format_with_citations,
    select_supporting_citations,
)
from grounded_answer.grounding.validator import GroundingValidator, overlap_score
from grounded_answer.llm.base import LLMContext, LLMProvider
from grounded_answer.llm.prompts import build_generation_prompt
from grounded_answer.retrieval.clause_ids import extract_clause_ids
from grounded_answer.temporal.extract import extract_temporal_context, merge_temporal
from grounded_answer.temporal.resolver import PolicyApplicabilityResolver

INSUFFICIENT_ANSWER = "I don't know based on the supplied policy manual."
MISSING_DATE_ANSWER = (
    "The applicable policy cannot be determined without the relevant date."
)
CONFLICT_ANSWER = "I cannot determine the applicable rule from the available information."


class AnswerService:
    def __init__(
        self,
        query_service: QueryService,
        llm: LLMProvider,
        grounding_validator: GroundingValidator | None = None,
        citation_validator: CitationValidator | None = None,
        applicability_resolver: PolicyApplicabilityResolver | None = None,
    ) -> None:
        self._query_service = query_service
        self._llm = llm
        self._grounding_validator = grounding_validator or GroundingValidator()
        self._citation_validator = citation_validator or CitationValidator()
        self._resolver = applicability_resolver

    def answer(self, question: Question, top_k: int = 8) -> Answer:
        temporal = merge_temporal(extract_temporal_context(question.text), question.temporal)
        retrieved = self._query_service.evidence_for(question, top_k=top_k)
        unresolved: tuple[str, ...] = ()
        if self._resolver is not None:
            resolution = self._resolver.resolve(retrieved, temporal)
            if resolution.conflict:
                return Answer(
                    text=CONFLICT_ANSWER,
                    citations=(),
                    grounding_status=GroundingStatus.INSUFFICIENT,
                )
            retrieved = resolution.evidence
            unresolved = resolution.unresolved_targets
        assessment = self._grounding_validator.assess(question, retrieved)
        if assessment.status is GroundingStatus.INSUFFICIENT:
            return Answer(
                text=self._abstain_text(question, unresolved, retrieved),
                citations=(),
                grounding_status=GroundingStatus.INSUFFICIENT,
            )
        if unresolved and self._unresolved_blocks_answer(question, assessment.evidence, unresolved):
            return Answer(
                text=MISSING_DATE_ANSWER,
                citations=(),
                grounding_status=GroundingStatus.INSUFFICIENT,
            )
        evidence = assessment.evidence
        structured = extract_structured_answer(question.text, evidence)
        if structured is not None:
            clause_ids = list(structured.clause_ids)
            for clause_id in extract_clause_ids(question.text):
                if clause_id in clause_ids:
                    continue
                if any(item.clause_id == clause_id for item in evidence):
                    clause_ids.append(clause_id)
            citations = citations_for(evidence, clause_ids)
            return Answer(
                text=format_with_citations(structured.sentence, citations),
                citations=citations,
                grounding_status=GroundingStatus.SUPPORTED,
            )
        context = LLMContext(question=question.text, evidence=evidence)
        prompt = build_generation_prompt(question.text, evidence)
        response = self._llm.generate(prompt, context)
        visible = clean_answer_text(response.text, keep_citations=True)
        citation_check = self._citation_validator.validate(visible, evidence)
        mentioned = extract_clause_ids(visible, require_section_sign=True)
        if mentioned and not citation_check.citations:
            return Answer(
                text=INSUFFICIENT_ANSWER,
                citations=(),
                grounding_status=GroundingStatus.INSUFFICIENT,
            )
        if citation_check.citations and mentioned:
            citations = citation_check.citations
        else:
            citations = select_supporting_citations(question.text, visible, evidence)
        text = format_with_citations(visible, citations)
        if not text:
            return Answer(
                text=INSUFFICIENT_ANSWER,
                citations=(),
                grounding_status=GroundingStatus.INSUFFICIENT,
            )
        return Answer(
            text=text,
            citations=citations,
            grounding_status=GroundingStatus.SUPPORTED,
        )

    def _abstain_text(
        self,
        question: Question,
        unresolved: tuple[str, ...],
        remaining: tuple,
    ) -> str:
        if unresolved and self._question_needs_unresolved_clause(question, remaining, unresolved):
            return MISSING_DATE_ANSWER
        return INSUFFICIENT_ANSWER

    def _question_needs_unresolved_clause(
        self,
        question: Question,
        remaining: tuple,
        unresolved: tuple[str, ...],
    ) -> bool:
        unresolved_score = 0
        if self._resolver is not None:
            unresolved_score = max(
                (
                    overlap_score(question.text, self._resolver.clause_content(clause_id))
                    for clause_id in unresolved
                ),
                default=0,
            )
        remaining_score = max(
            (overlap_score(question.text, item.content) for item in remaining),
            default=0,
        )
        return unresolved_score > 1 or remaining_score > 1

    def _unresolved_blocks_answer(
        self,
        question: Question,
        remaining_evidence,
        unresolved: tuple[str, ...],
    ) -> bool:
        remaining_score = max(
            (overlap_score(question.text, item.content) for item in remaining_evidence),
            default=0,
        )
        unresolved_score = 0
        if self._resolver is not None:
            unresolved_score = max(
                (
                    overlap_score(question.text, self._resolver.clause_content(clause_id))
                    for clause_id in unresolved
                ),
                default=0,
            )
        return unresolved_score > remaining_score + 1