"""Grounding prompt: question + retrieved evidence + instructions."""

from __future__ import annotations

from collections.abc import Sequence

from grounded_answer.domain.evidence import Evidence
from grounded_answer.llm.base import LLMContext

GROUNDING_INSTRUCTIONS = """1. Use only the supplied evidence.
2. Do not invent policy.
3. Preserve qualifications and exceptions.
4. Cite the relevant clause.
5. If evidence is insufficient, explicitly say so."""


def format_evidence(evidence: Sequence[Evidence]) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    blocks = []
    for item in evidence:
        blocks.append(f"{item.clause_id} ({item.source})\n{item.content}")
    return "\n\n".join(blocks)


def build_generation_prompt(question: str, evidence: Sequence[Evidence]) -> str:
    return (
        "USER QUESTION\n"
        f"{question.strip()}\n\n"
        "RETRIEVED POLICY EVIDENCE\n"
        f"{format_evidence(evidence)}\n\n"
        "GROUNDING INSTRUCTIONS\n"
        f"{GROUNDING_INSTRUCTIONS}"
    )


def prompt_from_context(context: LLMContext) -> str:
    return build_generation_prompt(context.question, context.evidence)
