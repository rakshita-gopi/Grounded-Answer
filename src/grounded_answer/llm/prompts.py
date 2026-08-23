"""Grounding prompt: question + retrieved evidence + instructions."""

from __future__ import annotations

from collections.abc import Sequence

from grounded_answer.domain.evidence import Evidence
from grounded_answer.llm.base import LLMContext

GROUNDING_INSTRUCTIONS = """1. Use only the supplied evidence.
2. Do not invent policy.
3. Preserve qualifications and exceptions.
4. Cite the relevant clause and, where shown, the amendment paragraph.
5. If evidence is insufficient, explicitly say so.
6. Do not choose between policy versions; the supplied evidence is already the applicable text."""


def format_evidence(evidence: Sequence[Evidence]) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    blocks = []
    for item in evidence:
        meta_parts = [item.source]
        if item.policy_version and item.policy_version != "original":
            meta_parts.append(f"version={item.policy_version}")
        if item.applicability_basis:
            meta_parts.append(f"basis={item.applicability_basis}")
        if item.applicable_period:
            meta_parts.append(f"period={item.applicable_period}")
        header = f"{item.clause_id} ({'; '.join(meta_parts)})"
        blocks.append(f"{header}\n{item.content}")
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
