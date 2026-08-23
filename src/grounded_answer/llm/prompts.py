"""Grounding prompt: question + retrieved evidence + instructions."""

from __future__ import annotations

from collections.abc import Sequence

from grounded_answer.domain.evidence import Evidence
from grounded_answer.llm.base import LLMContext

GENERATION_SYSTEM_PROMPT = (
    "Answer the question in one or two short plain-English sentences. "
    "Use only the supplied evidence. Do not cite clauses; the application adds citations. "
    "Do not write chain-of-thought, analysis, or preamble. /no_think"
)

GROUNDING_INSTRUCTIONS = """Write one or two concise sentences that answer the question.

The supplied evidence has already been retrieved and resolved for temporal applicability. Do not perform your own policy applicability reasoning.

You MUST:
- Use only the supplied evidence.
- answer directly in simple natural English
- Preserve exact policy numbers, dates, percentages, dollar amounts, and conditions from the evidence
- Preserve qualifications and exceptions.
- give a concise direct answer
- If evidence is insufficient, explicitly say so.

You MUST NOT:
- explain reasoning, repeat the question, or discuss retrieval or evidence processing
- begin with planning language such as "Okay", "Let me", "The user is asking", "I need to", or "First, I"
- mention the model, embeddings, or implementation details
- Do not use outside knowledge or world facts.
- Do not invent policy.
- invent a citation or policy section
- return JSON
- return markdown headings or tables
- choose between policy versions; the supplied evidence is already the applicable text
- Cite the relevant clause; the application appends validated citations

Output only the final plain-English answer. Do not add [§...] citations."""


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
