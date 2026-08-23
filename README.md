# Grounded Answer

A CLI-first system that answers questions about Calder County's Household Support Program policy.

Answers must be grounded in the supplied policy corpus, include citations to the supporting clauses, and abstain when the corpus does not support an answer.

## Current status

**Stage A baseline.** The original policy corpus is loaded, retrieval and evidence assembly work, grounding and citation checks run before an answer is accepted, and the CLI and evaluation runner are executable. Amendments, temporal resolution, MCP, and an HTTP API are not implemented.

## Problem

Policy questions need answers that can be traced to numbered clauses (for example `§2.1.2`). The system must not invent policy, and it must say so when the supplied manual does not support an answer.

## Solution

1. Ingest `data/policy/policy-manual.md` into Parts, Sections, and Clauses.
2. Retrieve structure-aware hits (PageIndex when configured; otherwise a deterministic local fallback).
3. Assemble canonical `Evidence` from ingested clauses.
4. Decide SUPPORTED or INSUFFICIENT before calling the LLM.
5. Generate from question + evidence + grounding instructions.
6. Keep only citations that exist in retrieved evidence and in the corpus.

## Key features

- Clause identifiers such as `§2.1.2`, `§2.4.1`, and `§6.6.1`
- Retrieval behind a `Retriever` port (no PageIndex types in application code)
- Grounding abstention for missing or off-topic evidence
- Rejection of invented citations such as `§99.9.9`
- CLI: `python -m grounded_answer ask "..."`
- Evaluation: `python scripts/evaluate.py`

## Architecture

```text
CLI
  → AnswerService
      → QueryService → Retriever → EvidenceAssembler → EvidenceValidator
      → GroundingValidator  (SUPPORTED | INSUFFICIENT)
      → LLMProvider
      → CitationValidator
  → printed answer / evidence / grounding
```

## Technology stack

- Python 3.10+
- pytest
- Optional: `pageindex` and an LLM provider (`openai` / `openai-compatible`)

## Repository structure

```text
data/policy/          original corpus and manifest
src/grounded_answer/  domain, ingestion, retrieval, evidence, grounding, citations, LLM, application, CLI
tests/unit/           unit tests
tests/integration/    question → retrieval → evidence → answer
evaluation/original/  Stage A questions and expected facts
scripts/evaluate.py   Stage A evaluation runner
```

## Prerequisites

- Python 3.10 or later

## Installation

From the repository root:

```text
pip install -r requirements.txt
```

## Environment configuration

Copy `.env.example` to `.env`. Do not commit `.env`.

- Leave `PAGEINDEX_API_KEY` and `PAGEINDEX_DOC_ID` empty to use the local retriever.
- Leave `LLM_PROVIDER` empty to use the stub LLM (no network).
- Set `LLM_PROVIDER=openai` (or `openai-compatible`), `LLM_MODEL`, and `LLM_API_KEY` or `OPENAI_API_KEY` to generate real answers.

## Loading the policy corpus

The original manual is already at `data/policy/policy-manual.md`. Ingestion reads that file and `data/policy/manifest.json`. There is no separate load command.

## Running the CLI

Windows (PowerShell), from the repository root:

```text
$env:PYTHONPATH="src"
python -m grounded_answer ask "What are the eligibility requirements?"
python -m grounded_answer ask "What is the boiling point of helium?"
```

macOS / Linux:

```text
PYTHONPATH=src python -m grounded_answer ask "What are the eligibility requirements?"
```

Supported answers print `ANSWER`, `EVIDENCE` (clause IDs), and `GROUNDING` / `SUPPORTED`. Unsupported questions print `GROUNDING` / `INSUFFICIENT` and:

```text
I don't know based on the supplied policy manual.
```

## Running tests

```text
pytest
pytest tests/unit
pytest tests/integration
```

## Running evaluation

```text
$env:PYTHONPATH="src"
python scripts/evaluate.py
```

macOS / Linux:

```text
PYTHONPATH=src python scripts/evaluate.py
```

The runner prints totals measured from executing `evaluation/original/`. With the stub LLM, answer-correctness is typically low because the stub does not write policy facts; evidence, citation, and abstention scores still reflect retrieval and grounding.

## Grounding and abstention

- No evidence, or evidence that does not overlap the question → `INSUFFICIENT`, no LLM call.
- LLM citations must exist in retrieved evidence and in the corpus.
- An answer that cites only invented IDs is rejected.

## Limitations

- Default LLM is a stub; configure `LLM_PROVIDER` for generated policy prose.
- PageIndex is optional and unused unless API credentials are set.
- Stage A uses only the original policy manual. Date-dependent amendments are not applied.

## Troubleshooting

- `No module named grounded_answer`: set `PYTHONPATH=src` from the repository root.
- Unicode errors in the Windows console: the CLI requests UTF-8 stdout; evidence lines use `§`.
- Evaluation answer scores near zero with the stub: expected; use a real `LLM_PROVIDER` to score generated text.
