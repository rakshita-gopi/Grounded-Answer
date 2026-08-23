# Grounded Answer

A CLI-first system that answers questions about Calder County's Household Support Program policy.

Answers must be grounded in the supplied policy corpus, include citations to the supporting clauses, and abstain when the corpus does not support an answer.

## Current status

The original policy corpus is in `data/policy/policy-manual.md`. Core domain models live under `src/grounded_answer/domain/`. Policy ingestion parses the manual into Parts, Sections, and Clauses (for example `§2.1.2`). Retrieval is accessed through a `Retriever` interface. Raw hits are converted to canonical Evidence by the evidence assembly layer. `AnswerService` coordinates query, retrieval, grounding, and LLM generation. A deterministic grounding validator abstains when evidence is missing or off-topic. Citation validation keeps only clause IDs that exist in retrieved evidence and the policy corpus. Ask questions with the CLI. Stage A unit tests cover parsing, retrieval, evidence, grounding, and citations. Integration tests cover question → retrieval → evidence → answer.

## Prerequisites

- Python 3.10 or later

## Installation

```text
pip install -r requirements.txt
```

## Running tests

```text
pytest
pytest tests/unit
pytest tests/integration
```

## Running the CLI

From the repository root:

```text
$env:PYTHONPATH="src"
python -m grounded_answer ask "What are the eligibility requirements?"
```

On macOS/Linux:

```text
PYTHONPATH=src python -m grounded_answer ask "What are the eligibility requirements?"
```

## Environment configuration

Copy `.env.example` to `.env`. Leave `PAGEINDEX_API_KEY` empty to use the local retrieval fallback. Leave `LLM_PROVIDER` empty to use the stub LLM. Do not commit `.env`.
