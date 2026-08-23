# Grounded Answer

A CLI-first system that answers questions about Calder County's Household Support Program policy.

Answers must be grounded in the supplied policy corpus, include citations to the supporting clauses, and abstain when the corpus does not support an answer.

## Current status

The original policy corpus is in `data/policy/policy-manual.md`. Core domain models live under `src/grounded_answer/domain/`. Policy ingestion parses the manual into Parts, Sections, and Clauses (for example `§2.1.2`). Retrieval is accessed through a `Retriever` interface. Raw hits are converted to canonical Evidence by the evidence assembly layer. Text generation goes through an `LLMProvider` interface; provider and model are selected by configuration. Question answering is not wired yet.

## Prerequisites

- Python 3.10 or later

## Installation

```text
pip install -r requirements.txt
```

## Running tests

```text
pytest
```

## Environment configuration

Copy `.env.example` to `.env`. Leave `PAGEINDEX_API_KEY` empty to use the local retrieval fallback. Leave `LLM_PROVIDER` empty to use the stub LLM. Do not commit `.env`.
