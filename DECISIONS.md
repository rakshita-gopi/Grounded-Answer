# Architecture Decisions

## Decision 001 — CLI-first

The hackathon allows a CLI/notebook submission and does not require a frontend. The first working version is a command-line application.

## Decision 002 — Retrieval abstraction

Application code retrieves evidence through a `Retriever` interface (`retrieve(query) -> evidence`). This keeps the rest of the system independent of any specific retrieval library or vendor.

## Decision 003 — PageIndex

PageIndex is the intended structure-aware retrieval backend. Its SDK and credentials stay inside `PageIndexRetriever`. When PageIndex cannot run (missing package or API key), a deterministic Part/Section/Clause retriever is used so tests and local execution stay reproducible.

## Decision 004 — Canonical evidence assembly

Retrievers return raw hits. `EvidenceAssembler` maps those hits onto ingested policy clauses before any answer is generated, so evidence content comes from the corpus rather than from adapter snippets.

## Decision 005 — LLM provider abstraction

Answer generation calls `LLMProvider.generate(prompt, context)`. The provider and model come from `LLM_PROVIDER` and `LLM_MODEL`, so the application is not hard-coded to one vendor.

## Decision 006 — Application service coordinates generation

`QueryService` and `AnswerService` orchestrate retrieval and LLM generation. They depend on ports (`Retriever`, `LLMProvider`), not on PageIndex or a specific model vendor.

## Decision 007 — Grounding before generation

`GroundingValidator` decides SUPPORTED or INSUFFICIENT from retrieved evidence before the LLM is called. Off-topic hits are treated as insufficient. There is no PARTIALLY_SUPPORTED status.
