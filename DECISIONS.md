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

## Decision 008 — Citation validation

Citations on an answer must resolve to retrieved evidence and to a real policy clause. Invented identifiers such as §99.9.9 are dropped; an answer that cites only invented clauses is rejected.

## Decision 009 — CLI as the first interface

The first user interface is `python -m grounded_answer ask`. It calls `AnswerService` and does not import PageIndex or a specific LLM vendor.

## Decision 010 — Stage A V1 checkpoint

Stage A is frozen here: original corpus, retrieval, grounding, citations, CLI, tests, and evaluation. Amendments, temporal resolution, and MCP wait until after tag `v1.0-grounded-answer`.

# Stage B — Surprise Challenge Decisions

## Decision 011 — Amendment as a separate source

The original policy document is immutable. Amendment No. 2026-01 is ingested from `data/amendments/` and never written back into `policy-manual.md`. Query-time resolution chooses which text applies.

## Decision 012 — Temporal policy resolution before generation

Policy applicability is resolved in `PolicyApplicabilityResolver` before the LLM is called. Determination-date rules (amendment paragraphs 1, 3, and 4) are distinct from change-of-circumstances rules (paragraph 2). A claim that spans 1 March 2026 is segmented and apportioned under §7.4.3.

## Decision 013 — Effective evidence

The LLM receives only the clause text that has been determined to apply. It is not given both $120 and $175 as alternatives and asked to pick.

## Decision 014 — Historical preservation

Original values such as `$120 per month` and `20 per cent` remain in the source manual so pre-effective-date questions can still be answered.

## Decision 015 — Deterministic applicability

Date-dependent resolution is application/domain logic. Missing required dates produce a clarification abstention rather than an assumed date. No graph database, extra LLM router, or MCP layer was added; Stage A ports (Retriever, LLMProvider) were reused.
