# Architecture Decisions

## Decision 001 — CLI-first

The hackathon allows a CLI/notebook submission and does not require a frontend. The first working version is a command-line application.

## Decision 002 — Retrieval abstraction

Application code retrieves evidence through a `Retriever` interface (`retrieve(query) -> evidence`). This keeps the rest of the system independent of any specific retrieval library or vendor.

## Decision 003 — Local retrieval, no PageIndex

PageIndex was never installed or used. Retrieval is Ollama `qwen3-embedding:4b` when `OLLAMA_BASE_URL` is set, otherwise the deterministic Part/Section/Clause lexical retriever used by tests. There is no PageIndex SDK, API key, or adapter.

## Decision 004 — Canonical evidence assembly

Retrievers return raw hits. `EvidenceAssembler` maps those hits onto ingested policy clauses before any answer is generated, so evidence content comes from the corpus rather than from adapter snippets.

## Decision 005 — LLM provider abstraction

Answer generation calls `LLMProvider.generate(prompt, context)`. The provider and model come from `LLM_PROVIDER` and `LLM_MODEL`, so the application is not hard-coded to one vendor.

## Decision 006 — Application service coordinates generation

`QueryService` and `AnswerService` orchestrate retrieval and LLM generation. They depend on ports (`Retriever`, `LLMProvider`), not on a specific retrieval or model vendor.

## Decision 007 — Grounding before generation

`GroundingValidator` decides SUPPORTED or INSUFFICIENT from retrieved evidence before the LLM is called. Off-topic hits are treated as insufficient. There is no PARTIALLY_SUPPORTED status.

## Decision 008 — Citation validation

Citations on an answer must resolve to retrieved evidence and to a real policy clause. Invented identifiers such as §99.9.9 are dropped; an answer that cites only invented clauses is rejected.

## Decision 009 — CLI as the first interface

The first user interface is `python -m grounded_answer ask`. It calls `AnswerService` and does not import a specific retrieval or LLM vendor.

## Decision 010 — Stage A V1 checkpoint

Stage A was frozen at this point: original corpus, retrieval,
grounding, citations, CLI, tests, and evaluation.

Amendments, temporal resolution, and MCP were intentionally deferred
until after the `v1.0-grounded-answer` checkpoint. Stage B subsequently
extended this foundation for Amendment No. 2026-01.

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

## Decision 016 — Ollama Qwen3 Embedding 4B

Local retrieval uses the official Ollama model `qwen3-embedding:4b` through an `EmbeddingProvider` port. The application does not call Ollama from AnswerService or the temporal resolver. The model is configurable (`OLLAMA_EMBEDDING_MODEL`). This avoids an external embedding API and keeps evaluation self-contained. The lexical fallback remains for tests when `OLLAMA_BASE_URL` is unset. Query text is embedded with a policy-retrieval instruction; document chunks are not. No retrieval-quality benchmark was measured for this change.

## Decision 017 — Docker Compose evaluation

The evaluator path is `docker compose up --build`. The Python app and Ollama run as separate services. The app uses `OLLAMA_BASE_URL=http://ollama:11434`. Models live in the `ollama_data` volume so later starts do not re-download. Generated embeddings live in the `index_data` volume so later starts do not re-embed unless sources or the model change. An `ollama-init-embed` service pulls `qwen3-embedding:4b` and `ollama-init-llm` pulls `qwen3:4b` if they are missing. The app generates answers with `LLM_PROVIDER=ollama` and `LLM_MODEL=qwen3:4b`. This does not replace the existing local Python workflow.

## Decision 018 — Ollama Qwen3 4B generation

Supported answers are generated by official Ollama `qwen3:4b` through `LLMProvider` only when a structured fact cannot be extracted from effective evidence. Retrieval still uses `qwen3-embedding:4b`. The two models are not interchangeable. AnswerService still grounds and resolves applicability before generation; insufficient questions never call the LLM. The model name is `LLM_MODEL` (default `qwen3:4b`). Stub remains the provider when `LLM_PROVIDER` is empty or `stub`, which is what Stage A/B tests use.

## Decision 019 — Deterministic short answers after grounding

Where effective evidence contains a typed fact (money, duration, percentage, or a simple modal condition), the application formats a one-sentence answer and appends only the citations that support that answer. Broader eligibility questions are synthesised from the effective Part 2 clauses without exposing model reasoning. Qwen3 thinking is disabled (`think: false`, `/no_think`) and `LLM_NUM_PREDICT` defaults to 60 for the fallback path. Any remaining model preamble is stripped before the answer is shown. Surprise-challenge questions are not hardcoded; extraction reads the already-resolved clause text. No external API key is required.
