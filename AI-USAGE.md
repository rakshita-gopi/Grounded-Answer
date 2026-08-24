# AI Usage

## AI tools used

- Cursor (Grok) for repository scaffolding and later implementation assistance.

## Purpose

Incremental implementation of Grounded Answer according to the project plan.

## Phase 1

Created the repository skeleton: README, DECISIONS, AI-USAGE, gitignore, env example, requirements.txt, docker-compose.yml, and empty package/layout directories.

## Phase 2

Placed the original policy manual at `data/policy/policy-manual.md` without editing its content, and created `data/policy/manifest.json` from the fields specified for the original corpus.

## Phase 3

Defined Stage A domain objects: Policy, PolicyClause, Question, Evidence, Citation, Answer, and GroundingStatus. Amendment, versioning, and temporal types were not added.

## Phase 4

Added a structure-aware policy parser (Part / Section / Clause), an ingestion service that maps parsed clauses onto domain objects, and unit tests for document loading, clause detection, clause IDs, and text preservation.

## Phase 5

Added the retrieval port (`Retriever`), retrieval query model, and `RetrievalService`. No PageIndex adapter or local fallback was added.

## Phase 6

Added a deterministic structure-aware local retriever (Part/Section/Clause) behind the existing `Retriever` port, used for tests and runs without Ollama.

## Phase 7

Added the evidence assembly pipeline: raw retrieval hits are converted to canonical `Evidence` objects from ingested clauses, then checked by `EvidenceValidator`. This layer does not generate answers.

## Phase 8

Added the LLM port (`generate(prompt, context)`), grounding prompt construction (question + evidence + instructions), and configuration-selected providers (`stub`, `openai`, `openai-compatible`). No grounded-answer service was added.

## Phase 9

Added `QueryService` and `AnswerService`. The application flow is question → retrieval → evidence → LLM → answer. Empty evidence abstains without calling the LLM. Dedicated grounding and citation validators were not added.

## Phase 10

Added `GroundingValidator`, which classifies retrieved evidence as SUPPORTED or INSUFFICIENT before generation. Off-topic retrieval now abstains. Citation validation was not added.

## Phase 11

Added `CitationValidator`. Generated answers may only cite clause IDs present in retrieved evidence and, when supplied, the policy corpus. Invented IDs such as §99.9.9 are dropped or cause abstention.

## Phase 12

Added the CLI (`python -m grounded_answer ask`) and a bootstrap that wires retrieval, grounding, citation validation, and the LLM behind application services.

## Phase 13

Added Stage A integration tests for the question → retrieval → evidence → answer path, including abstention and rejection of invented clause IDs. Existing unit tests for the parser, evidence assembly, grounding, and citations remain in `tests/unit`.

## Phase 14

Added the original evaluation dataset (`evaluation/original/questions.json` and `expected.json`) with direct, multi-condition, cross-reference, exact-clause, and unsupported questions. Expected facts were taken from the supplied policy manual. The evaluation runner was not added.

## Phase 15

Added `scripts/evaluate.py`, which runs the original dataset and prints measured answer, evidence, citation, and abstention scores.

## Phase 16

Added the Stage A V1 checkpoint README (actual runnable setup only). Amendments, temporal reasoning, and MCP were not added.

## Stage B

Extended Stage A for Amendment No. 2026-01: separate amendment ingestion, temporal context extraction, applicability resolution, effective evidence, CLI date flags, surprise evaluation, and regression tests. The organizer-provided amendment and `docs/READ ME FIRST.md` were not generated or edited. Policy applicability is resolved in application/domain code; the LLM is not asked to choose between conflicting versions. Human verification: pytest (86 passed) and both evaluation datasets were run on this machine. The legal meaning of the amendment was not independently validated beyond implementing the text as supplied.

## Ollama embeddings and Docker

Added an `EmbeddingProvider` port, an Ollama HTTP adapter for `qwen3-embedding:4b`, a file-backed index with model/dimension metadata, Docker Compose (app + Ollama + model init), and README evaluator instructions. Existing Stage A/B tests continue to use lexical retrieval when `OLLAMA_BASE_URL` is unset. A later change added `OllamaLLMProvider` for `qwen3:4b` chat generation, still behind `LLMProvider`, with the stub retained when `LLM_PROVIDER` is unset. Structured answers are now extracted from effective evidence after grounding so evaluators are not blocked on CPU chat generation for typed policy facts.

## Human verification

Repository layout and file contents should be reviewed before each commit.

## Known limitations

AI-generated scaffolding must be reviewed. This log records only work that was actually performed.

## PageIndex

The unused PageIndex adapter, optional `pageindex` dependency, and `PAGEINDEX_*` configuration were removed. The project runs on Ollama embeddings or the lexical retriever.
