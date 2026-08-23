# AI Usage

## AI tools used

- Cursor (Grok) for repository scaffolding and later implementation assistance.

## Purpose

Incremental implementation of Grounded Answer according to the project plan.

## Phase 1

AI assisted creation of the repository skeleton: README, DECISIONS, AI-USAGE, gitignore, env example, requirements.txt, docker-compose.yml, and empty package/layout directories.

## Phase 2

AI assisted placing the original policy manual at `data/policy/policy-manual.md` without editing its content, and creating `data/policy/manifest.json` from the fields specified for the original corpus.

## Phase 3

AI assisted definition of Stage A domain objects: Policy, PolicyClause, Question, Evidence, Citation, Answer, and GroundingStatus. Amendment, versioning, and temporal types were not added.

## Phase 4

AI assisted a structure-aware policy parser (Part / Section / Clause), an ingestion service that maps parsed clauses onto domain objects, and unit tests for document loading, clause detection, clause IDs, and text preservation.

## Phase 5

AI assisted the retrieval port (`Retriever`), retrieval query model, and `RetrievalService`. No PageIndex adapter or local fallback was added.

## Phase 6

AI assisted a PageIndex adapter behind the existing `Retriever` port, a factory that selects PageIndex only when credentials are present, and a deterministic structure-aware local fallback used for tests and runs without external services.

## Phase 7

AI assisted the evidence assembly pipeline: raw retrieval hits are converted to canonical `Evidence` objects from ingested clauses, then checked by `EvidenceValidator`. This layer does not generate answers.

## Phase 8

AI assisted the LLM port (`generate(prompt, context)`), grounding prompt construction (question + evidence + instructions), and configuration-selected providers (`stub`, `openai`, `openai-compatible`). No grounded-answer service was added.

## Phase 9

AI assisted `QueryService` and `AnswerService`. The application flow is question → retrieval → evidence → LLM → answer. Empty evidence abstains without calling the LLM. Dedicated grounding and citation validators were not added.

## Phase 10

AI assisted `GroundingValidator`, which classifies retrieved evidence as SUPPORTED or INSUFFICIENT before generation. Off-topic retrieval now abstains. Citation validation was not added.

## Phase 11

AI assisted `CitationValidator`. Generated answers may only cite clause IDs present in retrieved evidence and, when supplied, the policy corpus. Invented IDs such as §99.9.9 are dropped or cause abstention.

## Human verification

Repository layout and file contents should be reviewed before each commit.

## Known limitations

AI-generated scaffolding must be reviewed. This log records only work that was actually performed.
