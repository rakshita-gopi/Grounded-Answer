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

## Human verification

Repository layout and file contents should be reviewed before each commit.

## Known limitations

AI-generated scaffolding must be reviewed. This log records only work that was actually performed.
