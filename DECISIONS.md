# Architecture Decisions

## Decision 001 — CLI-first

The hackathon allows a CLI/notebook submission and does not require a frontend. The first working version is a command-line application.

## Decision 002 — Retrieval abstraction

Application code retrieves evidence through a `Retriever` interface (`retrieve(query) -> evidence`). This keeps the rest of the system independent of any specific retrieval library or vendor.
