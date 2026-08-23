# Ollama embedding integration

## Current retrieval architecture

Stage A/B retrieval is behind a `Retriever` port:

`retrieve(query) -> Sequence[RetrievalHit]`

Callers (`RetrievalService`, `QueryService`, `AnswerService`) never import PageIndex or a vendor SDK.

Current implementations:

| Component | Role |
|---|---|
| `PageIndexRetriever` | Optional structure-aware backend when `PAGEINDEX_API_KEY` and `PAGEINDEX_DOC_ID` are set |
| `DeterministicStructureRetriever` | Default local backend: Part/Section/Clause term overlap + clause-id boost |
| `CompositeRetriever` | Merges original-policy hits with amendment hits |

There is **no** embedding interface, **no** vector index, and **no** local embedding model. Stage A tests and the default CLI path use **lexical** retrieval. PageIndex is unused unless credentials are present.

Embeddings are **not** generated anywhere in the current code.

## Proposed Ollama path

Add a small `EmbeddingProvider` port and an `OllamaEmbeddingProvider` that calls the Ollama HTTP embed API.

When `OLLAMA_BASE_URL` is set (Docker always sets it; local `.env` may set it):

```text
Retriever
  → EmbeddingRetriever
      → EmbeddingProvider
          → OllamaEmbeddingProvider
              → POST {OLLAMA_BASE_URL}/api/embed
```

Document vectors are stored under `data/index/` with metadata (provider, model, dimension, index version, source fingerprints). Queries use a task instruction; documents do not. Docker Compose persists that directory in the `index_data` volume.

If `OLLAMA_BASE_URL` is unset (pytest `environ={}`), the factory keeps `DeterministicStructureRetriever`. PageIndex selection is unchanged.

If stored index metadata does not match the configured provider, model, dimension, or source fingerprints, the index is rebuilt rather than reused. A model change never silently reuses old vectors.

## Why this sits behind Retriever

Application and temporal-resolution code already depend on `Retriever`, not on how hits are scored. Swapping lexical overlap for cosine similarity over Ollama vectors does not change evidence assembly, grounding, applicability, or citations.

The model name is configured once (`OLLAMA_EMBEDDING_MODEL`, default `qwen3-embedding:4b`). It is not hard-coded in AnswerService or the resolver.

## What remains unchanged

- Original `data/policy/policy-manual.md`
- Separate `data/amendments/Amendment No. 2026-01.md`
- Temporal applicability resolver
- Grounding and citation validators
- CLI `ask` output format
- PageIndex behind `Retriever`
- Stage A/B tests that inject `environ={}` (lexical path)
