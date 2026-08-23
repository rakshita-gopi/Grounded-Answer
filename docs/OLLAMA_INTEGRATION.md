# Ollama embedding integration

## Retrieval architecture

Stage A/B retrieval is behind a `Retriever` port:

`retrieve(query) -> Sequence[RetrievalHit]`

Callers (`RetrievalService`, `QueryService`, `AnswerService`) never import a vendor SDK.

Current implementations:

| Component | Role |
|---|---|
| `EmbeddingRetriever` | Default evaluation path: cosine similarity over Ollama `qwen3-embedding:4b` vectors |
| `DeterministicStructureRetriever` | Lexical fallback when `OLLAMA_BASE_URL` is unset (pytest `environ={}`) |
| `CompositeRetriever` | Merges original-policy hits with amendment hits |

When `OLLAMA_BASE_URL` is set (Docker always sets it; local `.env` may set it):

```text
Retriever
  → EmbeddingRetriever
      → EmbeddingProvider
          → OllamaEmbeddingProvider
              → POST {OLLAMA_BASE_URL}/api/embed
```

Document vectors are stored under `data/index/` with metadata (provider, model, dimension, index version, source fingerprints). Queries use a task instruction; documents do not. Docker Compose persists that directory in the `index_data` volume.

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
- Stage A/B tests that inject `environ={}` (lexical path)
