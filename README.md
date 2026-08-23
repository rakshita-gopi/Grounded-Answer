# Grounded Answer

CLI that answers questions about Calder County's Household Support Program using only the supplied policy corpus and Amendment No. 2026-01. Answers are grounded in retrieved clauses, cite real identifiers, and abstain when the corpus does not support a claim or a required date is missing.

## 1. What the project does

You type a policy question. The system retrieves supporting clauses from the original manual and the amendment, resolves which version applies for the date you asked about, then generates a grounded answer with citations. It does not invent policy.

## 2. Architecture

```text
CLI
  → AnswerService
      → temporal extraction
      → retrieval (original policy + amendment)
      → policy applicability resolver
      → effective evidence
      → grounding
      → LLM
      → citation validation
  → answer
```

The original manual stays unchanged. The amendment is a separate source. Retrieval is behind a `Retriever` port: Ollama `qwen3-embedding:4b` when `OLLAMA_BASE_URL` is set, otherwise a deterministic lexical fallback used by tests.

## 3. Requirements (Docker evaluation)

- Docker
- Docker Compose
- Internet on the first run (to pull images and the embedding model, about 2.5 GB)

You do not need a host Python or Ollama install for this path.

## 4. Quick start

```text
cd Grounded-Answer
cp .env.example .env
docker compose up --build
```

Wait until the logs show `System ready` and `Grounded Answer is ready`.

## 5. First run

The first start downloads `qwen3-embedding:4b` and `qwen3:4b` from the official Ollama library (about 2.5 GB each) into a Docker volume. Indexing the policy and amendment then runs once against the embedding model. Typed policy facts (amounts, reporting days) are answered without waiting on the chat model. The chat model is used only when a fact cannot be extracted from the effective evidence.

## 6. Subsequent runs

```text
docker compose down
docker compose up
```

The Ollama volume `ollama_data` keeps the model. The `index_data` volume keeps generated embeddings. Neither is re-downloaded or rebuilt unless you run `docker compose down -v`.

## 7. Asking questions

In a second terminal, from the same project directory:

```text
docker compose exec grounded-answer python -m grounded_answer ask "What are the eligibility requirements?"
```

The reply is printed in that terminal as `ANSWER`, `EVIDENCE` (clause IDs), and `GROUNDING` (`SUPPORTED` or `INSUFFICIENT`).

Optional dates:

```text
docker compose exec grounded-answer python -m grounded_answer ask "What is the first monthly earnings disregard?" --determination-date 2026-03-15
```

Environment check:

```text
docker compose exec grounded-answer python -m grounded_answer verify
```

## 8. Demonstration questions

```text
docker compose exec grounded-answer python -m grounded_answer ask "What are the eligibility requirements?"

docker compose exec grounded-answer python -m grounded_answer ask "For a determination made on 15 February 2026, what is the first monthly earnings disregard?"

docker compose exec grounded-answer python -m grounded_answer ask "For a determination made on 15 March 2026, what is the first monthly earnings disregard?"

docker compose exec grounded-answer python -m grounded_answer ask "The claimant's circumstances changed on 20 February 2026. How many calendar days does the recipient have to report the change?"

docker compose exec grounded-answer python -m grounded_answer ask "The claimant's circumstances changed on 5 March 2026. How many calendar days does the recipient have to report the change?"

docker compose exec grounded-answer python -m grounded_answer ask "The claim runs from 20 February 2026 to 10 March 2026. What earnings disregard figures apply?"

docker compose exec grounded-answer python -m grounded_answer ask "What is the first monthly earnings disregard?"

docker compose exec grounded-answer python -m grounded_answer ask "What is the boiling point of helium?"
```

Expected grounding behaviour:

- 15 February 2026 determination → original **$120** (`§6.4.1`)
- 15 March 2026 determination → amended **$175**
- Change on 20 February 2026 → **10 calendar days**
- Change on 5 March 2026 → **14 calendar days**
- Spanning claim → both figures and `§7.4.3`
- Earnings disregard with no date → `The applicable policy cannot be determined without the relevant date.`
- Helium → `I don't know based on the supplied policy manual.`

Docker Compose sets `LLM_PROVIDER=ollama` and `LLM_MODEL=qwen3:4b`. After grounding, if the effective evidence contains a directly extractable fact (amount, duration, percentage, yes/no), the application writes a short plain-English answer **without calling the chat model**, so those questions are fast on CPU. Broader questions such as eligibility requirements are synthesised from the effective clauses. The printed answer never includes model reasoning or planning language. Citations are appended by the application, and only clauses that support the answer are cited, for example `[§6.4.1]`. Qwen3 is used only when a fact cannot be extracted. Missing-date and unsupported questions still abstain **before** any generation. Embeddings stay on `qwen3-embedding:4b`.

## 9. Evaluation

Inside the running stack:

```text
docker compose exec grounded-answer pytest
docker compose exec grounded-answer python /app/scripts/evaluate.py
```

Or on the host (local Python path below):

```text
pytest
python scripts/evaluate.py --dataset surprise
```

## 10. Troubleshooting

- **Docker is not running:** start Docker Desktop, then `docker compose up --build`.
- **Model init is slow:** first run pulls `qwen3-embedding:4b` and `qwen3:4b` (~2.5 GB each). Watch `ollama-init-embed` and `ollama-init-llm`.
- **Port 11434 in use:** stop a host Ollama process, or change the published port in `docker-compose.yml`.
- **Stale/incompatible index:** the app refuses to reuse vectors built with a different embedding model. Rebuild with `docker compose down` then `docker compose up --build`, or reset volumes with `docker compose down -v`.
- **Ollama unavailable:** `ERROR: Ollama is unavailable at http://ollama:11434`. Wait for health, or `docker compose logs ollama`.
- **Reset everything including the model:** `docker compose down -v` then `docker compose up --build`.
- **`No module named grounded_answer` on the host:** set `PYTHONPATH=src` (see local development).

## 11. Local development (no Docker)

Python 3.10+, from the repository root:

```text
pip install -r requirements.txt
cp .env.example .env
```

Lexical retrieval (no Ollama), Windows:

```text
$env:PYTHONPATH="src"
python -m grounded_answer ask "What are the eligibility requirements?"
pytest
python scripts/evaluate.py
```

Local Ollama: install Ollama, pull `qwen3-embedding:4b` and `qwen3:4b`, keep `OLLAMA_BASE_URL=http://localhost:11434`, `LLM_PROVIDER=ollama`, and `LLM_MODEL=qwen3:4b` in `.env`.

Set `LLM_PROVIDER=stub` (or leave it empty outside Docker) for the stub generator.
