# Grounded Answer

CLI that answers questions about Calder County's Household Support Program using only the supplied policy corpus and Amendment No. 2026-01.

## 1. What the project does

You type a policy question. The system:

- retrieves supporting clauses from the original policy manual and Amendment No. 2026-01 (kept as separate sources)
- resolves which version of a clause applies for the dates in the question
- establishes that effective evidence before generating an answer
- cites real policy identifiers such as `[§6.4.1]`
- abstains when the corpus does not support the claim, or when a required date is missing

It does not invent policy. It does not use PageIndex. Docker evaluation does not need a host Python install, a host Ollama install, or any API key.

## 2. Architecture

```text
CLI
  ↓
AnswerService
  ↓
Temporal context
  ↓
Retrieval (original policy + amendment)
  ↓
Policy applicability resolver
  ↓
Effective evidence
  ↓
Grounding
  ↓
Answer strategy
  ├── Deterministic fact extraction → plain-English answer
  └── Qwen3 LLM (only if extraction cannot answer) → plain-English answer
  ↓
Citation validation
  ↓
Final answer
```

- Retrieval uses Ollama `qwen3-embedding:4b` when `OLLAMA_BASE_URL` is set (Docker always sets this). A deterministic lexical retriever remains for tests and local runs without Ollama.
- The original `data/policy/policy-manual.md` is never overwritten. Amendment No. 2026-01 lives under `data/amendments/`.
- Date-dependent applicability is resolved before any answer is written.
- Directly extractable facts (amounts, reporting periods, percentages, simple yes/no rules) are formatted without calling the chat model. Broader questions may use `qwen3:4b`.
- The printed answer is short plain English with citations. Model planning or chain-of-thought is not shown.

## 3. Technology stack

What this repository actually uses:

- Python 3.10 application, invoked as a CLI (`python -m grounded_answer`)
- Docker and Docker Compose for evaluation
- Ollama in Compose (`qwen3-embedding:4b` for retrieval, `qwen3:4b` for fallback generation)
- File-backed embedding index under `data/index/` (Compose volume `index_data`)
- Deterministic lexical retrieval when embeddings are not configured
- `pytest` and `scripts/evaluate.py`

**PageIndex is not used.** There is no PageIndex API key and no PageIndex service.

**No external LLM API key is required for Docker evaluation.** Compose talks to the local Ollama service. Optional OpenAI-compatible settings in `.env.example` are for local development only and are not used by the evaluator path.

There is no extra database, MCP layer, or GPU requirement.

## 4. Requirements (Docker evaluation)

- Git
- Docker Desktop (macOS / Windows) or Docker Engine + Docker Compose (Linux)
- Internet access on the first run
- About 8 GB RAM is recommended for a comfortable run (not a hard minimum)

You do **not** need:

- Python on the host
- Ollama on the host
- an API key
- an OpenAI, Gemini, Anthropic, or PageIndex account

The first run downloads the required Docker images and Ollama models and may require several GB of disk space depending on the image/model versions. It can take several minutes.

## 5. Before starting the project

Docker must already be running. This is a Docker environment issue, not an application error.

1. Open **Docker Desktop** on macOS or Windows and wait until it reports that Docker is running. On Linux, ensure Docker Engine is running (`sudo systemctl start docker` if you use systemd).
2. Confirm the daemon is ready:

```text
docker info
```

The output must include both a **Client** section and a **Server** section.

If you see `failed to connect to the docker API` or `Cannot connect to the Docker daemon`, Docker is installed but the daemon is not ready. Wait for Docker Desktop / Engine to finish starting, then run `docker info` again. Do not start Compose until the Server section appears.

On Linux, if Docker requires elevated privileges on your machine, use your usual Docker setup or `sudo` as appropriate.

## 6. Quick start (recommended evaluation path)

No `.env` file is required. Compose sets `OLLAMA_BASE_URL=http://ollama:11434`, `LLM_PROVIDER=ollama`, and `LLM_MODEL=qwen3:4b`.

```text
git clone <repository-url>
cd <cloned-directory>
docker info
docker compose up --build
```

Leave that terminal running. The first start prints a banner:

```text
First startup downloads local AI models and builds the search index. This may take several minutes. Do not interrupt the process.
```

Watch the Compose service names to see which stage is running:

| Service / log | Meaning |
|---|---|
| pulling `ollama/ollama` or `python` images | Docker is downloading images |
| `ollama-init-embed` / `Downloading embedding model` | `qwen3-embedding:4b` is being pulled |
| `ollama-init-embed` / `already present. Skipping download` | embedding model is already in `ollama_data` |
| `ollama-init-llm` / `Downloading generation model` | `qwen3:4b` is being pulled |
| `ollama-init-llm` / `already present. Skipping download` | generation model is already in `ollama_data` |
| `grounded-answer` / `Building retrieval index` | policy/amendment vectors are being created once |
| `grounded-answer` / `Using existing retrieval index` | `index_data` was reused; clauses are not re-embedded |
| `grounded-answer` / `ready for queries` then `System ready.` | you can `exec` ask/verify |

Do not interrupt a first-time download. Init containers skip the pull when the model is already stored.

In a **second** terminal, from the same directory, check readiness (preferred over reading a single log line):

```text
docker compose exec grounded-answer python -m grounded_answer verify
```

Success prints check marks for the policy corpus, amendment, Ollama, embedding model, generation model, and index, then `System ready.` Failure prints `Not ready.` with the failing checks.

Then ask a question (see below).

Later starts, after a clean stop:

```text
docker compose down
docker compose up
```

Volumes `ollama_data` and `index_data` keep models and embeddings. A later `docker compose down` followed by `docker compose up` **does not** re-download models or rebuild the index when those volumes are intact. Use `docker compose down -v` only if you intend to wipe them.

## 7. First run

On `docker compose up --build`:

- Docker builds the application image (and pulls base images if they are not cached).
- The `ollama` service starts. Models are stored in volume `ollama_data`.
- `ollama-init-embed` downloads `qwen3-embedding:4b` **only if it is not already present**.
- `ollama-init-llm` then downloads `qwen3:4b` **only if it is not already present**.
- The app checks the retrieval index in volume `index_data` and embeds the policy and amendment **only if a compatible index is missing**. Later starts log `Using existing retrieval index` and skip re-embedding.

The first run is slower because images, models, and the index may still need to be created. Later runs reuse those volumes when they are still valid.

Common factual questions (amounts, reporting days, percentages, simple yes/no rules) are answered from the effective clause text without calling the chat model, so they are typically quicker. Unsupported questions and missing-date questions also skip the chat model. Broader questions may call `qwen3:4b` and take longer, especially on CPU. There is no fixed response-time guarantee.

## 8. Asking questions

```text
docker compose exec grounded-answer python -m grounded_answer ask "QUESTION"
```

Example:

```text
docker compose exec grounded-answer python -m grounded_answer ask "What are the eligibility requirements?"
```

Supported answers look like:

```text
ANSWER
------------------------
The countable resources limit is $4,000. [§2.4.1]
EVIDENCE
------------------------
§2.4.1
GROUNDING
------------------------
SUPPORTED
```

| Field | Meaning |
|---|---|
| `ANSWER` | Short plain-English sentence(s), with citations such as `[§6.4.1]` |
| `EVIDENCE` | Clause identifiers that support the answer |
| `GROUNDING` | `SUPPORTED` or `INSUFFICIENT` |

If the system abstains, it prints `GROUNDING` / `INSUFFICIENT` and the abstention sentence only (no `ANSWER` / `EVIDENCE` blocks). Internal model reasoning is not part of the output.

## 9. Demonstration questions

Dates in the question text are enough; these do not need extra flags. Behaviour comes from retrieval, amendment applicability, and grounding — not from hardcoded special cases.

```text
docker compose exec grounded-answer python -m grounded_answer ask "What are the eligibility requirements?"
```

Supported. Concise Part 2 eligibility conditions, with citations to retrieved Part 2 clauses.

```text
docker compose exec grounded-answer python -m grounded_answer ask "For a determination made on 15 February 2026, what is the first monthly earnings disregard?"
```

Supported. Original **$120** per month. `[§6.4.1]`

```text
docker compose exec grounded-answer python -m grounded_answer ask "For a determination made on 15 March 2026, what is the first monthly earnings disregard?"
```

Supported. Amended **$175** per month. `[§6.4.1]`

```text
docker compose exec grounded-answer python -m grounded_answer ask "The claimant's circumstances changed on 20 February 2026. How many calendar days does the recipient have to report the change?"
```

Supported. **10 calendar days.** `[§4.3.2]`

```text
docker compose exec grounded-answer python -m grounded_answer ask "The claimant's circumstances changed on 5 March 2026. How many calendar days does the recipient have to report the change?"
```

Supported. **14 calendar days.** `[§4.3.2]`

```text
docker compose exec grounded-answer python -m grounded_answer ask "The claim runs from 20 February 2026 to 10 March 2026. What earnings disregard figures apply?"
```

Supported. Both applicable figures (**$120** and **$175**), with earnings and apportionment evidence (typically `[§6.4.1]` and `[§7.4.3]`).

```text
docker compose exec grounded-answer python -m grounded_answer ask "What is the first monthly earnings disregard?"
```

Insufficient. `The applicable policy cannot be determined without the relevant date.`

```text
docker compose exec grounded-answer python -m grounded_answer ask "What is the boiling point of helium?"
```

Insufficient. `I don't know based on the supplied policy manual.`

## 10. Date options

Dates can be written in the question, or passed as flags (`YYYY-MM-DD`):

```text
--determination-date
--change-of-circumstances-date
--claim-start-date
--claim-end-date
```

Example:

```text
docker compose exec grounded-answer python -m grounded_answer ask "What is the first monthly earnings disregard?" --determination-date 2026-03-15
```

Amendment No. 2026-01 is effective **1 March 2026**. Which version of a rule applies depends on the relevant date, so the system must resolve applicability before answering.

## 11. Surprise challenge (temporal policy)

- Amendment No. 2026-01 is ingested from `data/amendments/` and is never written into the original manual.
- Different amendment paragraphs use different dates: paragraphs 1, 3, and 4 follow the **determination** date; paragraph 2 follows the **change-of-circumstances** date. Those dates are not interchangeable.
- A claim period that crosses 1 March 2026 can require more than one applicable figure; the award is apportioned under §7.4.3.
- `PolicyApplicabilityResolver` selects the effective clause text **before** generation. Missing required dates produce an abstention rather than an assumed date.

## 12. No API key / self-contained evaluation

For Docker evaluation, no OpenAI, Gemini, Anthropic, PageIndex, or other external API key is required. You do not paste a key into `.env`.

Compose uses the Ollama service on the Docker network. `.env.example` documents optional local/OpenAI-compatible settings; they are **not** required for evaluation.

## 13. PageIndex

PageIndex is not used in the submitted implementation.

- no PageIndex API key
- no PageIndex service
- retrieval is the `Retriever` port: Ollama embeddings in Docker, lexical fallback when `OLLAMA_BASE_URL` is unset

## 14. Evaluation commands

Inside the running stack:

```text
docker compose exec grounded-answer pytest
docker compose exec grounded-answer python /app/scripts/evaluate.py
```

`scripts/evaluate.py` accepts `--dataset original`, `--dataset surprise`, or `--dataset all` (default).

Host copies of those commands are in the optional local-development section. They are not required for hackathon evaluation.

## 15. Troubleshooting

**Docker daemon not running** (`failed to connect to the docker API` / `Cannot connect to the Docker daemon`)

Start Docker Desktop or Docker Engine. Confirm `docker info` shows a Server section. This is a Docker environment issue, not an application error.

**Ollama unavailable** (`ERROR: Ollama is unavailable at http://ollama:11434`)

Wait for health, or inspect logs:

```text
docker compose ps
docker compose logs ollama
docker compose logs grounded-answer
```

**Port 11434 already in use**

Compose publishes Ollama on host port 11434. Stop a host Ollama process, or change the published port in `docker-compose.yml`.

**Stale or incompatible index**

The app does not reuse vectors built with a different embedding model.

```text
docker compose down
docker compose up --build
```

To also discard models and the index:

```text
docker compose down -v
docker compose up --build
```

**First run / model init is slow**

Leave Compose running. The first pull of `qwen3-embedding:4b` and `qwen3:4b` happens only when those models are missing from `ollama_data`. On a later `docker compose up`, init logs should say `already present. Skipping download`, and the app should say `Using existing retrieval index`. CPU-only machines are slower on first embed and on chat fallback. There is no promised response time.

**Service looks stuck or failed**

```text
docker compose ps
docker compose logs
docker compose exec grounded-answer python -m grounded_answer verify
```

## 16. Platform notes

**macOS:** Docker Desktop is recommended. Open it before Compose. Apple Silicon is supported through Docker Desktop.

**Windows:** Docker Desktop is recommended and must be running first. PowerShell and Command Prompt can both run the Docker commands above. The evaluator path does not use Unix-only commands such as `cp`.

**Linux:** Docker Engine + Compose. The daemon must be running. Use a standard shell; add `sudo` only if that is how Docker is configured on the machine.

The same Compose commands are used on all three platforms.

## 17. Optional: local development without Docker

This section is **not** required for hackathon evaluation.

Python 3.10+, from the repository root. Copy `.env.example` to `.env` if you want local Ollama or an optional OpenAI-compatible provider.

Lexical retrieval (no Ollama). On Windows:

```text
pip install -r requirements.txt
set PYTHONPATH=src
python -m grounded_answer ask "What are the eligibility requirements?"
pytest
python scripts/evaluate.py
```

PowerShell: `$env:PYTHONPATH="src"` instead of `set PYTHONPATH=src`. Unix: `export PYTHONPATH=src`.

Optional local Ollama: install Ollama, pull `qwen3-embedding:4b` and `qwen3:4b`, and set `OLLAMA_BASE_URL=http://localhost:11434`, `LLM_PROVIDER=ollama`, and `LLM_MODEL=qwen3:4b` in `.env`. Leave `LLM_PROVIDER` empty or `stub` for the stub generator.

## 18. Evaluator checklist

- [ ] Docker Desktop / Docker Engine is running
- [ ] `docker info` shows a Server section
- [ ] Repository cloned
- [ ] `docker compose up --build` completed successfully
- [ ] `docker compose exec grounded-answer python -m grounded_answer verify` succeeds
- [ ] A demonstration question returns ANSWER / EVIDENCE / GROUNDING (or INSUFFICIENT for the abstention cases)
- [ ] No API key was required
