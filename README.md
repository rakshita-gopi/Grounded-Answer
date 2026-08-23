# Grounded Answer

A CLI-first system that answers questions about Calder County's Household Support Program policy.

Answers must be grounded in the supplied policy corpus, include citations to the supporting clauses, and abstain when the corpus does not support an answer.

## Current status

**Stage B.** The original policy corpus is unchanged. Amendment No. 2026-01 is ingested as a separate source. Policy applicability is resolved from dates in the question (or CLI flags) before the LLM is called. Stage A grounding, citations, CLI, and evaluation remain in place.

## Problem

Policy questions need answers that can be traced to numbered clauses (for example `§2.1.2`). The system must not invent policy, and it must say so when the supplied manual does not support an answer.

## Solution

1. Ingest `data/policy/policy-manual.md` into Parts, Sections, and Clauses.
2. Ingest `data/amendments/Amendment No. 2026-01.md` as a separate amendment (the original manual is not edited).
3. Retrieve structure-aware hits from the original policy and, where relevant, the amendment.
4. Extract temporal context and resolve which version of each clause applies.
5. Assemble canonical **effective** `Evidence` (one applicable text per clause, or segmented text for a claim that spans 1 March 2026).
6. Decide SUPPORTED or INSUFFICIENT before calling the LLM. If an amended clause is required and the relevant date is missing, ask for that date instead of guessing.
7. Generate from question + effective evidence + grounding instructions.
8. Keep only citations that exist in retrieved evidence and in the corpus (original clauses, inserted `§10.5.3A`, and amendment paragraphs).

## Key features

- Clause identifiers such as `§2.1.2`, `§2.4.1`, `§6.6.1`, and inserted `§10.5.3A`
- Retrieval behind a `Retriever` port (no PageIndex types in application code)
- Date-aware applicability for Amendment No. 2026-01 without editing the original manual
- Grounding abstention for missing evidence, off-topic hits, or missing required dates
- Rejection of invented citations such as `§99.9.9`
- CLI: `python -m grounded_answer ask "..."`
- Evaluation: `python scripts/evaluate.py`

## Architecture

```text
CLI
  → AnswerService
      → temporal context (from question text and optional CLI dates)
      → QueryService → Retriever (original policy + amendment) → EvidenceAssembler
      → PolicyApplicabilityResolver  (effective evidence)
      → GroundingValidator  (SUPPORTED | INSUFFICIENT)
      → LLMProvider
      → CitationValidator
  → printed answer / evidence / grounding
```

## Technology stack

- Python 3.10+
- pytest
- Optional: `pageindex` and an LLM provider (`openai` / `openai-compatible`)

## Repository structure

```text
data/policy/            original corpus and manifest (immutable)
data/amendments/        Amendment No. 2026-01.md
docs/READ ME FIRST.md   surprise-challenge brief
src/grounded_answer/    domain, ingestion, amendments, retrieval, temporal, evidence, grounding, citations, LLM, application, CLI
tests/unit/             unit tests
tests/integration/      Stage A and Stage B pipelines
evaluation/original/    Stage A questions and expected facts
evaluation/surprise/    amendment and temporal questions
scripts/evaluate.py     evaluation runner
```

## Prerequisites

- Python 3.10 or later

## Installation

From the repository root:

```text
pip install -r requirements.txt
```

## Environment configuration

Copy `.env.example` to `.env`. Do not commit `.env`.

- Leave `PAGEINDEX_API_KEY` and `PAGEINDEX_DOC_ID` empty to use the local retriever.
- Leave `LLM_PROVIDER` empty to use the stub LLM (no network).
- Set `LLM_PROVIDER=openai` (or `openai-compatible`), `LLM_MODEL`, and `LLM_API_KEY` or `OPENAI_API_KEY` to generate real answers.

## Loading the policy corpus

The original manual is already at `data/policy/policy-manual.md`. Ingestion reads that file and `data/policy/manifest.json`. There is no separate load command.

## Running the CLI

Windows (PowerShell), from the repository root:

```text
$env:PYTHONPATH="src"
python -m grounded_answer ask "What are the eligibility requirements?"
python -m grounded_answer ask "What is the boiling point of helium?"
python -m grounded_answer ask "For a determination made on 15 March 2026, what is the first monthly earnings disregard?"
python -m grounded_answer ask "What is the first monthly earnings disregard?" --determination-date 2026-03-15
```

macOS / Linux:

```text
PYTHONPATH=src python -m grounded_answer ask "What are the eligibility requirements?"
```

Supported answers print `ANSWER`, `EVIDENCE` (clause IDs), and `GROUNDING` / `SUPPORTED`. Unsupported questions print `GROUNDING` / `INSUFFICIENT` and:

```text
I don't know based on the supplied policy manual.
```

If an amended rule is involved and the relevant date is missing:

```text
The applicable policy cannot be determined without the relevant date.
```

Optional CLI dates: `--determination-date`, `--change-of-circumstances-date`, `--claim-start-date`, `--claim-end-date` (ISO `YYYY-MM-DD`). Dates written in the question are extracted automatically when present.

## Running tests

```text
pytest
pytest tests/unit
pytest tests/integration
```

## Running evaluation

```text
$env:PYTHONPATH="src"
python scripts/evaluate.py
python scripts/evaluate.py --dataset original
python scripts/evaluate.py --dataset surprise
```

macOS / Linux:

```text
PYTHONPATH=src python scripts/evaluate.py
```

The runner prints totals measured from executing the selected dataset(s). Default is both `original` and `surprise`. With the stub LLM, answer-correctness is typically low because the stub does not write policy facts; evidence, citation, and abstention scores still reflect retrieval, resolution, and grounding.

## Surprise Challenge

Amendment No. 2026-01 takes effect on **1 March 2026**. It is stored at `data/amendments/Amendment No. 2026-01.md` and is **not** merged into `data/policy/policy-manual.md`. That keeps historical (pre-amendment) text available so the system can choose which version applies.

**How temporal resolution works.** Dates are taken from the question or from CLI flags. Paragraphs 1, 3, and 4 of the amendment (earnings disregard, income thresholds, sanctions, and new `§10.5.3A`) follow the **determination date**. Paragraph 2 (reporting time limits in `§4.3.2` and `§9.1.4`) follows the **date of the change of circumstances**. A claim period that spans 1 March 2026 uses the figures in force on each day and `§7.4.3` apportionment. The resolver runs before the LLM; the model only sees effective evidence.

**Loading the amendment.** There is no separate load command. `create_answer_service()` reads the original manual and the amendment at startup. If the amendment file is missing, startup fails with a clear error rather than answering as if the amendment did not exist.

**Amendment-aware queries.** Include the relevant date in the question, or pass CLI flags. Examples:

```text
python -m grounded_answer ask "For a determination made on 15 February 2026, what is the first monthly earnings disregard?"
python -m grounded_answer ask "For a determination made on 15 March 2026, what is the first monthly earnings disregard?"
python -m grounded_answer ask "The claimant's circumstances changed on 5 March 2026. How many calendar days does the recipient have to report the change?"
python -m grounded_answer ask "The claim runs from 20 February 2026 to 10 March 2026. What earnings disregard figures apply?"
```

**Surprise-challenge evaluation:** `python scripts/evaluate.py --dataset surprise`

**Missing temporal information.** If the question depends on an amended clause and the needed date is not in the question or CLI flags, the system abstains with: `The applicable policy cannot be determined without the relevant date.` It does not assume “today” or the effective date.

## Grounding and abstention

- No evidence, or evidence that does not overlap the question → `INSUFFICIENT`, no LLM call.
- Amended clause needed but the relevant date is missing → clarification abstention, no LLM call.
- LLM citations must exist in retrieved evidence and in the corpus (including inserted `§10.5.3A` and amendment paragraphs such as `¶5.1`).
- An answer that cites only invented IDs is rejected.

## Limitations

- Default LLM is a stub; configure `LLM_PROVIDER` for generated policy prose.
- PageIndex is optional and unused unless API credentials are set. The amendment is always retrieved through the local structure-aware index, including when PageIndex is used for the original manual.
- When a question names a claim period that does not span 1 March 2026 and does not name a determination date, figures are taken as those in force during that period. An explicit determination date is required to apply paragraph 5.1 to an earlier period.

## Troubleshooting

- `No module named grounded_answer`: set `PYTHONPATH=src` from the repository root.
- Unicode errors in the Windows console: the CLI requests UTF-8 stdout; evidence lines use `§`.
- Evaluation answer scores near zero with the stub: expected; use a real `LLM_PROVIDER` to score generated text.
