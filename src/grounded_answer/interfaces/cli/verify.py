"""Human-readable environment checks for local and Docker evaluation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from grounded_answer.amendments.service import (
    AMENDMENT_FILENAME,
    DEFAULT_AMENDMENTS_DIR,
    AmendmentIngestionError,
    AmendmentIngestionService,
)
from grounded_answer.embeddings.config import embeddings_requested, ollama_embedding_model
from grounded_answer.embeddings.ollama import OllamaEmbeddingProvider, OllamaUnavailableError
from grounded_answer.ingestion.service import DEFAULT_CORPUS_DIR, IngestionService
from grounded_answer.llm.ollama import OllamaLLMProvider
from grounded_answer.llm.provider import LLMUnavailableError

REPO_ROOT = Path(__file__).resolve().parents[3]


def run_verify(environ: dict[str, str] | None = None, out=None) -> int:
    env = dict(environ) if environ is not None else dict(os.environ)
    stream = out or sys.stdout
    checks: list[tuple[str, bool, str]] = []

    policy_ok, policy_note = _check_policy()
    checks.append(("Policy manual", policy_ok, policy_note))
    amendment_ok, amendment_note = _check_amendment()
    checks.append(("Amendment corpus", amendment_ok, amendment_note))

    if embeddings_requested(env):
        ollama_ok, ollama_note = _check_ollama(env)
        checks.append(("Ollama service", ollama_ok, ollama_note))
        model_ok = ollama_ok
        model_note = (
            f"Model {ollama_embedding_model(env)} reachable"
            if ollama_ok
            else ollama_note
        )
        checks.append(("Qwen3 Embedding 4B", model_ok, model_note))
        index_ok, index_note = _check_index_dir(env)
        checks.append(("Index", index_ok, index_note))
    else:
        checks.append(("Ollama service", True, "not required (lexical retrieval)"))
        checks.append(("Qwen3 Embedding 4B", True, "not required (lexical retrieval)"))
        checks.append(("Index", True, "lexical retriever does not need a vector index"))

    if env.get("LLM_PROVIDER", "").strip().lower() == "ollama":
        gen_ok, gen_note = _check_generation(env)
        checks.append(("Qwen3 4B generation", gen_ok, gen_note))
    else:
        checks.append(("Qwen3 4B generation", True, "stub LLM (LLM_PROVIDER is not ollama)"))

    config_ok = policy_ok and amendment_ok and all(item[1] for item in checks)
    checks.append(("Configuration", config_ok, "required corpora and retrieval settings"))

    stream.write("Grounded Answer — Environment Check\n\n")
    failed: list[str] = []
    for label, ok, note in checks:
        mark = "✓" if ok else "✗"
        stream.write(f"{mark} {label}\n")
        if not ok:
            failed.append(f"{label}: {note}")
    stream.write("\n")
    if failed:
        stream.write("Not ready.\n")
        for item in failed:
            stream.write(f"- {item}\n")
        return 1
    stream.write("System ready.\n")
    return 0


def _check_policy() -> tuple[bool, str]:
    path = DEFAULT_CORPUS_DIR / "policy-manual.md"
    if not path.exists():
        return False, f"Missing {path}. Restore data/policy/policy-manual.md."
    try:
        IngestionService(DEFAULT_CORPUS_DIR).load_policy()
    except Exception as exc:
        return False, f"Could not ingest the policy manual: {exc}"
    return True, str(path)


def _check_amendment() -> tuple[bool, str]:
    path = DEFAULT_AMENDMENTS_DIR / AMENDMENT_FILENAME
    if not path.exists():
        return False, f"Missing {path}."
    try:
        AmendmentIngestionService(DEFAULT_AMENDMENTS_DIR).load_amendment()
    except AmendmentIngestionError as exc:
        return False, str(exc)
    return True, str(path)


def _check_ollama(env: dict[str, str]) -> tuple[bool, str]:
    try:
        OllamaEmbeddingProvider.from_environ(env).ping()
    except OllamaUnavailableError as exc:
        return False, str(exc)
    return True, "reachable"


def _check_index_dir(env: dict[str, str]) -> tuple[bool, str]:
    index_dir = Path(env.get("INDEX_DIR", "").strip() or (REPO_ROOT / "data" / "index"))
    if index_dir.exists():
        return True, f"{index_dir} (built on first retrieval if empty)"
    try:
        index_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"Cannot create {index_dir}: {exc}"
    return True, f"{index_dir} will be built on first retrieval"


def _check_generation(env: dict[str, str]) -> tuple[bool, str]:
    try:
        provider = OllamaLLMProvider.from_environ(env)
        provider.ping()
    except LLMUnavailableError as exc:
        return False, str(exc)
    return True, f"Model {provider.model_name} reachable"
