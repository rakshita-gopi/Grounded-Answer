"""Container entrypoint: wait for Ollama, then idle or run a command."""

from __future__ import annotations

import os
import sys
import time

from grounded_answer.application.bootstrap import create_answer_service
from grounded_answer.embeddings.config import embeddings_requested
from grounded_answer.embeddings.ollama import OllamaEmbeddingProvider, OllamaUnavailableError
from grounded_answer.interfaces.cli.verify import run_verify
from grounded_answer.llm.ollama import OllamaLLMProvider
from grounded_answer.llm.provider import LLMUnavailableError

STARTUP_BANNER = (
    "First startup downloads local AI models and builds the search index. "
    "This may take several minutes. Do not interrupt the process."
)


def _log(message: str) -> None:
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


def wait_for_ollama(attempts: int = 180, delay_seconds: float = 5.0) -> None:
    last_error = "Ollama did not become ready."
    need_embeddings = embeddings_requested(os.environ)
    need_generation = os.environ.get("LLM_PROVIDER", "").strip().lower() == "ollama"
    if not need_embeddings and not need_generation:
        _log("Ollama models are not required for this configuration.")
        return
    embedding = OllamaEmbeddingProvider.from_environ(os.environ) if need_embeddings else None
    generation = OllamaLLMProvider.from_environ(os.environ) if need_generation else None
    _log("Waiting for Ollama models to become reachable...")
    for attempt in range(attempts):
        try:
            if embedding is not None:
                embedding.ping()
            if generation is not None:
                generation.ping()
            _log("Ollama embedding and generation models are reachable.")
            return
        except (OllamaUnavailableError, LLMUnavailableError) as exc:
            last_error = str(exc)
            if attempt == 0 or (attempt + 1) % 6 == 0:
                _log(f"Still waiting for Ollama ({last_error})")
            time.sleep(delay_seconds)
    raise SystemExit(last_error)


def warm_index() -> None:
    if not embeddings_requested(os.environ):
        _log("Lexical retrieval is configured. No embedding index to build.")
        return
    _log("Checking retrieval index (reuses the index volume when it is still valid)...")
    create_answer_service(environ=os.environ, load_dotenv=False)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    _log(STARTUP_BANNER)
    wait_for_ollama()
    try:
        warm_index()
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    if not args or args == ["idle"]:
        _log("Startup checks complete. The application is ready for queries.")
        run_verify()
        sys.stdout.write(
            "\nGrounded Answer is ready. Ask a question with:\n"
            '  docker compose exec grounded-answer python -m grounded_answer ask "..."\n'
        )
        sys.stdout.flush()
        while True:
            time.sleep(3600)
    os.execvp(args[0], args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
