"""Embedding configuration. Model name lives here, not in application code."""

from collections.abc import Mapping

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "qwen3-embedding:4b"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 120
DEFAULT_INDEX_VERSION = 1
DEFAULT_QUERY_INSTRUCTION = (
    "Instruct: Given a policy question, retrieve the most relevant policy "
    "provisions that directly support the answer.\nQuery: {query}"
)


def ollama_base_url(environ: Mapping[str, str]) -> str:
    return environ.get("OLLAMA_BASE_URL", "").strip() or DEFAULT_OLLAMA_BASE_URL


def ollama_embedding_model(environ: Mapping[str, str]) -> str:
    return (
        environ.get("OLLAMA_EMBEDDING_MODEL", "").strip()
        or DEFAULT_OLLAMA_EMBEDDING_MODEL
    )


def ollama_timeout_seconds(environ: Mapping[str, str]) -> int:
    raw = environ.get("OLLAMA_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_OLLAMA_TIMEOUT_SECONDS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_OLLAMA_TIMEOUT_SECONDS


def query_instruction_template(environ: Mapping[str, str]) -> str:
    return environ.get("OLLAMA_QUERY_INSTRUCTION", "").strip() or DEFAULT_QUERY_INSTRUCTION


def embeddings_requested(environ: Mapping[str, str]) -> bool:
    """True when the caller explicitly enabled the Ollama embedding path.

    Tests pass environ={} and must keep using lexical retrieval.
    """
    backend = environ.get("RETRIEVAL_BACKEND", "").strip().lower()
    if backend in {"lexical", "local"}:
        return False
    if backend in {"ollama", "embedding", "embeddings"}:
        return True
    return bool(environ.get("OLLAMA_BASE_URL", "").strip())
