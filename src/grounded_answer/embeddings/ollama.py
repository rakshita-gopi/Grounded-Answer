"""Ollama HTTP embedding provider. Isolated from retrieval scoring."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence

from grounded_answer.embeddings.base import EmbeddingProvider
from grounded_answer.embeddings.config import (
    ollama_base_url,
    ollama_embedding_model,
    ollama_timeout_seconds,
    query_instruction_template,
)


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama cannot produce embeddings."""


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: int = 120,
        query_instruction: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._query_instruction = query_instruction

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> OllamaEmbeddingProvider:
        return cls(
            base_url=ollama_base_url(environ),
            model=ollama_embedding_model(environ),
            timeout_seconds=ollama_timeout_seconds(environ),
            query_instruction=query_instruction_template(environ),
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(self._format_query(text))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        batch_size = 8
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            vectors.extend(self._embed_many(chunk))
        return vectors

    def ping(self) -> None:
        url = f"{self._base_url}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=self._timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise OllamaUnavailableError(
                f"ERROR: Ollama is unavailable at {self._base_url}. "
                "Start it with `docker compose up` or a local Ollama service."
            ) from exc
        names = {
            str(item.get("name") or item.get("model") or "")
            for item in payload.get("models", [])
            if isinstance(item, dict)
        }
        matched = self._model in names or any(
            name.startswith(f"{self._model}") for name in names if name
        )
        if not matched:
            raise OllamaUnavailableError(
                f"ERROR: Embedding model {self._model!r} is not available in Ollama. "
                f"Pull it with: ollama pull {self._model}"
            )

    def _format_query(self, text: str) -> str:
        template = self._query_instruction
        if "{query}" in template:
            return template.replace("{query}", text.strip())
        return f"{template}\n{text.strip()}" if template else text.strip()

    def _embed_one(self, text: str) -> list[float]:
        return self._embed_many([text])[0]

    def _embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        payload = {"model": self._model, "input": list(texts)}
        request = urllib.request.Request(
            f"{self._base_url}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaUnavailableError(
                f"ERROR: Ollama embedding request failed ({exc.code}) at {self._base_url}. {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OllamaUnavailableError(
                f"ERROR: Ollama is unavailable at {self._base_url}. "
                "Start it with `docker compose up` or a local Ollama service."
            ) from exc
        except json.JSONDecodeError as exc:
            raise OllamaUnavailableError(
                "ERROR: Ollama returned a non-JSON embedding response."
            ) from exc

        raw = body.get("embeddings")
        if not isinstance(raw, list) or len(raw) != len(texts):
            raise OllamaUnavailableError(
                "ERROR: Ollama embedding response did not contain the expected vectors."
            )
        vectors = [_normalize(_as_floats(item)) for item in raw]
        return vectors


def _as_floats(item: object) -> list[float]:
    if not isinstance(item, list) or not item:
        raise OllamaUnavailableError(
            "ERROR: Ollama returned an empty or invalid embedding vector."
        )
    try:
        return [float(value) for value in item]
    except (TypeError, ValueError) as exc:
        raise OllamaUnavailableError(
            "ERROR: Ollama returned a non-numeric embedding vector."
        ) from exc


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise OllamaUnavailableError("ERROR: Ollama returned a zero embedding vector.")
    return [value / norm for value in vector]
