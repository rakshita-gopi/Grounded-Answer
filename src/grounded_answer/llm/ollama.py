"""Ollama chat generation. Isolated from retrieval and embeddings."""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping

from grounded_answer.embeddings.config import ollama_base_url, ollama_timeout_seconds
from grounded_answer.llm.base import LLMContext, LLMProvider, LLMResponse
from grounded_answer.llm.prompts import GENERATION_SYSTEM_PROMPT
from grounded_answer.llm.provider import LLMUnavailableError

DEFAULT_OLLAMA_LLM_MODEL = "qwen3:4b"
DEFAULT_OLLAMA_LLM_TIMEOUT_SECONDS = 300
DEFAULT_OLLAMA_NUM_PREDICT = 60
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OllamaLLMProvider(LLMProvider):
    """Generate answers through Ollama /api/chat. Never uses the embedding model."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: int = DEFAULT_OLLAMA_LLM_TIMEOUT_SECONDS,
        num_predict: int = DEFAULT_OLLAMA_NUM_PREDICT,
    ) -> None:
        if not model.strip():
            raise LLMUnavailableError("LLM_MODEL must be set when LLM_PROVIDER=ollama.")
        self._base_url = base_url.rstrip("/")
        self._model = model.strip()
        self._timeout_seconds = timeout_seconds
        self._num_predict = num_predict

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> OllamaLLMProvider:
        model = environ.get("LLM_MODEL", "").strip() or DEFAULT_OLLAMA_LLM_MODEL
        raw_timeout = environ.get("OLLAMA_LLM_TIMEOUT_SECONDS", "").strip()
        if raw_timeout:
            try:
                timeout = max(1, int(raw_timeout))
            except ValueError:
                timeout = DEFAULT_OLLAMA_LLM_TIMEOUT_SECONDS
        else:
            timeout = max(ollama_timeout_seconds(environ), DEFAULT_OLLAMA_LLM_TIMEOUT_SECONDS)
        raw_predict = environ.get("LLM_NUM_PREDICT", "").strip() or environ.get(
            "OLLAMA_NUM_PREDICT", ""
        ).strip()
        num_predict = DEFAULT_OLLAMA_NUM_PREDICT
        if raw_predict:
            try:
                num_predict = max(16, int(raw_predict))
            except ValueError:
                num_predict = DEFAULT_OLLAMA_NUM_PREDICT
        return cls(
            base_url=ollama_base_url(environ),
            model=model,
            timeout_seconds=timeout,
            num_predict=num_predict,
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, context: LLMContext) -> LLMResponse:
        del context
        user_content = prompt
        if self._model.lower().startswith("qwen3"):
            user_content = f"{prompt}\n\n/no_think"
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": GENERATION_SYSTEM_PROMPT,
                },
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": self._num_predict},
        }
        request = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        sys.stderr.write(
            f"Generating with {self._model} (CPU can take 1–4 minutes; do not press Ctrl+C)...\n"
        )
        sys.stderr.flush()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMUnavailableError(
                f"ERROR: Ollama generation request failed ({exc.code}) at {self._base_url}. {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMUnavailableError(
                f"ERROR: Ollama is unavailable at {self._base_url}. "
                "Start it with `docker compose up` or a local Ollama service."
            ) from exc
        except json.JSONDecodeError as exc:
            raise LLMUnavailableError("ERROR: Ollama returned a non-JSON generation response.") from exc
        text = _message_text(body)
        if not text:
            raise LLMUnavailableError("ERROR: Ollama generation response did not contain message content.")
        return LLMResponse(text=text, provider="ollama", model=self._model)

    def ping(self) -> None:
        url = f"{self._base_url}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=self._timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise LLMUnavailableError(
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
            raise LLMUnavailableError(
                f"ERROR: Generation model {self._model!r} is not available in Ollama. "
                f"Pull it with: ollama pull {self._model}"
            )


def _message_text(body: object) -> str:
    if not isinstance(body, dict):
        return ""
    message = body.get("message")
    raw = ""
    if isinstance(message, dict):
        raw = str(message.get("content") or "")
    elif isinstance(body.get("response"), str):
        raw = body["response"]
    return _strip_thinking(raw)


def _strip_thinking(raw: str) -> str:
    text = _THINK_BLOCK.sub("", raw)
    lower = text.lower()
    close = lower.rfind("</think>")
    if close != -1:
        text = text[close + len("</think>") :]
    open_tag = lower.find("<think>")
    if open_tag != -1:
        text = text[:open_tag]
    return text.strip()
