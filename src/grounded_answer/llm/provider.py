"""LLM provider implementations selected by configuration."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass

from grounded_answer.llm.base import LLMContext, LLMProvider, LLMResponse


class LLMUnavailableError(RuntimeError):
    """Raised when the configured LLM backend cannot be used."""


@dataclass(frozen=True, slots=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None) -> LLMConfig:
        env = environ if environ is not None else os.environ
        provider = env.get("LLM_PROVIDER", "").strip().lower()
        model = env.get("LLM_MODEL", "").strip()
        api_key = env.get("LLM_API_KEY", "").strip() or env.get("OPENAI_API_KEY", "").strip()
        base_url = env.get("LLM_BASE_URL", "https://api.openai.com/v1").strip()
        return cls(provider=provider, model=model, api_key=api_key, base_url=base_url.rstrip("/"))


class StubLLMProvider(LLMProvider):
    """Deterministic stand-in used when no external LLM is configured."""

    def __init__(self, text: str = "", model: str = "stub") -> None:
        self._text = text
        self._model = model
        self.calls: list[tuple[str, LLMContext]] = []

    def generate(self, prompt: str, context: LLMContext) -> LLMResponse:
        self.calls.append((prompt, context))
        text = self._text or (
            "I don't know based on the supplied policy evidence. "
            "(stub LLM provider; configure LLM_PROVIDER and LLM_MODEL to generate answers)"
        )
        return LLMResponse(text=text, provider="stub", model=self._model)


class OpenAICompatibleProvider(LLMProvider):
    """Chat-completions client. Works with OpenAI and compatible base URLs."""

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        if not api_key:
            raise LLMUnavailableError("An API key is required for the configured LLM provider.")
        if not model:
            raise LLMUnavailableError("LLM_MODEL must be set for the configured LLM provider.")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url

    def generate(self, prompt: str, context: LLMContext) -> LLMResponse:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise LLMUnavailableError(f"LLM request failed: {exc}") from exc
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError("LLM response did not contain message content.") from exc
        return LLMResponse(text=text, provider="openai-compatible", model=self._model)


def create_llm_provider(environ: Mapping[str, str] | None = None) -> LLMProvider:
    config = LLMConfig.from_environ(environ)
    if config.provider in {"", "stub"}:
        return StubLLMProvider(model=config.model or "stub")
    if config.provider in {"openai", "openai-compatible"}:
        return OpenAICompatibleProvider(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
        )
    if config.provider == "ollama":
        from grounded_answer.llm.ollama import OllamaLLMProvider

        return OllamaLLMProvider.from_environ(environ if environ is not None else os.environ)
    raise LLMUnavailableError(f"Unknown LLM_PROVIDER: {config.provider!r}")
