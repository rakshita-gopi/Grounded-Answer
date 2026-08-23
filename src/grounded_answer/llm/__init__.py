"""LLM abstraction used by the rest of the application."""

from grounded_answer.llm.base import LLMContext, LLMProvider, LLMResponse
from grounded_answer.llm.prompts import GROUNDING_INSTRUCTIONS, GENERATION_SYSTEM_PROMPT, build_generation_prompt
from grounded_answer.llm.ollama import OllamaLLMProvider
from grounded_answer.llm.provider import (
    LLMConfig,
    LLMUnavailableError,
    StubLLMProvider,
    create_llm_provider,
)

__all__ = [
    "GROUNDING_INSTRUCTIONS",
    "GENERATION_SYSTEM_PROMPT",
    "LLMConfig",
    "LLMContext",
    "LLMProvider",
    "LLMResponse",
    "LLMUnavailableError",
    "OllamaLLMProvider",
    "StubLLMProvider",
    "build_generation_prompt",
    "create_llm_provider",
]
