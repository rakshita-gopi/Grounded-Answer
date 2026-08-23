"""LLM port. Application code depends on this interface, not on a vendor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from grounded_answer.domain.evidence import Evidence


@dataclass(frozen=True, slots=True)
class LLMContext:
    question: str
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    provider: str
    model: str


class LLMProvider(ABC):
    """Generate text from a prompt and retrieval context.

    Provider and model are selected by configuration, not by this interface.
    """

    @abstractmethod
    def generate(self, prompt: str, context: LLMContext) -> LLMResponse:
        raise NotImplementedError
