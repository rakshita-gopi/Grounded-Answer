"""Embedding package: provider port and Ollama adapter."""

from grounded_answer.embeddings.base import EmbeddingProvider
from grounded_answer.embeddings.ollama import OllamaEmbeddingProvider, OllamaUnavailableError

__all__ = [
    "EmbeddingProvider",
    "OllamaEmbeddingProvider",
    "OllamaUnavailableError",
]
