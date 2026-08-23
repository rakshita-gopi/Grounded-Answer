import json
from unittest.mock import patch

from grounded_answer.embeddings.ollama import OllamaEmbeddingProvider, OllamaUnavailableError


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None


def test_embed_query_uses_instruction_and_normalizes() -> None:
    provider = OllamaEmbeddingProvider(
        "http://ollama:11434",
        "qwen3-embedding:4b",
        query_instruction="Instruct: retrieve policy.\nQuery: {query}",
    )
    captured = {}

    def fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        body = json.loads(request.data.decode("utf-8"))
        captured["body"] = body
        return _FakeResponse({"embeddings": [[3.0, 4.0]]})

    with patch("grounded_answer.embeddings.ollama.urllib.request.urlopen", fake_urlopen):
        vector = provider.embed_query("eligibility")

    assert captured["url"] == "http://ollama:11434/api/embed"
    assert captured["body"]["model"] == "qwen3-embedding:4b"
    assert "eligibility" in captured["body"]["input"][0]
    assert captured["body"]["input"][0].startswith("Instruct:")
    assert abs(vector[0] - 0.6) < 1e-9
    assert abs(vector[1] - 0.8) < 1e-9


def test_embed_documents_does_not_apply_query_instruction() -> None:
    provider = OllamaEmbeddingProvider(
        "http://localhost:11434",
        "qwen3-embedding:4b",
        query_instruction="Instruct: retrieve policy.\nQuery: {query}",
    )

    def fake_urlopen(request, timeout=0):
        body = json.loads(request.data.decode("utf-8"))
        assert body["input"] == ["clause text"]
        return _FakeResponse({"embeddings": [[1.0, 0.0]]})

    with patch("grounded_answer.embeddings.ollama.urllib.request.urlopen", fake_urlopen):
        vectors = provider.embed_documents(["clause text"])
    assert vectors == [[1.0, 0.0]]


def test_unavailable_ollama_raises_user_facing_error() -> None:
    provider = OllamaEmbeddingProvider("http://ollama:11434", "qwen3-embedding:4b")

    def fake_urlopen(request, timeout=0):
        raise TimeoutError("timed out")

    with patch("grounded_answer.embeddings.ollama.urllib.request.urlopen", fake_urlopen):
        try:
            provider.embed_query("hello")
            raised = False
        except OllamaUnavailableError as exc:
            raised = True
            assert "ERROR: Ollama is unavailable" in str(exc)
    assert raised
