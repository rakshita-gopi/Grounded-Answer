import json
from unittest.mock import patch

from grounded_answer.llm.base import LLMContext
from grounded_answer.llm.ollama import OllamaLLMProvider
from grounded_answer.llm.provider import LLMUnavailableError


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None


def test_generate_posts_chat_with_configured_model_not_embedding_model() -> None:
    provider = OllamaLLMProvider("http://ollama:11434", "qwen3:4b")
    captured = {}

    def fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            {"message": {"content": "The first monthly earnings disregard is $120. §6.4.1"}}
        )

    with patch("grounded_answer.llm.ollama.urllib.request.urlopen", fake_urlopen):
        response = provider.generate(
            "prompt",
            LLMContext(question="earnings disregard?", evidence=()),
        )

    assert captured["url"] == "http://ollama:11434/api/chat"
    assert captured["body"]["model"] == "qwen3:4b"
    assert captured["body"]["model"] != "qwen3-embedding:4b"
    assert captured["body"]["stream"] is False
    assert captured["body"]["think"] is False
    assert captured["body"]["messages"][0]["role"] == "system"
    assert "plain-English" in captured["body"]["messages"][0]["content"]
    assert captured["body"]["messages"][-1]["role"] == "user"
    assert captured["body"]["messages"][-1]["content"].startswith("prompt")
    assert "/no_think" in captured["body"]["messages"][-1]["content"]
    assert captured["body"]["options"]["temperature"] == 0
    assert captured["body"]["options"]["num_predict"] == 60
    assert response.provider == "ollama"
    assert response.model == "qwen3:4b"
    assert "$120" in response.text
    assert "§6.4.1" in response.text


def test_generate_strips_qwen_thinking_blocks() -> None:
    provider = OllamaLLMProvider("http://localhost:11434", "qwen3:4b")

    def fake_urlopen(request, timeout=0):
        return _FakeResponse(
            {
                "message": {
                    "content": (
                        "Okay, internal reasoning.\n</think>\n"
                        "The reporting period is 10 calendar days. §4.3.2"
                    )
                }
            }
        )

    with patch("grounded_answer.llm.ollama.urllib.request.urlopen", fake_urlopen):
        response = provider.generate("prompt", LLMContext(question="q", evidence=()))
    assert "internal reasoning" not in response.text
    assert "10 calendar days" in response.text


def test_generate_unavailable_raises_user_facing_error() -> None:
    provider = OllamaLLMProvider("http://ollama:11434", "qwen3:4b")

    def fake_urlopen(request, timeout=0):
        raise TimeoutError("timed out")

    with patch("grounded_answer.llm.ollama.urllib.request.urlopen", fake_urlopen):
        try:
            provider.generate("prompt", LLMContext(question="q", evidence=()))
            raised = False
        except LLMUnavailableError as exc:
            raised = True
            assert "ERROR: Ollama is unavailable" in str(exc)
    assert raised


def test_ping_requires_generation_model() -> None:
    provider = OllamaLLMProvider("http://ollama:11434", "qwen3:4b")

    def fake_urlopen(request, timeout=0):
        return _FakeResponse({"models": [{"name": "qwen3-embedding:4b"}]})

    with patch("grounded_answer.llm.ollama.urllib.request.urlopen", fake_urlopen):
        try:
            provider.ping()
            raised = False
        except LLMUnavailableError as exc:
            raised = True
            assert "qwen3:4b" in str(exc)
    assert raised
