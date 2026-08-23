from grounded_answer.domain.evidence import Evidence
from grounded_answer.llm.base import LLMContext
from grounded_answer.llm.prompts import GROUNDING_INSTRUCTIONS, build_generation_prompt
from grounded_answer.llm.provider import (
    LLMUnavailableError,
    StubLLMProvider,
    create_llm_provider,
)


def _evidence() -> Evidence:
    return Evidence(
        evidence_id="policy-manual.md:§6.6.1",
        clause_id="§6.6.1",
        content="A household is not eligible where countable income exceeds the applicable threshold.",
        source="policy-manual.md",
    )


def test_prompt_includes_question_evidence_and_grounding_rules() -> None:
    prompt = build_generation_prompt("What is the income threshold?", [_evidence()])
    assert "USER QUESTION" in prompt
    assert "What is the income threshold?" in prompt
    assert "RETRIEVED POLICY EVIDENCE" in prompt
    assert "§6.6.1" in prompt
    assert "policy-manual.md" in prompt
    assert "GROUNDING INSTRUCTIONS" in prompt
    assert GROUNDING_INSTRUCTIONS in prompt
    assert "Use only the supplied evidence." in prompt
    assert "Do not invent policy." in prompt
    assert "Preserve qualifications and exceptions." in prompt
    assert "Cite the relevant clause." in prompt
    assert "If evidence is insufficient, explicitly say so." in prompt


def test_stub_provider_receives_prompt_and_context() -> None:
    provider = StubLLMProvider(text="insufficient")
    context = LLMContext(question="What is an applicant?", evidence=(_evidence(),))
    prompt = build_generation_prompt(context.question, context.evidence)

    response = provider.generate(prompt, context)

    assert response.text == "insufficient"
    assert response.provider == "stub"
    assert provider.calls[0][0] == prompt
    assert provider.calls[0][1].question == "What is an applicant?"
    assert provider.calls[0][1].evidence[0].clause_id == "§6.6.1"


def test_factory_uses_stub_when_provider_unset() -> None:
    provider = create_llm_provider(environ={})
    assert isinstance(provider, StubLLMProvider)
    response = provider.generate("prompt", LLMContext(question="q", evidence=()))
    assert response.provider == "stub"


def test_factory_reads_provider_and_model_from_config() -> None:
    provider = create_llm_provider(environ={"LLM_PROVIDER": "stub", "LLM_MODEL": "local-test"})
    response = provider.generate("prompt", LLMContext(question="q", evidence=()))
    assert response.model == "local-test"


def test_factory_requires_key_and_model_for_openai() -> None:
    try:
        create_llm_provider(environ={"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-test"})
        raised = False
    except LLMUnavailableError:
        raised = True
    assert raised


def test_factory_rejects_unknown_provider() -> None:
    try:
        create_llm_provider(environ={"LLM_PROVIDER": "not-a-vendor"})
        raised = False
    except LLMUnavailableError:
        raised = True
    assert raised
