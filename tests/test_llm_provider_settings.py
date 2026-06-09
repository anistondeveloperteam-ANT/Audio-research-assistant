"""Tests for the OpenAI LLM provider + web model settings. No network/key needed."""
from backend.llm.streaming_provider import OpenAIProvider, get_provider
from webapp import settings


def test_get_provider_is_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    p = get_provider()
    assert p.name == "openai"
    assert p.model == "gpt-4o"
    assert p.is_available is True


def test_openai_unavailable_message_names_key():
    # Construct directly so we don't depend on the local .env having a key.
    p = OpenAIProvider(model="gpt-4o", api_key="")
    assert p.is_available is False
    assert "OPENAI_API_KEY" in p.unavailable_message()


def test_request_variants_adapt_to_model():
    # gpt-4o uses max_tokens first; the gpt-5 family uses max_completion_tokens first.
    old = OpenAIProvider(model="gpt-4o", api_key="x")._request_variants(100, 0.3)
    new = OpenAIProvider(model="gpt-5.5", api_key="x")._request_variants(100, 0.3)
    assert "max_tokens" in old[0]
    assert "max_completion_tokens" in new[0]


def test_model_list_is_openai_only(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    data = settings.list_models()
    assert data["current"]["provider"] == "openai"
    assert {o["provider"] for o in data["options"]} == {"openai"}
    assert any(o["model"] == "gpt-4o" for o in data["options"])
