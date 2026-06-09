"""Tests for LLM provider selection (OpenAI + OpenRouter). No network/key needed."""
from backend.llm.streaming_provider import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL,
    GeminiProvider,
    OpenAIProvider,
    OpenRouterProvider,
    get_provider,
)
from webapp import settings


def test_get_provider_selects_gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    p = get_provider()
    assert p.name == "gemini"
    assert p.model == DEFAULT_GEMINI_MODEL
    assert "generativelanguage.googleapis.com" in (p.base_url or "")


def test_gemini_unavailable_message_names_key():
    p = GeminiProvider(model="gemini-2.5-flash", api_key="")
    assert p.is_available is False
    assert "GEMINI_API_KEY" in p.unavailable_message()


def test_get_provider_selects_openrouter(monkeypatch):
    # All set in the process env -> load_dotenv(override=False) can't change them.
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    p = get_provider()
    assert p.name == "openrouter"
    assert p.model == DEFAULT_OPENROUTER_MODEL
    assert p.base_url == DEFAULT_OPENROUTER_BASE_URL


def test_get_provider_defaults_to_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    p = get_provider()
    assert p.name == "openai"
    assert p.model == "gpt-4o"


def test_openrouter_unavailable_message_names_key():
    # Construct directly so we don't depend on the local .env having a key.
    p = OpenRouterProvider(model="deepseek/deepseek-chat", api_key="")
    assert p.is_available is False
    assert "OPENROUTER_API_KEY" in p.unavailable_message()


def test_openai_unavailable_message_names_key():
    p = OpenAIProvider(model="gpt-4o", api_key="")
    assert "OPENAI_API_KEY" in p.unavailable_message()


def test_model_list_has_both_providers(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL", "deepseek/deepseek-r1")
    data = settings.list_models()
    assert data["current"] == {"provider": "openrouter", "model": "deepseek/deepseek-r1"}
    provs = {o["provider"] for o in data["options"]}
    assert "openai" in provs and "openrouter" in provs
    assert any(o["model"] == "deepseek/deepseek-chat" for o in data["options"])
