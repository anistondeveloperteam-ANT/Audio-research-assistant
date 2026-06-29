"""Tests for the OpenAI LLM provider + web model settings. No network/key needed."""
from backend.llm.streaming_provider import (
    OPENAI_BASE,
    OpenAIProvider,
    get_provider,
    route_model,
)
from webapp import settings


def test_openai_model_routes_to_openai_base_not_env(monkeypatch):
    # Regression: even when OPENAI_BASE_URL points at the free Gemini default, a catalog OpenAI model
    # (GPT-5.5) must hit OpenAI's OWN endpoint — not fall back to that env (which sent the request to
    # the wrong base and failed with "Request URL is missing an 'http://' or 'https://' protocol").
    monkeypatch.setenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    monkeypatch.setenv("OPENAI_CLOUD_KEY", "sk-test")
    base, key = route_model("gpt-5.5")
    assert base == OPENAI_BASE == "https://api.openai.com/v1"
    assert key == "sk-test"
    assert get_provider("gpt-5.5").base_url == "https://api.openai.com/v1"


def test_each_catalog_model_routes_to_its_own_base():
    assert route_model("gemini-2.5-flash")[0].startswith("https://generativelanguage.googleapis.com")
    assert route_model("mistral-large-latest")[0] == "https://api.mistral.ai/v1"
    assert route_model("gpt-5.5")[0] == "https://api.openai.com/v1"


def test_unlisted_models_route_by_vendor_prefix():
    # Pixtral (Mistral's VISION model, used for figure understanding) isn't in the catalog, but must
    # hit the Mistral endpoint -- not fall through to OpenAI as it did before.
    assert route_model("pixtral-large-latest")[0] == "https://api.mistral.ai/v1"
    assert route_model("pixtral-12b-latest")[0] == "https://api.mistral.ai/v1"
    assert route_model("ministral-8b-latest")[0] == "https://api.mistral.ai/v1"
    # gemini-prefixed unknown -> gemini; a truly unknown model still falls back to openai
    assert route_model("gemini-9-ultra")[0].startswith("https://generativelanguage.googleapis.com")
    assert route_model("some-unknown-model")[0] == "https://api.openai.com/v1"


def test_get_provider_is_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gemini-2.5-flash")
    p = get_provider()
    assert p.name == "openai"           # one OpenAI-compatible client for every provider
    assert p.model == "gemini-2.5-flash"
    assert p.is_available is True


def test_openai_unavailable_message_names_key():
    # Construct directly so we don't depend on the local .env having a key.
    p = OpenAIProvider(model="gemini-2.5-flash", api_key="")
    assert p.is_available is False
    assert "OPENAI_API_KEY" in p.unavailable_message()


def test_request_variants_have_token_fallbacks():
    # Common shape first, with a max_completion_tokens fallback for stricter APIs.
    v = OpenAIProvider(model="llama-3.3-70b-versatile", api_key="x")._request_variants(100, 0.3)
    assert "max_tokens" in v[0]
    assert any("max_completion_tokens" in variant for variant in v)


class _Delta:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.delta = _Delta(content)
        self.finish_reason = None


class _Chunk:
    def __init__(self, content):
        self.choices = [_Choice(content)]


def test_stream_chat_shrinks_to_affordable_budget(monkeypatch):
    # Regression: a 402 "can only afford 180" must shrink the budget BELOW 180 and still
    # yield content — not silently return an empty answer.
    p = OpenAIProvider(model="gpt-5.5", api_key="k", base_url="")
    tried = []

    def fake_open(client, openai_mod, msgs, budget, temperature):
        tried.append(budget)
        if budget > 180:
            raise Exception("Error code: 402 - requires more credits ... can only afford 180 tokens")
        return [_Chunk("MVDR is "), _Chunk("a beamformer.")]

    monkeypatch.setattr(p, "_open_stream", fake_open)
    out = "".join(c for c in p.stream_chat([{"role": "user", "content": "hi"}], max_tokens=8000)
                  if isinstance(c, str))
    assert out == "MVDR is a beamformer."        # non-empty: the shrink succeeded
    assert tried[0] == 8000 and tried[-1] <= 180  # shrank from 8000 to within the cap


def test_dropdown_lists_the_catalog(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gemini-2.5-flash")
    data = settings.list_models()
    assert data["current"]["provider"] == "openai"   # one OpenAI-compatible client
    models = {o["model"] for o in data["options"]}
    assert {"gemini-2.5-flash", "mistral-large-latest", "codestral-latest", "gpt-5.5"} <= models
    # removed providers must be gone everywhere
    assert not ({"llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen-3-235b-a22b",
                 "llama-3.3-70b", "deepseek/deepseek-chat"} & models)
    by_id = {o["model"]: o for o in data["options"]}
    assert by_id["codestral-latest"]["vendor"] == "Mistral" and by_id["codestral-latest"]["free"]
    assert by_id["gpt-5.5"]["free"] is False


def test_route_model_resolves_each_provider(monkeypatch):
    from backend.llm.streaming_provider import route_model, GEMINI_BASE
    for k in ("GEMINI_API_KEY", "MISTRAL_API_KEY", "OPENAI_CLOUD_KEY"):
        monkeypatch.setenv(k, "k-" + k)
    assert route_model("gemini-2.5-flash") == (GEMINI_BASE, "k-GEMINI_API_KEY")
    assert route_model("mistral-large-latest")[0] == "https://api.mistral.ai/v1"
    assert route_model("codestral-latest")[0] == "https://api.mistral.ai/v1"
    # OpenAI models resolve to OpenAI's EXPLICIT endpoint (not "" -> SDK env fallback).
    assert route_model("gpt-5.5") == ("https://api.openai.com/v1", "k-OPENAI_CLOUD_KEY")


def test_dropdown_marks_missing_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    by_id = {o["model"]: o for o in settings.list_models()["options"]}
    assert by_id["gemini-2.5-flash"]["available"] is True
    assert by_id["codestral-latest"]["available"] is False   # no Mistral key
