"""Tests for the OpenAI LLM provider + web model settings. No network/key needed."""
from backend.llm.streaming_provider import OpenAIProvider, get_provider
from webapp import settings


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


def test_dropdown_has_exactly_two_models(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gemini-2.5-flash")
    data = settings.list_models()
    assert data["current"]["provider"] == "openai"   # one OpenAI-compatible client
    assert {o["model"] for o in data["options"]} == {"gemini-2.5-flash", "gpt-5.5"}


def test_route_model_two_providers(monkeypatch):
    from backend.llm.streaming_provider import route_model, GEMINI_BASE
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("OPENAI_CLOUD_KEY", "o-key")
    assert route_model("gemini-2.5-flash") == (GEMINI_BASE, "g-key")
    assert route_model("gpt-5.5") == ("", "o-key")     # OpenAI (empty base = api.openai.com)


def test_dropdown_labels_and_add_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.delenv("OPENAI_CLOUD_KEY", raising=False)
    labels = [o["label"] for o in settings.list_models()["options"]]
    assert any("Gemini · gemini-2.5-flash" in x for x in labels)
    # GPT-5.5 present but flagged as needing a key
    assert any("OpenAI · gpt-5.5" in x and "add key" in x for x in labels)
