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
    # Regression: an OpenRouter 402 "can only afford 180" must shrink the budget BELOW
    # 180 and still yield content — not silently return an empty answer.
    p = OpenAIProvider(model="deepseek/deepseek-chat", api_key="k", base_url="http://openrouter")
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


def test_model_list_is_openai_only(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    data = settings.list_models()
    assert data["current"]["provider"] == "openai"
    assert {o["provider"] for o in data["options"]} == {"openai"}
    assert any(o["model"] == "gpt-4o" for o in data["options"])


def test_route_model_free_providers(monkeypatch):
    from backend.llm.streaming_provider import route_model, GEMINI_BASE, GROQ_BASE
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("GROQ_API_KEY", "q-key")
    assert route_model("gemini-2.5-flash") == (GEMINI_BASE, "g-key")
    assert route_model("llama-3.3-70b-versatile") == (GROQ_BASE, "q-key")
    # a Groq model id containing a slash must NOT be mis-routed to OpenRouter
    assert route_model("openai/gpt-oss-20b")[0] == GROQ_BASE
    # unrelated names still route to their providers
    assert route_model("deepseek/deepseek-chat")[0].endswith("openrouter.ai/api/v1")
    assert route_model("qwen3:8b")[0].endswith("11434/v1")


def test_dropdown_lists_free_groq_and_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    labels = [o["label"] for o in settings.list_models()["options"]]
    assert any("Gemini · gemini-2.5-flash" in x for x in labels)
    # Groq present but flagged as needing a key
    assert any("Groq · llama-3.3-70b-versatile" in x and "add key" in x for x in labels)
