"""Two universal code-agent fixes, fully offline:
  BUG 1 — escalation respects the user's selected model + falls back to an available model on a 429
          (ResilientProvider / model chain) instead of failing the request.
  BUG 2 — a reported value that violates the request's exact definition (wrong point/index/units)
          is caught by the spec-derived held-out checks and regenerated."""
import types

import pytest

import backend.agent.loop as loop


class FakeProv:
    def __init__(self, avail=True, mode="ok", text="hello world"):
        self.avail = avail
        self.mode = mode
        self.text = text
        self.calls = 0

    @property
    def is_available(self):
        return self.avail

    def stream_chat(self, messages, system="", max_tokens=2048, temperature=0.3, yield_reasoning=False):
        self.calls += 1
        if self.mode == "429":
            raise RuntimeError("Error code: 429 - rate limit exceeded")
        if self.mode == "auth":
            raise RuntimeError("401 invalid api key")
        if self.mode == "transient_then_ok" and self.calls < 2:
            raise RuntimeError("503 service unavailable")
        for w in self.text.split():
            yield w + " "


def _no_sleep(monkeypatch):
    monkeypatch.setattr(loop.time, "sleep", lambda *a, **k: None)


# ====================== BUG 1 ======================
def test_is_transient_err_classifies():
    assert loop._is_transient_err("Error code: 429") is True
    assert loop._is_transient_err("503 unavailable") is True
    assert loop._is_transient_err("Connection error / APITimeoutError") is True
    assert loop._is_transient_err("invalid request: bad parameter") is False


def test_resilient_falls_back_on_rate_limit(monkeypatch):
    _no_sleep(monkeypatch)
    a, b = FakeProv(mode="429"), FakeProv(mode="ok", text="B answer here")
    monkeypatch.setattr(loop, "get_provider", lambda m: {"A": a, "B": b}[m])
    events = []
    rp = loop.ResilientProvider(["A", "B"], emit=events.append, max_retries=2)
    out = "".join(t for t in rp.stream_chat([{"role": "user", "content": "x"}]) if isinstance(t, str))
    assert "Banswerhere" in out.replace(" ", "")          # used the fallback model's output
    assert any(e["type"] == "warning" and "switching to B" in e["message"] for e in events)
    assert rp.model == "B"                                 # B preferred next time


def test_resilient_does_not_retry_then_fails_only_if_all_fail(monkeypatch):
    _no_sleep(monkeypatch)
    a, b = FakeProv(mode="429"), FakeProv(mode="429")
    monkeypatch.setattr(loop, "get_provider", lambda m: {"A": a, "B": b}[m])
    rp = loop.ResilientProvider(["A", "B"], max_retries=1)
    with pytest.raises(Exception):                         # raises ONLY because every model failed
        list(rp.stream_chat([{"role": "user", "content": "x"}]))


def test_resilient_retries_transient_then_succeeds(monkeypatch):
    _no_sleep(monkeypatch)
    a = FakeProv(mode="transient_then_ok", text="ok now")
    monkeypatch.setattr(loop, "get_provider", lambda m: {"A": a}[m])
    rp = loop.ResilientProvider(["A"], max_retries=3)
    out = "".join(t for t in rp.stream_chat([{"role": "user", "content": "x"}]) if isinstance(t, str))
    assert "ok" in out and a.calls == 2                    # retried once, then succeeded


def test_resilient_is_available(monkeypatch):
    monkeypatch.setattr(loop, "get_provider", lambda m: {"A": FakeProv(avail=False),
                                                         "B": FakeProv(avail=True)}[m])
    assert loop.ResilientProvider(["A", "B"]).is_available is True
    monkeypatch.setattr(loop, "get_provider", lambda m: FakeProv(avail=False))
    assert loop.ResilientProvider(["A"]).is_available is False


def test_user_selected_model(monkeypatch):
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", loop.DEFAULT_OPENAI_MODEL)
    assert loop._user_selected_model() is False
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.5")
    assert loop._user_selected_model() is True
    monkeypatch.setenv("OPENAI_MODEL", loop.DEFAULT_OPENAI_MODEL)
    monkeypatch.setenv("AGENT_MODEL", "codestral-latest")
    assert loop._user_selected_model() is True


def test_agent_model_chain_user_first_and_available(monkeypatch):
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.5")
    monkeypatch.setenv("AGENT_MODEL_STRONG", "mistral-large-latest")
    avail = {"gpt-5.5": True, "mistral-large-latest": True,
             "gemini-2.5-flash": False, "codestral-latest": False}
    monkeypatch.setattr(loop, "_model_available", lambda m: avail.get(m, False))
    chain = loop._agent_model_chain()
    assert chain[0] == "gpt-5.5"                            # the user's choice comes first
    assert "mistral-large-latest" in chain                 # configured fallback included
    assert "gemini-2.5-flash" not in chain                 # unavailable filtered out


def test_escalated_chain_respects_user_selection(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_STRONG", "mistral-large-latest")
    monkeypatch.setattr(loop, "_model_available", lambda m: True)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.5")          # user explicitly selected
    primary = ["gpt-5.5", "mistral-large-latest"]
    assert loop._escalated_chain(primary) == primary       # keeps the user's chain
    monkeypatch.setenv("OPENAI_MODEL", loop.DEFAULT_OPENAI_MODEL)   # no selection
    esc = loop._escalated_chain(["gemini-2.5-flash", "mistral-large-latest"])
    assert esc[0] == "mistral-large-latest"                # strong leads only with no selection


# ====================== BUG 2 ======================
def test_req_and_invariants_demand_exact_definition():
    req = loop._REQ_SYSTEM.lower()
    assert "exact definition" in req and "point/index" in req
    inv = loop._INVARIANTS_SYSTEM.lower()
    assert "per explicitly requested output" in inv
    assert "point/index" in inv
    assert "initial" in inv and "final" in inv             # the off-by-point example
    assert "wrong point" in inv


def test_verify_heldout_rejects_definition_mismatch(monkeypatch):
    # The spec/definition check (e.g. reported "initial" != value at step 0) fails on the demo input.
    heldout = "def test_invariant_initial():\n    pass\ndef test_hidden_a():\n    pass\n"
    monkeypatch.setattr(loop, "_run_against_tests",
                        lambda *a, **k: (types.SimpleNamespace(
                            ok=True, stdout="", stderr="initial value mismatch", error=""), 1, 2))
    ok, _p, _t, _ = loop._verify_heldout("SOLUTION", heldout, seeds=2)
    assert ok is False
