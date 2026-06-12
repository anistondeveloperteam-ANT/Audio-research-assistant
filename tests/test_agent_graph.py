"""
PHASE 3 #2: the multi-agent graph. All wrapped functions are stubbed, so these run fully
offline (no LLM, Docker, Redis, or network) and exercise the graph structure:
fan-in of both fetchers, the grader loop honoring the round cap, and the fallback.
"""
import pytest

import backend.agent.graph as G
from backend.agent.code_runner import RunResult
from backend.agent.hooks import HookDecision


class _FakeProvider:
    is_available = True


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_BACKEND", "none")        # no checkpoint file in tests
    monkeypatch.setenv("MAX_TOKENS_PER_TASK", "1000000")
    monkeypatch.setattr(G, "get_provider", lambda *a, **k: _FakeProvider())
    monkeypatch.setattr(G, "docker_available", lambda: True)
    monkeypatch.setattr(G, "pre_run", lambda code, task="": HookDecision(True, ""))
    monkeypatch.setattr(G, "run_python", lambda code, **k: RunResult(True, 0, "out\n", "", 0.1))
    import webapp.chat_logic as CL
    monkeypatch.setattr(CL, "local_rag_enabled", lambda: False)
    import backend.external_search as ES
    monkeypatch.setattr(ES, "is_web_search_enabled", lambda: False)
    return monkeypatch


def test_happy_path_passes_in_one_round(stubbed):
    stubbed.setenv("AGENTIC_MAX_VERIFY_ROUNDS", "3")
    stubbed.setenv("AGENTIC_MIN_VERIFY_SCORE", "80")
    stubbed.setattr(G, "_generate_code", lambda *a: "print('x')")
    stubbed.setattr(G, "review", lambda text, task="": {
        "summary": "ok", "scores": {"relevance": 10}, "recommendation": "accept", "suggestions": []})
    stubbed.setattr(G, "verify_answer", lambda provider, **k: {"ok": True, "score": 95})

    events = []
    res = G.run_agent_graph("do x", on_event=events.append)
    assert res.success is True
    assert res.best_code == "print('x')"
    assert [e["type"] for e in events] == ["think", "code", "run", "run_result", "reflect", "final"]


def test_loops_to_round_cap_when_below_threshold(stubbed):
    stubbed.setenv("AGENTIC_MAX_VERIFY_ROUNDS", "2")
    stubbed.setenv("AGENTIC_MIN_VERIFY_SCORE", "80")
    stubbed.setattr(G, "_generate_code", lambda *a: "print('x')")
    stubbed.setattr(G, "review", lambda text, task="": {
        "summary": "weak", "scores": {"relevance": 9}, "recommendation": "minor revision", "suggestions": []})
    stubbed.setattr(G, "verify_answer", lambda provider, **k: {"ok": False, "score": 30})  # always fails

    rounds = [e["iteration"] for e in _collect(G, "do x") if e["type"] == "think"]
    assert rounds == [1, 2]            # planner ran exactly to the cap, no further


def test_fanin_merges_both_fetchers_into_coder_context(stubbed):
    captured = {}

    def cap_gen(provider, memory_context, last, directive):
        captured["brief"] = memory_context
        return "print(1)"

    stubbed.setattr(G, "_generate_code", cap_gen)
    stubbed.setattr(G, "review", lambda text, task="": {
        "summary": "ok", "scores": {"relevance": 10}, "recommendation": "accept", "suggestions": []})
    stubbed.setattr(G, "verify_answer", lambda provider, **k: {"ok": True, "score": 95})
    # turn BOTH fetchers on with stub evidence
    import webapp.chat_logic as CL
    stubbed.setattr(CL, "local_rag_enabled", lambda: True)
    import backend.retrieval.hybrid_retrieve as HR
    stubbed.setattr(HR, "hybrid_retrieve", lambda q, top_k=6: [{"title": "L", "text": "local-evidence"}])
    import backend.external_search as ES
    stubbed.setattr(ES, "is_web_search_enabled", lambda: True)

    class _S:
        title, text, snippet = "E", "external-evidence", ""

    stubbed.setattr(ES, "gather_external_evidence", lambda q, max_results=6: ([_S()], []))

    G.run_agent_graph("do x", on_event=lambda e: None)
    assert "local-evidence" in captured["brief"]       # fan-in reducer merged...
    assert "external-evidence" in captured["brief"]     # ...both fetchers' evidence


def test_falls_back_to_run_agent_without_langgraph(monkeypatch):
    monkeypatch.setattr(G, "graph_available", lambda: False)
    called = {}

    def fake_run_agent(task, **kw):
        called["task"] = task
        return G.AgentResult(task, True, "code", "out", "answer", [])

    monkeypatch.setattr(G, "run_agent", fake_run_agent)
    res = G.run_agent_graph("fallback task", on_event=lambda e: None)
    assert called["task"] == "fallback task"
    assert res.answer == "answer"


def _collect(mod, task):
    events = []
    mod.run_agent_graph(task, on_event=events.append)
    return events
