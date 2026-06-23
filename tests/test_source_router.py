"""SOURCE ROUTER: decide WHICH source a question needs BEFORE answering, so the indexed corpus is not
the default for everything. A general-knowledge / reasoning question is answered from the model's own
knowledge (no corpus, no citations, no 'the sources' framing); a current/time-sensitive question goes to
a WEB search (never stale corpus/training presented as current); a corpus-subject question retrieves and
cites; mixed uses both.

Proves:
  (a) a general-knowledge/reasoning question -> reasoning, with NO retrieval and NO citation;
  (b) a current/time-sensitive question -> WEB search (local corpus skipped), answered from web;
  (c) a corpus-subject question -> retrieval + citation;
  (e) a mixed question -> corpus + web;
  plus decide_source unit behaviour (deterministic fast-paths, verdict mapping, fail-open) and prompt.
(d) — no citation on an unsupported claim / no irrelevant source injects an error — is enforced by the
relevance gate (tests/test_source_relevance_gate.py) and is structurally guaranteed on the reasoning
route here: no source is retrieved at all (test (a)).

Deterministic + offline: provider/memory mocked, and the router verdict is forced per test. No network.
"""
import pytest

import webapp.chat_logic as cl
from backend.answering import source_router as sr
from backend.memory.store import MemoryStore


# ======================================================================================
# Unit: decide_source / classify_source / prompt.
# ======================================================================================
class _R:
    is_available = True

    def __init__(self, word):
        self.word = word

    def stream_chat(self, messages, system="", **k):
        return [self.word]


@pytest.fixture(autouse=True)
def _enable_router(monkeypatch):
    monkeypatch.setenv("SOURCE_ROUTER", "true")
    sr.clear_cache()


def test_deterministic_fast_paths():
    assert sr.decide_source(None, "what is 6 x 5 step by step", freshness=False, calc=True) == sr.REASONING
    assert sr.decide_source(None, "the latest GPT model", freshness=True, calc=False) == sr.WEB


def test_llm_verdicts_map_to_routes():
    assert sr.decide_source(_R("reasoning"), "q1", freshness=False, calc=False) == sr.REASONING
    sr.clear_cache()
    assert sr.decide_source(_R("web"), "q2", freshness=False, calc=False) == sr.WEB
    sr.clear_cache()
    assert sr.decide_source(_R("documents"), "q3", freshness=False, calc=False) == sr.CORPUS
    sr.clear_cache()
    assert sr.decide_source(_R('"current"'), "q4", freshness=False, calc=False) == sr.WEB


def test_fail_open_to_corpus():
    # unavailable provider, unparseable verdict, and disabled router all fall open to corpus (retrieve).
    assert sr.decide_source(None, "qa", freshness=False, calc=False) == sr.CORPUS
    sr.clear_cache()
    assert sr.decide_source(_R("garble"), "qb", freshness=False, calc=False) == sr.CORPUS


def test_router_disabled_returns_corpus(monkeypatch):
    monkeypatch.setenv("SOURCE_ROUTER", "false")
    assert sr.decide_source(_R("reasoning"), "qc", freshness=False, calc=False) == sr.CORPUS


def test_prompt_distinguishes_the_three_bases():
    s = sr._SOURCE_SYSTEM.lower()
    assert "reasoning" in s and "web" in s and "documents" in s
    assert "requires" in s and ("latest" in s or "current" in s) and "own knowledge" in s


# ======================================================================================
# End-to-end wiring: each route drives the correct path. The router verdict is forced.
# ======================================================================================
class _P:
    is_available = True
    model = "fake"

    def stream_chat(self, messages, system="", **k):
        s = (system or "").lower()
        if "answer-quality judge" in s:
            return ['{"ok": true, "score": 95}']
        if "own knowledge and step-by-step reasoning" in s:                 # reasoning draft
            return ["Photosynthesis converts light, water, and CO2 into glucose and oxygen. " * 2]
        if "meticulous, broad-domain research assistant" in s:              # evidence draft (cites)
            return ["Per the source, the method works as follows [1]. " * 2]
        return [messages[-1]["content"] if messages else ""]

    def unavailable_message(self):
        return "n/a"


def _base_env(monkeypatch, *, web, local):
    for k, v in {"ENABLE_ANSWER_CACHE": "false", "ENABLE_LOCAL_RAG": "true" if local else "false",
                 "ENABLE_WEB_SEARCH": "true" if web else "false", "CRAG_ENABLED": "true",
                 "AUTO_REVIEW": "false", "CODE_INTENT_SEMANTIC": "false",
                 "ENABLE_AGENTIC_ANSWER_LOOP": "false", "AGENTIC_INDEPENDENT_VERIFY": "false"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(cl, "_deep_queries", lambda q: [q])
    monkeypatch.setattr("backend.answering.query_refine.refine_query", lambda q: q)
    monkeypatch.setattr(cl, "get_provider", lambda *a, **k: _P())


def _drive(sid, q):
    done, events = None, []
    for ev in cl.stream_chat_events(sid, q):
        events.append(ev)
        if ev["type"] in ("done", "error", "sanity"):
            done = ev
            break
    return done, events


def _fail(*a, **k):
    raise AssertionError("this source must not be searched for this route")


def test_a_general_knowledge_routes_to_reasoning_no_retrieval_no_citation(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    monkeypatch.setattr(cl, "_memory", mem)
    _base_env(monkeypatch, web=True, local=True)
    monkeypatch.setattr(cl, "decide_source", lambda *a, **k: sr.REASONING)
    monkeypatch.setattr(cl, "_gather_local_items", _fail)      # reasoning route must NOT retrieve
    monkeypatch.setattr(cl, "_gather_external_items", _fail)

    done, _ = _drive(sid, "What is photosynthesis?")
    assert done and done["type"] == "done"
    assert "photosynthesis" in done["answer"].lower()          # answered from reasoning
    assert "[1]" not in done["answer"]                         # no corpus/web citation forced
    assert done.get("cached") is not True


def test_b_current_question_routes_to_web_skipping_local_corpus(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    monkeypatch.setattr(cl, "_memory", mem)
    _base_env(monkeypatch, web=True, local=True)
    monkeypatch.setattr(cl, "decide_source", lambda *a, **k: sr.WEB)
    monkeypatch.setattr(cl, "_gather_local_items", _fail)      # WEB route must skip the static corpus
    ext_calls = {"n": 0}

    def ext(q, k):
        ext_calls["n"] += 1
        return ([{"source_type": "web", "title": "News 2026", "url": "http://x/n",
                  "text": "the current state as of 2026", "score": 0.8}], [])

    monkeypatch.setattr(cl, "_gather_external_items", ext)
    done, _ = _drive(sid, "what is the latest model this year")
    assert done and done["type"] == "done"
    assert ext_calls["n"] >= 1                                 # the web WAS searched (not the corpus)


def test_c_corpus_subject_routes_to_retrieval_and_cites(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    monkeypatch.setattr(cl, "_memory", mem)
    _base_env(monkeypatch, web=True, local=True)
    monkeypatch.setattr(cl, "decide_source", lambda *a, **k: sr.CORPUS)
    local_calls = {"n": 0}

    def local(q, mode):
        local_calls["n"] += 1                                 # STRONG: two high-relevance chunks
        return ([{"source_type": "local_pdf", "title": "Beamforming Paper", "section": "Method",
                  "text": "MVDR keeps the target direction undistorted", "score": 0.72,
                  "page_start": 1, "page_end": 2},
                 {"source_type": "local_pdf", "title": "Beamforming Paper", "section": "Results",
                  "text": "noise is minimized in other directions", "score": 0.68,
                  "page_start": 3, "page_end": 4}], [])

    monkeypatch.setattr(cl, "_gather_local_items", local)
    monkeypatch.setattr(cl, "_gather_external_items", _fail)   # STRONG -> corpus only, no web
    done, events = _drive(sid, "How does MVDR beamforming reduce noise in the papers?")
    assert done and done["type"] == "done"
    assert local_calls["n"] >= 1                               # the corpus WAS retrieved
    assert "[1]" in done["answer"]                             # and cited


def test_e_mixed_uses_corpus_and_web(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    monkeypatch.setattr(cl, "_memory", mem)
    _base_env(monkeypatch, web=True, local=True)
    monkeypatch.setattr(cl, "decide_source", lambda *a, **k: sr.CORPUS)
    calls = {"local": 0, "ext": 0}

    def local(q, mode):
        calls["local"] += 1                                   # ONE relevant chunk -> PARTIAL
        return ([{"source_type": "local_pdf", "title": "Paper", "section": "Intro",
                  "text": "established finding about the method", "score": 0.6,
                  "page_start": 1, "page_end": 2}], [])

    def ext(q, k):
        calls["ext"] += 1                                     # PARTIAL also searches the web
        return ([{"source_type": "web", "title": "Recent Update", "url": "http://x/u",
                  "text": "a more recent development", "score": 0.6}], [])

    monkeypatch.setattr(cl, "_gather_local_items", local)
    monkeypatch.setattr(cl, "_gather_external_items", ext)
    # (avoid freshness words like 'recent/latest' here — those force the web-only route by design)
    done, _ = _drive(sid, "What do the papers and broader literature say about the method's tradeoffs?")
    assert done and done["type"] == "done"
    assert calls["local"] >= 1 and calls["ext"] >= 1          # PARTIAL -> corpus + web (mixed)
