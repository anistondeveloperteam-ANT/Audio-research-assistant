"""SOURCE-RELEVANCE GATE: a retrieved source may only ground / be cited in the answer if it
genuinely addresses the question. Topically-similar-but-irrelevant hits (which reranker scores
can't catch) are set aside BEFORE drafting; if none are relevant, the question is answered from
reasoning with no spurious citation. The answer follows the correct reasoning/evidence, never
whatever happened to be retrieved.

Proves:
  (a) when retrieved sources are irrelevant, they are discarded and the answer comes from reasoning
      with no citation;
  (b) when a source genuinely supports the answer, it IS kept and cited;
  (c) mixed retrieval keeps the relevant source and drops the irrelevant one (no citation to a
      source that doesn't support the claim, no answer distortion from the dropped source);
  plus unit behaviour (subset / none / all / unseen-kept / fail-open) and prompt content.

Deterministic + offline: the provider is mocked (content-routed). No network, no Docker, no LLM.
"""
import pytest

import backend.answering.relevance_gate as rg
import webapp.chat_logic as cl
from backend.memory.store import MemoryStore


@pytest.fixture(autouse=True)
def _enable_gate(monkeypatch):
    # conftest disables the relevance gate suite-wide for deterministic routing; these tests
    # exercise the gate itself, so opt back in (the provider is mocked, so still offline).
    monkeypatch.setenv("SOURCE_RELEVANCE_GATE", "true")


# ==============================================================================================
# Unit: relevant_source_indices — the judge + fail-open contract.
# ==============================================================================================
class _Judge:
    is_available = True
    model = "fake"

    def __init__(self, reply, *, raise_exc=None):
        self.reply, self._raise = reply, raise_exc

    def stream_chat(self, messages, system="", **k):
        if self._raise is not None:
            raise self._raise
        return [self.reply]


_ITEMS3 = [
    {"title": "A", "text": "alpha"},
    {"title": "B", "text": "bravo"},
    {"title": "C", "text": "charlie"},
]


def _idx(reply, items=_ITEMS3, **kw):
    return rg.relevant_source_indices(_Judge(reply), question="q", items=items, **kw)


def test_judge_keeps_only_relevant_indices():
    assert _idx('{"relevant": [2]}') == {2}


def test_judge_none_relevant_returns_empty_set():
    assert _idx('{"relevant": []}') == set()           # genuinely none -> discard all -> reason


def test_judge_all_relevant():
    assert _idx('{"relevant": [1, 2, 3]}') == {1, 2, 3}


def test_out_of_range_indices_are_dropped():
    assert _idx('{"relevant": [1, 99]}') == {1}


def test_unseen_sources_beyond_cap_are_kept():
    items = [{"title": str(i), "text": "t"} for i in range(14)]
    # Only the first 12 are judged; the judge keeps #1, and the 2 unjudged (#13, #14) are kept too.
    assert rg.relevant_source_indices(_Judge('{"relevant": [1]}'),
                                      question="q", items=items, max_items=12) == {1, 13, 14}


def test_fail_open_on_unparseable_verdict():
    assert _idx("not json at all") == {1, 2, 3}         # can't parse -> keep ALL (never strip)
    assert _idx('{"other": [1]}') == {1, 2, 3}          # wrong shape -> keep ALL


def test_fail_open_on_provider_exception():
    res = rg.relevant_source_indices(_Judge("", raise_exc=RuntimeError("boom")),
                                     question="q", items=_ITEMS3)
    assert res == {1, 2, 3}


def test_fail_open_when_provider_unavailable():
    class _Down:
        is_available = False

        def stream_chat(self, *a, **k):
            raise AssertionError("must not be called when unavailable")

    assert rg.relevant_source_indices(_Down(), question="q", items=_ITEMS3) == {1, 2, 3}


def test_fail_open_when_gate_disabled(monkeypatch):
    monkeypatch.setenv("SOURCE_RELEVANCE_GATE", "false")
    assert _idx('{"relevant": []}') == {1, 2, 3}        # disabled -> keep ALL (no judge effect)


def test_empty_items_returns_empty():
    assert rg.relevant_source_indices(_Judge('{"relevant": []}'), question="q", items=[]) == set()


def test_relevance_prompt_is_strict_about_direct_relevance():
    s = rg._RELEVANCE_SYSTEM.lower()
    assert "directly" in s                              # direct relevance, not topical overlap
    assert "same broad topic" in s and "strict" in s    # explicitly reject same-topic-only
    assert "relevant" in s


# ==============================================================================================
# End-to-end via stream_chat_events: the gate decides grounding vs. reasoning, and citations.
# ==============================================================================================
class _Provider:
    """Content-routed mock: relevance verdict + evidence/reasoning drafts + verifiers."""
    is_available = True
    model = "fake"

    def __init__(self, *, relevant_json, evidence_answer="(unused)"):
        self.relevant_json = relevant_json
        self.evidence_answer = evidence_answer

    def stream_chat(self, messages, system="", **k):
        s = (system or "").lower()
        if "strict relevance filter" in s:                          # THE GATE
            return [self.relevant_json]
        if "own knowledge and step-by-step reasoning" in s:         # reasoning draft
            return ["Approximately 30.3 MB. 44100*16*2*180/8 = 31,752,000 bytes ~= 30.3 MB. " * 2]
        if "answer-quality judge" in s:                             # reasoning verify
            return ['{"ok": true, "score": 95}']
        if "meticulous, broad-domain research assistant" in s:      # evidence draft
            return [self.evidence_answer]
        return [messages[-1]["content"] if messages else ""]

    def unavailable_message(self):
        return "n/a"


def _setup(monkeypatch, tmp_path, *, local_items):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    monkeypatch.setattr(cl, "_memory", mem)
    for k, v in {"ENABLE_ANSWER_CACHE": "false", "ENABLE_LOCAL_RAG": "true",
                 "ENABLE_WEB_SEARCH": "true", "CRAG_ENABLED": "true", "AUTO_REVIEW": "false",
                 "CODE_INTENT_SEMANTIC": "false", "SOURCE_RELEVANCE_GATE": "true",
                 "ENABLE_AGENTIC_ANSWER_LOOP": "false", "AGENTIC_INDEPENDENT_VERIFY": "false"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(cl, "_deep_queries", lambda q: [q])
    monkeypatch.setattr(cl, "_gather_local_items", lambda q, mode: (local_items, []))
    monkeypatch.setattr(cl, "_gather_external_items", lambda q, k: ([], []))
    monkeypatch.setattr("backend.answering.query_refine.refine_query", lambda q: q)
    return mem, sid


def _drive(sid, question):
    events = []
    for ev in cl.stream_chat_events(sid, question):
        events.append(ev)
        if ev["type"] in ("done", "error", "sanity"):
            break
    done = events[-1]
    sources_events = [e for e in events if e["type"] == "sources"]
    last_sources = sources_events[-1]["sources"] if sources_events else []
    return done, last_sources, events


# A RESEARCH question (not a self-contained calculation), so it flows through retrieval + the relevance
# gate. (A bare calculation is answered directly by reasoning and never reaches retrieval — see
# test_simple_reasoning.py — so it would not exercise the gate.)
_AUDIO_Q = "What is the standard formula for uncompressed PCM audio storage size?"


def test_irrelevant_sources_are_discarded_and_answered_from_reasoning(tmp_path, monkeypatch):
    # An irrelevant (topically-audio) paper is retrieved for an audio-storage CALCULATION. The gate
    # judges it irrelevant -> it is discarded and the question is answered from reasoning, no citation.
    irrelevant = [{"source_type": "local_pdf", "title": "Audio Reasoning Benchmark", "section": "Intro",
                   "text": "a benchmark for audio question answering", "score": 0.5,
                   "page_start": 1, "page_end": 2}]
    mem, sid = _setup(monkeypatch, tmp_path, local_items=irrelevant)
    monkeypatch.setattr(cl, "get_provider", lambda: _Provider(relevant_json='{"relevant": []}'))

    done, last_sources, _ = _drive(sid, _AUDIO_Q)
    assert done["type"] == "done"
    assert "30.3" in done["answer"]                              # answered from REASONING
    assert "[1]" not in done["answer"]                          # no spurious citation
    assert last_sources == []                                   # the irrelevant source is not shown


def test_genuinely_relevant_source_is_kept_and_cited(tmp_path, monkeypatch):
    relevant = [{"source_type": "local_pdf", "title": "Digital Audio Storage Formula", "section": "PCM",
                 "text": "uncompressed size = sample_rate * bit_depth * channels * seconds / 8",
                 "score": 0.7, "page_start": 3, "page_end": 4}]
    mem, sid = _setup(monkeypatch, tmp_path, local_items=relevant)
    monkeypatch.setattr(cl, "get_provider", lambda: _Provider(
        relevant_json='{"relevant": [1]}',
        evidence_answer="Storage = sample_rate x bit_depth x channels x seconds / 8 [1]. ~30.3 MB."))

    done, last_sources, _ = _drive(sid, _AUDIO_Q)
    assert done["type"] == "done"
    assert "[1]" in done["answer"]                              # the supporting source IS cited
    assert len(last_sources) == 1 and last_sources[0]["title"] == "Digital Audio Storage Formula"


def test_mixed_keeps_relevant_drops_irrelevant(tmp_path, monkeypatch):
    mixed = [
        {"source_type": "local_pdf", "title": "Digital Audio Storage Formula", "section": "PCM",
         "text": "uncompressed size = sample_rate * bit_depth * channels * seconds / 8",
         "score": 0.6, "page_start": 3, "page_end": 4},
        {"source_type": "local_pdf", "title": "Audio Reasoning Benchmark", "section": "Intro",
         "text": "a benchmark for audio question answering", "score": 0.55,
         "page_start": 1, "page_end": 2},
    ]
    mem, sid = _setup(monkeypatch, tmp_path, local_items=mixed)
    monkeypatch.setattr(cl, "get_provider", lambda: _Provider(
        relevant_json='{"relevant": [1]}',                      # only the formula paper is relevant
        evidence_answer="Storage = sample_rate x bit_depth x channels x seconds / 8 [1]. ~30.3 MB."))

    done, last_sources, events = _drive(sid, _AUDIO_Q)
    assert done["type"] == "done"
    titles = [s["title"] for s in last_sources]
    assert "Digital Audio Storage Formula" in titles            # relevant kept + cited
    assert "Audio Reasoning Benchmark" not in titles            # irrelevant dropped (not cited)
    assert any(e["type"] == "status" and "Set aside" in e.get("message", "") for e in events)


# ==============================================================================================
# Prompt contract: the draft is told to cite only supporting sources and not bend to irrelevant ones.
# ==============================================================================================
def test_system_prompt_requires_cite_only_supporting_sources():
    s = cl.SYSTEM_PROMPT.lower()
    assert "use only sources that directly address" in s
    assert "do not bend the answer" in s
