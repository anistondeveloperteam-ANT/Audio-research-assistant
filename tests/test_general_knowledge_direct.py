"""GENERAL-KNOWLEDGE / REASONING questions are answered DIRECTLY from the model's own knowledge — never
forced through the corpus/evidence frame, never refused for lack of corpus.

Two coupled bugs this fixes: a general-knowledge question routed to the corpus/evidence path was either
(1) REFUSED ("couldn't produce a confident answer" / "not in the sources") or (2) FRAMED around evidence
("according to the provided sources", a "Limitations of the Evidence" section, citations to irrelevant
corpus docs). Root cause: that path assumed every answer must come from the corpus.

The decisive, topic-agnostic signal: an evidence-path draft that cites NO source = the model answered
from its OWN knowledge -> hand off to the clean reasoning path (no citations, no evidence-framing, no
irrelevant sources, origin-independent confidence, never refused). A genuine corpus answer cites >=1
source and is left untouched.

Proves (a)-(e). Deterministic + offline: provider / memory / retrieval mocked; the source router and
relevance gate are disabled suite-wide (conftest), so these questions deterministically take the
corpus/evidence path — exactly the path that used to misbehave.
"""
import re

import webapp.chat_logic as cl
from backend.answering import agentic_answer as aa
from backend.memory.store import MemoryStore

# Evidence-frame phrasings that must NOT appear in a clean general-knowledge answer.
_FRAME_MARKERS = (
    "according to the provided sources", "according to the sources", "limitations of the evidence",
    "the provided sources", "the available sources", "the sources provided", "based on the provided",
    "the evidence does not", "the sources do not", "the provided documents", "the retrieved",
)
_REFUSAL_MARKERS = (
    "couldn't produce a confident answer", "not in the sources", "does not contain",
    "do not contain", "couldn't find",
)


def _has_citation(text: str) -> bool:
    return bool(re.search(r"\[\d+\]", text or ""))


def _drive(sid, q):
    """Run the pipeline, returning (done_event, last_sources_payload)."""
    last_sources = None
    for ev in cl.stream_chat_events(sid, q):
        if ev["type"] == "sources":
            last_sources = ev["sources"]
        if ev["type"] in ("done", "error", "sanity"):
            return ev, last_sources
    return None, last_sources


class _KnowledgeProvider:
    """A general-knowledge question: the EVIDENCE draft answers cleanly and CITES NOTHING (uses no
    source). The evidence verifier would FAIL it for lack of grounding — which is exactly the bug the
    handoff sidesteps by routing to the origin-independent reasoning judge."""
    is_available = True
    model = "fake"
    _CLEAN = ("This is a clear, correct, direct answer drawn entirely from well-established general "
              "knowledge; no document is needed to state it, and none is referenced. " * 2)

    def stream_chat(self, messages, system="", **k):
        s = (system or "").lower()
        if "independent checker" in s:
            return ['{"agrees": true, "confidence": 90}']
        if "answer-quality judge" in s:                          # reasoning verify (origin-independent)
            return ['{"ok": true, "score": 96}']
        if "evidence verifier" in s:                             # would reject for lack of grounding
            return ['{"ok": false, "score": 40, "citation_issues": ["no citations"]}']
        if "own knowledge and step-by-step reasoning" in s:      # the REASONING draft (clean)
            return [self._CLEAN]
        if "meticulous, broad-domain research assistant" in s:   # the EVIDENCE draft (clean, NO [n])
            return [self._CLEAN]
        return [messages[-1]["content"] if messages else ""]

    def unavailable_message(self):
        return "n/a"


class _CorpusProvider:
    """A genuine corpus-subject question: the EVIDENCE draft CITES [1] (the indexed source is really
    used) -> no handoff, citations and sources kept."""
    is_available = True
    model = "fake"

    def stream_chat(self, messages, system="", **k):
        s = (system or "").lower()
        if "independent checker" in s:
            return ['{"agrees": true, "confidence": 90}']
        if "answer-quality judge" in s:
            return ['{"ok": true, "score": 95}']
        if "evidence verifier" in s:
            return ['{"ok": true, "score": 92}']
        if "own knowledge and step-by-step reasoning" in s:
            return ["A reasoned answer."]
        if "meticulous, broad-domain research assistant" in s:   # EVIDENCE draft -> genuinely cites [1]
            return ["The indexed method applies spectral masking to suppress noise [1]. " * 2]
        return [messages[-1]["content"] if messages else ""]

    def unavailable_message(self):
        return "n/a"


# An IRRELEVANT corpus hit (CRAG keeps it -> the general-knowledge Q lands on the evidence path), and a
# RELEVANT one (a real corpus-subject question).
_IRRELEVANT = [{"source_type": "local_pdf", "title": "Audio Benchmark", "section": "Intro",
                "text": "an audio reasoning benchmark", "score": 0.5, "page_start": 1, "page_end": 2}]
_RELEVANT = [{"source_type": "local_pdf", "title": "Spectral Masking for Speech Enhancement",
              "section": "Method", "text": "we apply spectral masking to suppress noise",
              "score": 0.85, "page_start": 3, "page_end": 4}]


def _corpus_env(monkeypatch, mem, provider, *, local_items):
    monkeypatch.setattr(cl, "_memory", mem)
    for k, v in {"ENABLE_ANSWER_CACHE": "false", "ENABLE_LOCAL_RAG": "true", "ENABLE_WEB_SEARCH": "true",
                 "CRAG_ENABLED": "true", "AUTO_REVIEW": "false", "CODE_INTENT_SEMANTIC": "false",
                 "ENABLE_AGENTIC_ANSWER_LOOP": "true", "AGENTIC_INDEPENDENT_VERIFY": "false"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(cl, "_deep_queries", lambda q: [q])
    monkeypatch.setattr(cl, "_gather_local_items", lambda q, mode: (list(local_items), []))
    monkeypatch.setattr(cl, "_gather_external_items", lambda q, k: ([], []))
    monkeypatch.setattr("backend.answering.query_refine.refine_query", lambda q: q)
    monkeypatch.setattr(cl, "get_provider", lambda *a, **k: provider)


# ---- (a) general knowledge -> clean direct answer, NO citations, NO evidence-framing, NO sources -----
def test_a_general_knowledge_answered_directly_no_citations_no_framing(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _corpus_env(monkeypatch, mem, _KnowledgeProvider(), local_items=_IRRELEVANT)
    done, srcs = _drive(sid, "What is the boiling point of water at sea level?")
    assert done and done["type"] == "done"
    ans = done["answer"]
    assert not _has_citation(ans)                                # NO corpus citations
    low = ans.lower()
    assert not any(m in low for m in _FRAME_MARKERS)             # NO "according to the sources" / limitations
    assert (srcs or []) == []                                    # NO irrelevant sources shown


# ---- (b) NEVER refused for lack of corpus ------------------------------------------------------------
def test_b_answerable_question_is_never_refused_for_lack_of_corpus(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _corpus_env(monkeypatch, mem, _KnowledgeProvider(), local_items=_IRRELEVANT)
    done, _ = _drive(sid, "Who wrote the play Romeo and Juliet?")
    assert done and done["type"] == "done"
    low = done["answer"].lower()
    assert not any(m in low for m in _REFUSAL_MARKERS)


# ---- (c) confidence / answerability decoupled from corpus presence -----------------------------------
def test_c_confidence_does_not_depend_on_corpus_presence(tmp_path, monkeypatch):
    # The reasoning verifier judges on MERITS, not citations...
    assert "do not require citations" in aa._VERIFY_REASONING_SYSTEM.lower()
    # ...so a general-knowledge answer is delivered AND cached VERIFIED even though no corpus backed it
    # (the evidence verifier returned ok=false/40 — which must NOT gate it).
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _corpus_env(monkeypatch, mem, _KnowledgeProvider(), local_items=_IRRELEVANT)
    monkeypatch.setenv("ENABLE_ANSWER_CACHE", "true")
    q = "What is the boiling point of water at sea level?"
    done, _ = _drive(sid, q)
    assert done and done["type"] == "done"
    assert mem.find_cached_answer(user_id="local", question=q) is not None   # delivered + verified


# ---- (d) citations / evidence-framing ONLY when sources genuinely address the question ---------------
def test_d_citations_only_when_sources_genuinely_used(tmp_path, monkeypatch):
    # general knowledge -> NO citations, NO sources
    mem1 = MemoryStore(tmp_path / "g.db")
    s1 = mem1.create_session(user_id="local")
    _corpus_env(monkeypatch, mem1, _KnowledgeProvider(), local_items=_IRRELEVANT)
    gk, gk_srcs = _drive(s1, "What is the capital of France?")
    assert not _has_citation(gk["answer"]) and (gk_srcs or []) == []
    # a genuine corpus-subject question -> citations AND sources
    mem2 = MemoryStore(tmp_path / "c.db")
    s2 = mem2.create_session(user_id="local")
    _corpus_env(monkeypatch, mem2, _CorpusProvider(), local_items=_RELEVANT)
    cp, cp_srcs = _drive(s2, "What method does the indexed paper use to suppress noise?")
    assert _has_citation(cp["answer"]) and cp_srcs and len(cp_srcs) >= 1


# ---- (e) a real corpus-subject question still retrieves and cites correctly --------------------------
def test_e_real_corpus_question_still_retrieves_and_cites(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    _corpus_env(monkeypatch, mem, _CorpusProvider(), local_items=_RELEVANT)
    done, srcs = _drive(sid, "What method does the indexed paper use to suppress noise?")
    assert done and done["type"] == "done"
    assert _has_citation(done["answer"])                         # genuine corpus answer KEEPS its citation
    assert srcs and len(srcs) >= 1                               # and shows the source it actually used
