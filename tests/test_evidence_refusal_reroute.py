"""Root-cause regression: a SELF-CONTAINED question (e.g. an audio-storage calculation) must not be
refused with 'the provided evidence does not contain ...' just because the topical corpus returned
IRRELEVANT documents. The system detects an evidence-insufficient non-answer and re-answers from
reasoning. General (not tied to the audio question).
"""
import webapp.chat_logic as cl
from backend.memory.store import MemoryStore


# The EXACT failure from the bug report (8 irrelevant audio sources, refused) — must be detected.
_SCREENSHOT_REFUSAL = (
    "The provided evidence does not contain information on how to calculate the storage needed for an "
    "uncompressed audio file based on its sample rate, bit depth, and duration. The sources discuss "
    "topics such as audio reasoning benchmarks [1, 2], audio inpainting [3], video temporal reasoning "
    "[4], audio quality assessment [5], audio-language model alignment [6], and machine learning model "
    "properties [7, 8]."
)


def test_screenshot_refusal_is_detected():
    assert cl._is_evidence_refusal(_SCREENSHOT_REFUSAL)


def test_various_evidence_refusal_phrasings_detected():
    for txt in (
        "The sources do not cover this topic.",
        "The retrieved evidence does not address the question; the documents discuss other matters.",
        "The available sources don't contain the requested information.",
        "Unfortunately the provided context does not include the details needed to answer this.",
    ):
        assert cl._is_evidence_refusal(txt), txt


def test_real_answer_that_leads_with_the_answer_is_not_a_refusal():
    # A correct, reasoned answer LEADS with the answer; a brief later caveat must not trip the detector.
    good = ("Approximately 30.3 MB. Storage = sample_rate x bit_depth x channels x seconds / 8. "
            "= 44100 x 16 x 2 x 180 / 8 bytes = 31,752,000 bytes ~= 30.3 MB (or 31.75 MB decimal). "
            "(The library sources don't cover this, but it is standard arithmetic.)")
    assert not cl._is_evidence_refusal(good)
    assert not cl._is_evidence_refusal("Paris is the capital of France. It has about 2.1M residents.")
    assert not cl._is_evidence_refusal("")


def test_self_contained_reasoning_is_allowed_in_the_answer_prompt():
    s = cl.SYSTEM_PROMPT.lower()
    # The prompt must permit answering a self-contained / standard question from reasoning instead of
    # replying that the sources don't cover it.
    assert "self-contained" in s or "your own reasoning" in s
    assert "calculation" in s


# ----- end-to-end: the exact bug — irrelevant audio docs retrieved, draft refuses -> re-answer -----
class _RerouteProvider:
    is_available = True
    model = "fake"

    def stream_chat(self, messages, system="", **k):
        s = (system or "").lower()
        if "independent checker" in s:                           # the INDEPENDENT confirmation pass
            return ['{"agrees": true, "confidence": 90}']
        if "answer-quality judge" in s:                          # reasoning verify
            return ['{"ok": true, "score": 95}']
        if "evidence verifier" in s:                             # evidence verify (the refusal is "grounded")
            return ['{"ok": true, "score": 85}']
        if "own knowledge and step-by-step reasoning" in s:      # the REASONING draft
            return ["Approximately 30.3 MB. 44100*16*2*180/8 = 31,752,000 bytes, about 30.3 MB. " * 2]
        if "meticulous, broad-domain research assistant" in s:   # the EVIDENCE draft -> refuses
            return ["The provided evidence does not contain information on how to calculate this. "
                    "The sources discuss audio benchmarks [1]."]
        return [messages[-1]["content"] if messages else ""]

    def unavailable_message(self):
        return "n/a"


def test_irrelevant_evidence_reroutes_a_self_contained_question_to_reasoning(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    monkeypatch.setattr(cl, "_memory", mem)
    for k, v in {"ENABLE_ANSWER_CACHE": "false", "ENABLE_LOCAL_RAG": "true",
                 "ENABLE_WEB_SEARCH": "true", "CRAG_ENABLED": "true", "AUTO_REVIEW": "false",
                 "CODE_INTENT_SEMANTIC": "false"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(cl, "_deep_queries", lambda q: [q])
    # Retrieval returns a tangentially-related (irrelevant) audio doc -> CRAG PARTIAL -> evidence path.
    monkeypatch.setattr(cl, "_gather_local_items", lambda q, mode: (
        [{"source_type": "local_pdf", "title": "Audio Benchmark Paper", "section": "Intro",
          "text": "an audio reasoning benchmark", "score": 0.5, "page_start": 1, "page_end": 2}], []))
    monkeypatch.setattr(cl, "_gather_external_items", lambda q, k: ([], []))
    monkeypatch.setattr("backend.answering.query_refine.refine_query", lambda q: q)
    monkeypatch.setattr(cl, "get_provider", lambda: _RerouteProvider())

    done = None
    for ev in cl.stream_chat_events(
            sid, "How much storage for 3 minutes of 44.1 kHz 16-bit stereo audio, in MB?"):
        if ev["type"] in ("done", "error", "sanity"):
            done = ev
            break
    assert done and done["type"] == "done"
    assert "30.3" in done["answer"]                              # answered from REASONING
    assert "evidence does not contain" not in done["answer"].lower()   # the refusal was NOT shipped
