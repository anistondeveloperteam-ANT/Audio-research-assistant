from backend.answering.agentic_answer import answer_logic_version
from backend.memory.store import MemoryStore, question_similarity
from webapp import chat_logic


def test_question_similarity_accepts_close_rephrase():
    sim = question_similarity(
        "How does MVDR beamforming reduce noise?",
        "Explain how MVDR beamforming reduces noise",
    )
    assert sim >= 0.88
    assert question_similarity(
        "How does MVDR beamforming reduce noise?",
        "What is transformer attention?",
    ) < 0.5


def test_answer_cache_is_per_user(tmp_path):
    mem = MemoryStore(tmp_path / "memory.db")
    alice = mem.create_session(user_id="alice")
    bob = mem.create_session(user_id="bob")
    mem.cache_answer(
        user_id="alice",
        session_id=alice,
        question="How does MVDR beamforming reduce noise?",
        answer="MVDR uses spatial filtering to preserve the target direction.",
        sources=[{"n": 1, "title": "Paper"}],
    )

    hit = mem.find_cached_answer(
        user_id="alice",
        question="Explain how MVDR beamforming reduces noise",
        min_similarity=0.88,
    )
    assert hit is not None
    assert hit["sources"][0]["title"] == "Paper"

    miss = mem.find_cached_answer(
        user_id="bob",
        question="Explain how MVDR beamforming reduces noise",
        min_similarity=0.88,
    )
    assert miss is None
    assert bob


def test_delete_turn_pair_removes_session_cache(tmp_path):
    mem = MemoryStore(tmp_path / "memory.db")
    sid = mem.create_session(user_id="local")
    mem.append_turn(sid, "user", "How does MVDR beamforming reduce noise?")
    mem.append_turn(sid, "assistant", "Cached answer")
    mem.cache_answer(
        user_id="local",
        session_id=sid,
        question="How does MVDR beamforming reduce noise?",
        answer="Cached answer",
        sources=[{"n": 1}],
    )

    assert mem.find_cached_answer(
        user_id="local",
        question="How does MVDR beamforming reduce noise?",
        min_similarity=1.0,
    )
    mem.delete_turn_pair(sid, 0)
    assert mem.find_cached_answer(
        user_id="local",
        question="How does MVDR beamforming reduce noise?",
        min_similarity=1.0,
    ) is None


def test_chat_stream_reuses_cached_answer_without_search(tmp_path, monkeypatch):
    mem = MemoryStore(tmp_path / "memory.db")
    sid = mem.create_session(user_id="local")
    answer = (
        "MVDR beamforming reduces noise by keeping the target direction "
        "undistorted while minimizing output power from other directions."
    )
    mem.cache_answer(
        user_id="local",
        session_id=sid,
        question="How does MVDR beamforming reduce noise?",
        answer=answer,
        sources=[{"n": 1, "title": "MVDR paper", "text": "source"}],
        logic_version=answer_logic_version(),       # a current-logic entry is eligible for reuse
    )

    monkeypatch.setattr(chat_logic, "_memory", mem)
    monkeypatch.setenv("ENABLE_ANSWER_CACHE", "true")
    monkeypatch.setenv("ANSWER_CACHE_MIN_SIMILARITY", "0.88")

    def fail_search(*args, **kwargs):
        raise AssertionError("search should not run for a cache hit")

    monkeypatch.setattr(chat_logic, "_gather_local_items", fail_search)
    monkeypatch.setattr(chat_logic, "_gather_external_items", fail_search)

    events = list(chat_logic.stream_chat_events(
        sid,
        "Explain how MVDR beamforming reduces noise",
    ))
    assert events[-1]["type"] == "done"
    assert events[-1]["cached"] is True
    assert events[-1]["answer"] == answer
    assert len(mem.get_turns(sid)) == 2


# ---- hardening: never serve a DIFFERENT question -----------------------
def test_argument_swap_is_not_reused(tmp_path):
    mem = MemoryStore(tmp_path / "m.db")
    s = mem.create_session(user_id="u")
    mem.cache_answer(user_id="u", session_id=s, question="Is Adam better than SGD?",
                     answer="Adam adapts the learning rate per parameter and is often faster.",
                     sources=[{"n": 1}])
    # Opposite comparison scores ~0.95 lexically but must NOT be reused.
    assert mem.find_cached_answer(
        user_id="u", question="Is SGD better than Adam?", min_similarity=0.80) is None


def test_identifier_change_is_not_reused(tmp_path):
    mem = MemoryStore(tmp_path / "m.db")
    s = mem.create_session(user_id="u")
    mem.cache_answer(user_id="u", session_id=s, question="what is the price of an A100 gpu",
                     answer="The A100 is a data-center GPU priced around several thousand dollars.",
                     sources=[{"n": 1}])
    assert mem.find_cached_answer(
        user_id="u", question="what is the price of an H100 gpu", min_similarity=0.50) is None


# ---- semantic matching -------------------------------------------------
def test_semantic_match_reuses_paraphrase(tmp_path):
    mem = MemoryStore(tmp_path / "m.db")
    s = mem.create_session(user_id="u")
    mem.cache_answer(user_id="u", session_id=s, question="reduce background noise in audio",
                     answer="Use spectral subtraction or a neural denoiser to suppress noise.",
                     sources=[{"n": 1}], embedding=[1.0, 0.0, 0.0], embedding_meta="google:m:3")
    # Different words, lexical bar impossibly high -> only a semantic match can win.
    hit = mem.find_cached_answer(
        user_id="u", question="speech denoising methods", min_similarity=0.999,
        query_embedding=[1.0, 0.0, 0.0], query_meta="google:m:3", min_semantic=0.9)
    assert hit is not None and hit["match_kind"] == "semantic"


def test_semantic_requires_matching_provider_meta(tmp_path):
    mem = MemoryStore(tmp_path / "m.db")
    s = mem.create_session(user_id="u")
    mem.cache_answer(user_id="u", session_id=s, question="reduce background noise in audio",
                     answer="Denoise it.", sources=[{"n": 1}],
                     embedding=[1.0, 0.0], embedding_meta="google:a:2")
    # Vectors from a different provider/model are not comparable -> no semantic hit.
    assert mem.find_cached_answer(
        user_id="u", question="speech denoising methods", min_similarity=0.999,
        query_embedding=[1.0, 0.0], query_meta="local:b:2", min_semantic=0.5) is None


# ---- per-user consistency ----------------------------------------------
def test_cache_deduped_per_user_across_sessions(tmp_path):
    mem = MemoryStore(tmp_path / "m.db")
    a = mem.create_session(user_id="u")
    b = mem.create_session(user_id="u")
    mem.cache_answer(user_id="u", session_id=a, question="What is the FFT?", answer="first", sources=[{"n": 1}])
    mem.cache_answer(user_id="u", session_id=b, question="What is the FFT?", answer="second", sources=[{"n": 1}])
    hit = mem.find_cached_answer(user_id="u", question="What is the FFT?", min_similarity=1.0)
    assert hit is not None and hit["answer"] == "second"   # one row, last write wins


def test_edit_in_another_session_invalidates_across_sessions(tmp_path):
    mem = MemoryStore(tmp_path / "m.db")
    a = mem.create_session(user_id="u")
    b = mem.create_session(user_id="u")
    mem.append_turn(b, "user", "What is the FFT?")
    mem.append_turn(b, "assistant", "ans")
    mem.cache_answer(user_id="u", session_id=a, question="What is the FFT?", answer="ans", sources=[{"n": 1}])
    mem.delete_turn_pair(b, 0)   # same user, different session
    assert mem.find_cached_answer(user_id="u", question="What is the FFT?", min_similarity=1.0) is None


# ---- chat_logic helpers ------------------------------------------------
def test_strip_answer_footers_drops_review_and_verification():
    body = "The real answer body."
    full = body + "\n\n**Auto-review:** accept (clarity 9).\n\nVerification: passed (90/100)."
    assert chat_logic._strip_answer_footers(full) == body
    assert chat_logic._strip_answer_footers(body) == body


def test_freshness_bypasses_time_sensitive_questions():
    assert chat_logic._freshness_sensitive("latest GPU benchmarks in 2026")
    assert chat_logic._freshness_sensitive("state of the art denoising")
    assert chat_logic._freshness_sensitive("as of June what changed")
    assert not chat_logic._freshness_sensitive("how does the FFT algorithm work")


def test_antonym_word_swap_is_not_reused_even_semantically(tmp_path):
    # Regression: a single content-word swap (encoder->decoder) with a near-identical
    # embedding must NOT serve the wrong answer through the semantic path.
    mem = MemoryStore(tmp_path / "m.db")
    s = mem.create_session(user_id="u")
    mem.cache_answer(
        user_id="u", session_id=s,
        question="How does the encoder transform the waveform into features?",
        answer="The encoder maps the waveform to compact latent features.",
        sources=[{"n": 1}], embedding=[1.0, 0.0, 0.0], embedding_meta="g:m:3")
    assert mem.find_cached_answer(
        user_id="u", question="How does the decoder transform the waveform into features?",
        min_similarity=0.5, query_embedding=[0.99, 0.1, 0.0], query_meta="g:m:3",
        min_semantic=0.5) is None


def test_embedding_without_meta_is_not_compared(tmp_path):
    # Regression: a vector stored without meta must never match (None == None bypass).
    mem = MemoryStore(tmp_path / "m.db")
    s = mem.create_session(user_id="u")
    mem.cache_answer(user_id="u", session_id=s, question="how to reduce echo",
                     answer="Use an adaptive filter to cancel echo.", sources=[{"n": 1}],
                     embedding=[1.0, 0.0, 0.0], embedding_meta=None)
    assert mem.find_cached_answer(
        user_id="u", question="completely different phrasing about something else",
        min_similarity=0.999, query_embedding=[1.0, 0.0, 0.0], query_meta=None,
        min_semantic=0.5) is None
