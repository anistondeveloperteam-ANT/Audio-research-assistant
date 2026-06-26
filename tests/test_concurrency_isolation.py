"""
CONCURRENCY ISOLATION — per-request settings (Fast/Deep mode, selected model, derived knobs) live in a
per-request contextvar, NOT process-global os.environ, so simultaneous requests can't clobber each other.

Proves: (a) many TRULY concurrent requests with DIFFERENT settings each execute with their OWN settings —
in the request thread AND inside nested thread-pool workers; (b) nothing per-request is written to
os.environ at request time; (c) a barrier-synchronised stress run (all requests bind different settings,
THEN race to read) stays isolated — the case a global-env design would fail; (d) the request context
preserves settings across a streamed generator's chunks (the server's bind_to_context), and a plain
(non-context) thread pool does NOT inherit — proving the propagation is what carries the settings.
"""
import concurrent.futures
import contextvars
import os
import threading

import pytest

import webapp.chat_logic as CL
from backend.answering import agentic_answer as AA
from backend.answering.research_modes import resolve_research_mode
from backend.llm.streaming_provider import get_provider, resolve_model_settings
from backend.common import request_context as rc

# Keep mode knobs unset as env so the profile (and only the profile) drives the values.
_KEYS = ["EXTERNAL_TOP_K", "ANSWER_MAX_TOKENS", "DEEP_SEARCH_SUBQUERIES", "AGENTIC_MAX_VERIFY_ROUNDS",
         "AUTO_REVIEW", "DEEP_MAX_LOOPS", "EXTERNAL_GATHER_TIMEOUT", "OPENAI_MODEL", "OPENAI_BASE_URL",
         "OPENAI_API_KEY"]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    rc.clear_request_settings()
    yield
    rc.clear_request_settings()


def _expected(mode):
    return {"external_top_k": 8 if mode == "fast" else 20,
            "answer_max_tokens": 3000 if mode == "fast" else 8000,
            "verify_rounds": 1 if mode == "fast" else 3,
            "auto_review": mode == "deep"}


def _observe(mode, model, barrier=None):
    """Simulate ONE request: bind its settings, optionally sync with peers to force interleaving, then
    read the live getters in this thread AND in a nested context-propagating worker."""
    rc.set_request_settings({**resolve_research_mode(mode), **resolve_model_settings(model)})
    if barrier is not None:
        barrier.wait()                              # everyone has bound DIFFERENT settings; now race to read
    main = {
        "external_top_k": CL._external_top_k(),
        "answer_max_tokens": CL._answer_max_tokens(),
        "verify_rounds": AA.max_verify_rounds(),
        "auto_review": AA.auto_review_enabled(),
        "model": get_provider().model,
    }
    with rc.ContextThreadPoolExecutor(max_workers=2) as ex:
        nested = ex.submit(lambda: (CL._external_top_k(), get_provider().model)).result()
    return main, nested


# ---- (a)+(c) many concurrent, barrier-synchronised, mixed-mode + mixed-model requests stay isolated ----
@pytest.mark.parametrize("n", [12, 24])
def test_concurrent_mixed_requests_keep_their_own_settings(n):
    specs = [(("fast", f"fast-model-{i}") if i % 2 == 0 else ("deep", f"deep-model-{i}")) for i in range(n)]
    barrier = threading.Barrier(n)
    results = [None] * n

    def run(i):
        results[i] = _observe(specs[i][0], specs[i][1], barrier)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    for i, (mode, model) in enumerate(specs):
        assert results[i] is not None, f"request {i} did not finish"
        main, nested = results[i]
        exp = _expected(mode)
        assert main["external_top_k"] == exp["external_top_k"], (i, mode, main)
        assert main["answer_max_tokens"] == exp["answer_max_tokens"], (i, mode, main)
        assert main["verify_rounds"] == exp["verify_rounds"], (i, mode, main)
        assert main["auto_review"] is exp["auto_review"], (i, mode, main)
        assert main["model"] == model, (i, main)
        # the nested thread-pool worker saw THIS request's settings, not a sibling's
        assert nested == (exp["external_top_k"], model), (i, nested)


# ---- (b) no per-request value is written to os.environ at request time ----
def test_no_per_request_write_to_os_environ():
    get_provider()                                   # warm-up: trigger get_provider's one-time load_dotenv
    before = dict(os.environ)                         # ...so the snapshot already reflects static .env config
    barrier = threading.Barrier(6)
    threads = [threading.Thread(target=_observe,
                                args=(("fast" if i % 2 else "deep"), f"m{i}", barrier)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert dict(os.environ) == before               # the whole concurrent run mutated nothing in os.environ


# ---- (d) context propagation: streamed chunks keep settings; a plain pool does NOT inherit ----
def test_bind_to_context_preserves_settings_across_streamed_chunks():
    rc.set_request_settings(resolve_research_mode("deep"))     # external_top_k = 20

    def gen():
        for _ in range(5):
            yield CL._external_top_k()

    wrapped = rc.bind_to_context(gen())               # snapshots the bound (deep) context
    rc.clear_request_settings()                       # outer context loses it -> only the snapshot has it
    seen = []
    while True:                                       # iterate like Starlette: each next in a FRESH outer ctx
        try:
            seen.append(contextvars.copy_context().run(next, wrapped))
        except StopIteration:
            break
    assert seen == [20, 20, 20, 20, 20]               # every streamed chunk still saw the deep setting


def test_context_pool_inherits_but_plain_pool_does_not():
    rc.set_request_settings(resolve_research_mode("fast"))     # external_top_k = 8
    with rc.ContextThreadPoolExecutor(max_workers=3) as ex:
        inherited = list(ex.map(lambda _: CL._external_top_k(), range(6)))
    assert inherited == [8, 8, 8, 8, 8, 8]            # workers inherited the request's FAST setting

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        plain = ex.submit(CL._external_top_k).result()
    assert plain == 20                                # a PLAIN worker has no context -> env/default (20)


# ---- hardening: binding a request's settings must NOT pollute the (reused) endpoint thread ----
def test_bound_stream_does_not_pollute_caller_context():
    assert not rc.has_request_settings()              # caller (endpoint thread) starts clean

    def gen():
        yield CL._external_top_k()

    wrapped = rc.bound_stream(resolve_research_mode("deep"), gen())
    assert not rc.has_request_settings()              # binding happened in the stream's OWN context only
    seen = []
    while True:                                        # iterate like Starlette (fresh outer ctx per chunk)
        try:
            seen.append(contextvars.copy_context().run(next, wrapped))
        except StopIteration:
            break
    assert seen == [20]                                # the stream still saw its deep setting
    assert not rc.has_request_settings()              # ...and the caller thread is STILL clean


def test_run_with_settings_does_not_pollute_caller_context():
    assert not rc.has_request_settings()
    seen = {}

    def worker():
        seen["model"] = get_provider().model

    runner = rc.run_with_settings(resolve_model_settings("worker-model"), worker)
    assert not rc.has_request_settings()              # capturing the runner did not pollute the caller
    t = threading.Thread(target=runner)
    t.start()
    t.join(timeout=10)
    assert seen["model"] == "worker-model"            # the worker thread saw the bound model
    assert not rc.has_request_settings()


# ---- model isolation in particular: concurrent get_provider() calls never cross models ----
def test_concurrent_get_provider_models_do_not_cross():
    n = 16
    models = [f"model-{i}" for i in range(n)]
    barrier = threading.Barrier(n)
    seen = [None] * n

    def run(i):
        rc.set_request_settings(resolve_model_settings(models[i]))
        barrier.wait()
        seen[i] = get_provider().model

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert seen == models                             # each request kept its own model under contention
