"""ACQUIRED KNOWLEDGE — the agent GROWS its own RAG corpus from verified findings (Phase 2).

When a VERIFIED answer cites external findings (web / paper / patent / repo), those passages are embedded
and stored; on a future SIMILAR question they are recalled into retrieval as evidence — so the agent
answers from a corpus that grew out of its own verified research, without re-fetching. Capture runs in
the BACKGROUND (zero added latency); recall reuses the already-computed query embedding (no extra call).

Proves: (a) store record→recall round-trip (semantic via matching embedding; lexical fallback);
(b) re-capturing the same finding UPSERTS and STRENGTHENS it (confidence up, not reset); (c) the table is
pruned/bounded; (d) existing_source_hashes + reinforce; (e) only CITED external findings are selected
(local + too-short skipped, deduped, full text taken from items); (f) capture is gated + verified-only +
skips re-embedding seen findings; (g) recall yields well-formed evidence items, gated off → nothing;
(h) the background dispatcher runs inline in sync mode, swallows errors, and flush is safe;
(i) end-to-end capture→recall grows the corpus and recalls it.
"""
import pytest

from backend.answering import acquired_knowledge as ak
from backend.answering import background as bg
from backend.memory.store import MemoryStore


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setenv("CORPUS_GROWTH", "true")         # conftest disables it suite-wide
    monkeypatch.setenv("LEARN_BACKGROUND_SYNC", "1")    # capture runs inline -> deterministic tests


def _mem(tmp_path):
    return MemoryStore(tmp_path / "m.db")


# ---- (a) store record -> recall round-trip: semantic via matching embedding, lexical fallback ----
def test_record_and_recall_learned_source_semantic(tmp_path):
    m = _mem(tmp_path)
    m.record_learned_source(user_id="u", content_hash="h1", text="Beamforming steers a mic array. " * 8,
                            title="Beamforming basics", url="http://x/bf", snippet="Beamforming steers...",
                            embedding=[1.0, 0.0, 0.0], embedding_meta="emb1")
    # different WORDS, but a near-identical passage embedding + same meta -> recalled semantically
    got = m.recall_learned_sources(user_id="u", question="zzz qqq", query_embedding=[0.98, 0.02, 0.0],
                                   query_meta="emb1")
    assert len(got) == 1 and got[0]["content_hash"] == "h1"
    # a different embedding provider tag is NOT comparable -> falls back to (failing) lexical -> nothing
    assert m.recall_learned_sources(user_id="u", question="zzz qqq", query_embedding=[0.98, 0.02, 0.0],
                                    query_meta="other") == []


def test_title_match_with_offtopic_body_is_not_recalled(tmp_path):
    # POISONING GUARD: a passage whose TITLE matches the question but whose BODY (what we actually inject)
    # is off-topic must NOT clear the relevance floor — relevance is scored on the body, not the title.
    m = _mem(tmp_path)
    m.record_learned_source(
        user_id="u", content_hash="poison", title="neural network training",
        snippet="A refrigerator compressor valve assembly for an HVAC patent filing, unrelated to ML.",
        text="A refrigerator compressor valve assembly for an HVAC patent filing, unrelated to ML. " * 4)
    # lexical-only (no embeddings): the title would have matched, but the body does not
    assert m.recall_learned_sources(user_id="u", question="how to do neural network training") == []


def test_recall_learned_source_lexical_fallback(tmp_path):
    m = _mem(tmp_path)
    m.record_learned_source(user_id="u", content_hash="h2",
                            text="The Opus codec compresses speech at low bitrate with high quality. " * 6,
                            title="Opus codec speech compression bitrate quality",
                            snippet="Opus codec compresses speech at low bitrate")
    # no embeddings anywhere -> lexical match of the question against title+snippet
    hits = m.recall_learned_sources(user_id="u",
                                    question="opus codec speech compression bitrate quality",
                                    min_relevance=0.4)
    assert hits and hits[0]["content_hash"] == "h2"


# ---- (b) re-capture UPSERTs and STRENGTHENS (confidence rises, single row) ----
def test_recapture_upserts_and_strengthens(tmp_path):
    m = _mem(tmp_path)
    first = m.record_learned_source(user_id="u", content_hash="dup", text="passage body here " * 6,
                                    title="t", embedding=[1.0, 0.0, 0.0], embedding_meta="emb1")
    second = m.record_learned_source(user_id="u", content_hash="dup", text="passage body here " * 6,
                                     title="t")  # cited again, no new embedding supplied
    assert first == second                              # same row, upgraded in place
    got = m.recall_learned_sources(user_id="u", question="passage body here",
                                   query_embedding=[1.0, 0.0, 0.0], query_meta="emb1", min_relevance=0.0)
    assert len(got) == 1
    assert got[0]["confidence"] > 1.0                   # strengthened, not reset to 1.0
    assert got[0]["hit_count"] >= 1
    # the original embedding is KEPT even though the re-capture supplied none (COALESCE) -> still semantic
    assert got[0]["relevance"] >= 0.99


# ---- (c) the grown corpus stays bounded (weakest evicted first) ----
def test_prune_keeps_corpus_bounded(tmp_path):
    m = _mem(tmp_path)
    for i in range(8):
        m.record_learned_source(user_id="u", content_hash=f"h{i}", text=f"finding number {i} body text",
                                title=f"t{i}")
    assert m.prune_learned_sources(user_id="u", max_per_user=5) >= 0
    m.prune_learned_sources(user_id="u", max_per_user=5)
    with m._conn() as conn:
        n = conn.execute("SELECT COUNT(*) AS c FROM learned_sources WHERE user_id = 'u'").fetchone()["c"]
    assert n <= 5


# ---- (d) existing_source_hashes + direct reinforcement ----
def test_existing_source_hashes_and_reinforce(tmp_path):
    m = _mem(tmp_path)
    sid = m.record_learned_source(user_id="u", content_hash="known", text="some stored body text here",
                                  title="t")
    assert m.existing_source_hashes(user_id="u", hashes=["known", "missing"]) == {"known"}
    assert m.existing_source_hashes(user_id="u", hashes=[]) == set()
    m.reinforce_learned_sources([sid])
    got = m.recall_learned_sources(user_id="u", question="some stored body text here", min_relevance=0.3)
    assert got and got[0]["confidence"] > 1.0 and got[0]["hit_count"] >= 1


# ---- (e) only CITED external findings are selected (local + short skipped, deduped, full text) ----
def test_select_cited_findings(tmp_path):
    items = [
        {"source_type": "web", "title": "A", "url": "http://a", "text": "X" * 200},          # n=1 keep
        {"source_type": "local_pdf", "title": "L", "url": "", "text": "Y" * 200},            # n=2 skip(local)
        {"source_type": "research_paper", "title": "P", "url": "http://p", "text": "Z" * 200},  # n=3 keep
        {"source_type": "web", "title": "S", "url": "http://s", "text": "tiny"},             # n=4 skip(short)
    ]
    cited = [
        {"n": 1, "source_type": "web", "title": "A", "url": "http://a", "text": "X" * 40},
        {"n": 2, "source_type": "local_pdf", "title": "L", "url": "", "text": "Y" * 40},
        {"n": 3, "source_type": "research_paper", "title": "P", "url": "http://p", "text": "Z" * 40},
        {"n": 4, "source_type": "web", "title": "S", "url": "http://s", "text": "tiny"},
    ]
    findings = ak._select_cited_findings(items, cited)
    kinds = {(f["source_type"], f["url"]) for f in findings}
    assert kinds == {("web", "http://a"), ("research_paper", "http://p")}
    # full text was taken from `items` (200 chars), not the 40-char cited preview
    assert all(len(f["text"]) >= 200 for f in findings)
    # duplicate cited source (same url+text) is collapsed by content hash
    dup = ak._select_cited_findings(items, cited + [dict(cited[0])])
    assert len([f for f in dup if f["url"] == "http://a"]) == 1


def test_content_hash_stable_and_url_text_sensitive():
    a = ak.content_hash("http://x/", "hello world body")
    assert a == ak.content_hash("http://x", "hello world body")   # trailing slash normalized
    assert a != ak.content_hash("http://y", "hello world body")   # different url
    assert a != ak.content_hash("http://x/", "different body")    # different text


# ---- (f) capture is gated + verified-only + skips re-embedding seen findings ----
def _fake_embedder(calls):
    def _embed(texts):
        if not texts:
            return [], None
        calls.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts], "embX"
    return _embed


def test_capture_findings_stores_only_when_enabled_and_verified(tmp_path, monkeypatch):
    m = _mem(tmp_path)
    calls = []
    monkeypatch.setattr(ak, "_embed_passages", _fake_embedder(calls))
    items = [{"source_type": "web", "title": "A", "url": "http://a", "text": "evidence body " * 20}]
    cited = [{"n": 1, "source_type": "web", "title": "A", "url": "http://a", "text": "evidence body"}]

    ak.capture_findings(m, user_id="u", question="q", items=items, cited_sources=cited, verified=False)
    assert m.recall_learned_sources(user_id="u", question="evidence body", min_relevance=0.0) == []  # unverified

    monkeypatch.setenv("CORPUS_GROWTH", "false")
    ak.capture_findings(m, user_id="u", question="q", items=items, cited_sources=cited, verified=True)
    assert m.recall_learned_sources(user_id="u", question="evidence body", min_relevance=0.0) == []  # gated off

    monkeypatch.setenv("CORPUS_GROWTH", "true")
    ak.capture_findings(m, user_id="u", question="q", items=items, cited_sources=cited, verified=True)
    stored = m.recall_learned_sources(user_id="u", question="evidence body",
                                      query_embedding=[1.0, 0.0, 0.0], query_meta="embX", min_relevance=0.0)
    assert len(stored) == 1 and stored[0]["url"] == "http://a"
    assert calls == [[("evidence body " * 20).strip()]]  # embedded exactly once (text is stripped)

    # re-capturing the SAME finding does NOT re-embed (already stored) but still strengthens it
    ak.capture_findings(m, user_id="u", question="q2", items=items, cited_sources=cited, verified=True)
    assert calls == [[("evidence body " * 20).strip()]]  # no second embed call
    again = m.recall_learned_sources(user_id="u", question="evidence body",
                                     query_embedding=[1.0, 0.0, 0.0], query_meta="embX", min_relevance=0.0)
    assert again[0]["confidence"] > 1.0


# ---- (g) recall_items yields well-formed evidence items; gated off -> nothing ----
def test_recall_items_shape_and_gating(tmp_path, monkeypatch):
    m = _mem(tmp_path)
    m.record_learned_source(user_id="u", content_hash="h", text="learned passage about codecs " * 5,
                            title="Codec passage", url="http://c", source_type="research_paper",
                            embedding=[1.0, 0.0, 0.0], embedding_meta="embX")
    items = ak.recall_items(m, user_id="u", question="anything", query_embedding=[1.0, 0.0, 0.0],
                            query_meta="embX")
    assert len(items) == 1
    it = items[0]
    assert it["source_type"] == "research_paper" and it["url"] == "http://c"
    assert it["text"].startswith("learned passage about codecs")
    assert 0.0 <= it["rerank_score"] <= 1.0 and it["score"] == it["rerank_score"]
    assert it["retrieval_sources"] == ["learned"]
    assert "_learned_id" not in it                       # no private field leaks into the source list

    monkeypatch.setenv("CORPUS_GROWTH", "false")
    assert ak.recall_items(m, user_id="u", question="anything", query_embedding=[1.0, 0.0, 0.0],
                           query_meta="embX") == []


# ---- (h) the background dispatcher: sync runs inline, errors are swallowed, flush is safe ----
def test_background_sync_runs_inline_and_swallows_errors(monkeypatch):
    monkeypatch.setenv("LEARN_BACKGROUND_SYNC", "1")
    seen = []
    bg.run(lambda x: seen.append(x), 7)
    assert seen == [7]                                  # ran inline

    def _boom():
        raise RuntimeError("nope")
    bg.run(_boom)                                       # must not raise
    bg.flush(timeout=1.0)                               # must not raise


# ---- (i) end-to-end: a verified answer GROWS the corpus, and it is RECALLED next time ----
def test_capture_then_recall_grows_corpus(tmp_path, monkeypatch):
    m = _mem(tmp_path)
    monkeypatch.setattr(ak, "_embed_passages", _fake_embedder([]))
    items = [{"source_type": "web", "title": "Latency in audio", "url": "http://lat",
              "text": "End-to-end audio latency budgets and how to measure them. " * 12}]
    cited = [{"n": 1, "source_type": "web", "title": "Latency in audio", "url": "http://lat",
              "text": "End-to-end audio latency budgets"}]
    ak.capture_findings(m, user_id="local", question="what is audio latency budget",
                        items=items, cited_sources=cited, verified=True)

    recalled = ak.recall_items(m, user_id="local", question="how do I measure audio latency",
                               query_embedding=[1.0, 0.0, 0.0], query_meta="embX")
    assert recalled and recalled[0]["url"] == "http://lat"
    assert recalled[0]["text"].startswith("End-to-end audio latency")
