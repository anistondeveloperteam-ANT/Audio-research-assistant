"""In-process embedding (backend.ingestion.embed_chunks.embed_pending_chunks).

This is the crash fix: the web UI embeds by REUSING the warm model in the server process instead of
spawning a fresh model-loading subprocess (which OOM-crashes a small GPU with 0xC0000005 / WinError
1455). These tests mock the DB cursor + the model, so they're fast and need no Oracle/GPU.
"""
import backend.ingestion.embed_chunks as ec


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self._fetch = []
        self.updates = []

    def execute(self, sql, params=None):
        low = sql.lower()
        if "from chunks c join papers" in low:               # the pending-chunks SELECT (with header cols)
            self._fetch = list(self._rows)
        elif "update chunks set embedding" in low:
            self.updates.append((params["chunk_id"], params["embedding"]))

    def fetchall(self):
        return self._fetch

    def close(self):
        pass


class FakeConn:
    def __init__(self, cur):
        self._cur = cur
        self.commits = 0

    def cursor(self):
        return self._cur

    def commit(self):
        self.commits += 1

    def close(self):
        pass


def _wire(monkeypatch, rows, embed=None, batch=16):
    # rows are (id, chunk_text[, context[, section[, title]]]) — padded to the 5 columns the SELECT returns.
    padded = [tuple(r) + ("",) * (5 - len(r)) for r in rows]
    cur = FakeCursor(padded)
    conn = FakeConn(cur)
    monkeypatch.setattr(ec, "connect", lambda: conn)
    monkeypatch.setattr(ec, "embed_documents", embed or (lambda texts: [[0.0] * 4 for _ in texts]))
    monkeypatch.setattr(ec, "BATCH_SIZE", batch)
    return cur, conn


def test_embeds_every_null_chunk(monkeypatch):
    cur, conn = _wire(monkeypatch, [(1, "doc one", ""), (2, "doc two", "ctx")])
    res = ec.embed_pending_chunks()
    assert res == {"embedded": 2, "total": 2, "cancelled": False}
    assert [u[0] for u in cur.updates] == [1, 2]            # both chunks written back


def test_prepends_context_to_document_text(monkeypatch):
    seen = {}

    def fake_embed(texts):
        seen["texts"] = list(texts)
        return [[0.0] * 4 for _ in texts]

    _wire(monkeypatch, [(7, "body text", "situating sentence")], embed=fake_embed)
    ec.embed_pending_chunks()
    assert seen["texts"] == ["situating sentence\nbody text"]  # context prepended; raw doc


def test_embeds_with_contextual_header(monkeypatch):
    """The paper title + section are prepended to the embedded text (contextual chunk header) — the
    instant, no-LLM accuracy boost. The stored chunk_text is untouched."""
    seen = {}

    def fake_embed(texts):
        seen["texts"] = list(texts)
        return [[0.0] * 4 for _ in texts]

    # row = (id, chunk, context, section, title)
    _wire(monkeypatch, [(3, "the chunk body", "", "Methods", "MVDR Beamforming Paper")], embed=fake_embed)
    ec.embed_pending_chunks()
    assert seen["texts"] == ["MVDR Beamforming Paper | Methods\nthe chunk body"]


def test_header_omits_empty_title_and_unknown_section(monkeypatch):
    seen = {}

    def fake_embed(texts):
        seen["texts"] = list(texts)
        return [[0.0] * 4 for _ in texts]

    _wire(monkeypatch, [(4, "body", "", "Unknown", "")], embed=fake_embed)
    ec.embed_pending_chunks()
    assert seen["texts"] == ["body"]                       # no title + "Unknown" section -> no header


def test_noop_and_never_loads_model_when_nothing_pending(monkeypatch):
    called = {"n": 0}

    def fake_embed(texts):
        called["n"] += 1
        return []

    _wire(monkeypatch, [], embed=fake_embed)
    res = ec.embed_pending_chunks()
    assert res == {"embedded": 0, "total": 0, "cancelled": False}
    assert called["n"] == 0                                 # model never touched


def test_reports_progress_per_batch(monkeypatch):
    _wire(monkeypatch, [(i, f"d{i}", "") for i in range(3)], batch=2)
    seen = []
    ec.embed_pending_chunks(progress=lambda done, total: seen.append((done, total)))
    assert seen == [(2, 3), (3, 3)]                         # two batches: 2/3 then 3/3


def test_cancels_between_batches_keeping_progress(monkeypatch):
    cur, conn = _wire(monkeypatch, [(i, f"d{i}", "") for i in range(5)], batch=2)
    cancel = {"n": 0}

    def should_cancel():
        cancel["n"] += 1
        return cancel["n"] > 1                              # allow the 1st batch, cancel before the 2nd

    res = ec.embed_pending_chunks(should_cancel=should_cancel)
    assert res["cancelled"] is True
    assert res["embedded"] == 2                             # first batch committed, then stopped
    assert len(cur.updates) == 2
