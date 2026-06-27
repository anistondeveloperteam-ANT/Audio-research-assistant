"""stream_ingest() orchestration: parse runs as a CPU subprocess while each finished paper is
embedded IN-PROCESS on the GPU (overlap), then a model-free migrate stage runs. The parser
subprocess and the embedder are faked, so the test is fast and needs no Oracle/GPU/Docling.
"""
import backend.ingestion.embed_chunks as ec
import webapp.ingest as ingest


class FakeProc:
    """Minimal stand-in for a Popen: streams canned stdout lines, then exits with `code`."""
    def __init__(self, lines, code=0):
        self._lines = list(lines)
        self._code = code
        self.returncode = None
        self.terminated = False

    @property
    def stdout(self):
        return self

    def readline(self):
        return (self._lines.pop(0) + "\n") if self._lines else ""   # "" == EOF

    def close(self):
        pass

    def wait(self):
        self.returncode = self._code
        return self._code

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True


def _fake_popen(scripted):
    def _popen(cmd, **kwargs):
        module = cmd[2]                                     # [python, "-m", module, *extra]
        lines, code = scripted.get(module, ([], 0))
        return FakeProc(lines, code)
    return _popen


def _common(monkeypatch):
    monkeypatch.setattr(ingest, "_post_embed_stages", lambda: [ingest.MIGRATE_STAGE])
    monkeypatch.setattr(ingest, "_clear_retrieval_caches", lambda *a, **k: None)
    monkeypatch.setattr(ingest, "library_stats", lambda: {"pdfs": 2})


def test_embeds_each_paper_as_it_parses_then_migrates(monkeypatch):
    ingest.begin_ingest([])
    _common(monkeypatch)
    scripted = {
        "backend.ingestion.ingest_papers": (
            ["Found 2 PDFs",
             "Ingested: a.pdf | parser=docling | pages_indexed=3/3 | chunks=8",
             "Ingested: b.pdf | parser=docling | pages_indexed=2/2 | chunks=5",
             "Ingestion summary:"], 0),
        "backend.database.vector_migration": (["Migrated vectors into the index"], 0),
    }
    monkeypatch.setattr(ingest.subprocess, "Popen", _fake_popen(scripted))

    calls = []
    monkeypatch.setattr(ec, "embed_pending_chunks",
                        lambda should_cancel=None: (calls.append(1) or
                                                    {"embedded": 4, "total": 4, "cancelled": False}))

    events = list(ingest.stream_ingest())
    types = [e["type"] for e in events]
    labels = [e["label"] for e in events if e["type"] == "stage"]

    # One embed pass per parsed paper (overlap) + a final pass = 3; the model is reused, not reloaded.
    assert len(calls) == 3
    assert labels == ["Reading & chunking the PDF", "Building embeddings", "Updating the vector index"]
    assert types[-1] == "done"
    assert events[-1]["library"] == {"pdfs": 2}


def test_embed_failure_surfaces_clean_error_not_a_crash(monkeypatch):
    """The whole point of the fix: an embed failure is reported as a clean message, never the cryptic
    subprocess crash the user saw."""
    ingest.begin_ingest([])
    _common(monkeypatch)
    monkeypatch.setattr(ingest.subprocess, "Popen",
                        _fake_popen({"backend.ingestion.ingest_papers": (["Ingested: a.pdf | x"], 0)}))

    def boom(should_cancel=None):
        raise RuntimeError("model load failed")

    monkeypatch.setattr(ec, "embed_pending_chunks", boom)
    events = list(ingest.stream_ingest())
    assert events[-1]["type"] == "error"
    assert "Building embeddings failed" in events[-1]["message"]


def test_parse_failure_surfaces_error_before_embedding(monkeypatch):
    ingest.begin_ingest([])
    _common(monkeypatch)
    monkeypatch.setattr(ingest.subprocess, "Popen",
                        _fake_popen({"backend.ingestion.ingest_papers": (["parser blew up"], 3)}))
    calls = []
    monkeypatch.setattr(ec, "embed_pending_chunks",
                        lambda should_cancel=None: calls.append(1) or {"embedded": 0, "total": 0, "cancelled": False})

    events = list(ingest.stream_ingest())
    assert events[-1]["type"] == "error"
    assert "exit code 3" in events[-1]["message"]
    assert calls == []                                      # never embedded after a parse failure


def test_pre_cancel_spawns_nothing(monkeypatch):
    ingest.begin_ingest([])
    with ingest._ingest_lock:
        ingest._ingest_state["cancelled"] = True
    spawned = []
    monkeypatch.setattr(ingest.subprocess, "Popen",
                        lambda *a, **k: spawned.append(1))
    events = list(ingest.stream_ingest())
    assert events[0]["type"] == "cancelled"
    assert spawned == []                                    # no parser subprocess started


# ---- single-flight guard: only one ingest at a time (prevents concurrent-run rate-limit storms) ----
def test_begin_ingest_is_single_flight():
    assert ingest.begin_ingest(["a.pdf"]) is True           # claims the slot
    try:
        assert ingest.begin_ingest(["b.pdf"]) is False      # a second run is refused while one is active
    finally:
        ingest._end_ingest()
    assert ingest.begin_ingest(["c.pdf"]) is True            # slot freed -> can start again
    ingest._end_ingest()


# ---- 'Finish embedding' (embed half-done papers without re-parsing) ----
def test_embed_pending_embeds_then_migrates(monkeypatch):
    ingest.begin_ingest([])
    _common(monkeypatch)
    monkeypatch.setattr(ingest.subprocess, "Popen",
                        _fake_popen({"backend.database.vector_migration": (["Migrated vectors"], 0)}))
    monkeypatch.setattr(ec, "embed_pending_chunks",
                        lambda should_cancel=None: {"embedded": 7, "total": 7, "cancelled": False})
    events = list(ingest.stream_embed_pending())
    labels = [e["label"] for e in events if e["type"] == "stage"]
    assert labels == ["Building embeddings", "Updating the vector index"]   # embed then migrate, no parse
    assert events[-1]["type"] == "done" and "Embedded 7" in events[-1]["message"]


def test_embed_pending_nothing_to_embed_skips_migrate(monkeypatch):
    ingest.begin_ingest([])
    _common(monkeypatch)
    spawned = []
    monkeypatch.setattr(ingest.subprocess, "Popen", lambda *a, **k: spawned.append(1))
    monkeypatch.setattr(ec, "embed_pending_chunks",
                        lambda should_cancel=None: {"embedded": 0, "total": 0, "cancelled": False})
    events = list(ingest.stream_embed_pending())
    assert events[-1]["type"] == "done" and "Nothing to embed" in events[-1]["message"]
    assert spawned == []                                    # nothing pending -> no migrate subprocess


def test_embed_pending_failure_surfaces_clean_error(monkeypatch):
    ingest.begin_ingest([])
    _common(monkeypatch)

    def boom(should_cancel=None):
        raise RuntimeError("model load failed")

    monkeypatch.setattr(ec, "embed_pending_chunks", boom)
    events = list(ingest.stream_embed_pending())
    assert events[-1]["type"] == "error" and "Building embeddings failed" in events[-1]["message"]


def test_embed_pending_releases_slot_when_done(monkeypatch):
    _common(monkeypatch)
    monkeypatch.setattr(ingest.subprocess, "Popen",
                        _fake_popen({"backend.database.vector_migration": (["ok"], 0)}))
    monkeypatch.setattr(ec, "embed_pending_chunks",
                        lambda should_cancel=None: {"embedded": 1, "total": 1, "cancelled": False})
    assert ingest.begin_ingest([]) is True
    list(ingest.stream_embed_pending())                     # run to completion
    assert ingest.begin_ingest(["b.pdf"]) is True           # slot released on finish
    ingest._end_ingest()


def test_stream_ingest_releases_slot_when_done(monkeypatch):
    _common(monkeypatch)
    monkeypatch.setattr(ingest.subprocess, "Popen",
                        _fake_popen({"backend.ingestion.ingest_papers": (["Ingested: a.pdf | x"], 0)}))
    monkeypatch.setattr(ec, "embed_pending_chunks",
                        lambda should_cancel=None: {"embedded": 1, "total": 1, "cancelled": False})
    assert ingest.begin_ingest(["a.pdf"]) is True
    list(ingest.stream_ingest())                             # run to completion
    assert ingest.begin_ingest(["b.pdf"]) is True            # stream_ingest released the slot on finish
    ingest._end_ingest()
