"""Multiple-PDF upload: save_pdfs handles a batch with content-hash dedup (across the library AND
within the batch) and mixed valid/invalid files. Filesystem only — no Oracle, no embeddings."""
import webapp.ingest as ingest


def test_save_pdfs_batch_dedups_and_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "PAPERS_DIR", tmp_path)
    a = b"%PDF-1.4 alpha"
    b = b"%PDF-1.4 beta"
    items = [
        ("a.pdf", a),
        ("b.pdf", b),
        ("a-again.pdf", a),     # same content as a -> intra-batch duplicate
        ("notes.txt", b"nope"),  # not a PDF -> error
    ]
    res = ingest.save_pdfs(items)

    assert res["total"] == 4
    assert res["saved"] == 2 and res["duplicate"] == 1 and res["error"] == 1
    assert len(list(tmp_path.glob("*.pdf"))) == 2          # only the two distinct PDFs on disk

    by_name = {r["name"]: r["status"] for r in res["results"]}
    assert by_name == {"a.pdf": "saved", "b.pdf": "saved",
                       "a-again.pdf": "duplicate", "notes.txt": "error"}


def test_save_pdfs_dedups_against_existing_library(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "PAPERS_DIR", tmp_path)
    data = b"%PDF-1.4 already here"
    assert ingest.save_pdf("first.pdf", data)["status"] == "saved"
    # a later batch containing the same content is skipped
    res = ingest.save_pdfs([("dup.pdf", data), ("new.pdf", b"%PDF-1.4 fresh")])
    assert res["saved"] == 1 and res["duplicate"] == 1
    assert len(list(tmp_path.glob("*.pdf"))) == 2


def test_save_pdfs_empty_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "PAPERS_DIR", tmp_path)
    res = ingest.save_pdfs([])
    assert res == {"results": [], "total": 0, "saved": 0, "duplicate": 0, "error": 0}


def test_ingest_log_noise_filter():
    # Noise the UI must NOT show (tqdm bars, dedup skips, recovered-page traces, internal logs).
    for noisy in [
        "Ingesting PDFs: 100%|####| 10/10 [01:32<00:00,  9.24s/it]",
        "Skipping already ingested: Foo.pdf",
        "Stage preprocess failed for run 1, pages [5]: std::bad_alloc",
        'File "x.py", line 5, in foo',
        "  Docling layout device: cpu (do_ocr=False, page_batch=1)",
        "   ",
    ]:
        assert ingest._is_log_noise(noisy) is True, noisy
    # Meaningful lines the UI MUST keep.
    for keep in [
        "Ingested: Foo.pdf | parser=docling | pages_indexed=8/8 | chunks=41 | parsed in 6.0s",
        "Page coverage: all pages indexed (no warnings).",
        "WARNING: 1 page(s) failed/empty and are NOT indexed: [5]",
        "Embedded chunks: 297/297",
        "Vector migration complete.",
    ]:
        assert ingest._is_log_noise(keep) is False, keep


def test_cancel_ingest_removes_only_the_in_progress_paper(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "PAPERS_DIR", tmp_path)
    (tmp_path / "new.pdf").write_bytes(b"%PDF-1.4 new")      # the in-progress upload
    (tmp_path / "keep.pdf").write_bytes(b"%PDF-1.4 keep")    # an already-indexed paper
    deleted_rows = []
    monkeypatch.setattr(ingest, "_delete_rows_by_filename", lambda n: deleted_rows.append(n))
    monkeypatch.setattr(ingest, "_clear_retrieval_caches", lambda **k: None)

    class _Proc:
        def __init__(self):
            self.terminated = False
        def poll(self):
            return None                                       # still running
        def terminate(self):
            self.terminated = True
        def wait(self, timeout=None):
            return 0
        def kill(self):
            self.terminated = True

    ingest.begin_ingest(["new.pdf"])
    proc = _Proc()
    ingest._register_ingest_proc(proc)

    res = ingest.cancel_ingest()
    assert res["cancelled"] is True and res["removed"] == ["new.pdf"]
    assert proc.terminated is True                            # subprocess stopped
    assert deleted_rows == ["new.pdf"]                        # only the in-progress paper's rows
    assert not (tmp_path / "new.pdf").exists()                # its PDF removed
    assert (tmp_path / "keep.pdf").exists()                   # other paper untouched


def test_stream_ingest_exits_immediately_when_pre_cancelled(monkeypatch):
    ingest.begin_ingest([])
    with ingest._ingest_lock:
        ingest._ingest_state["cancelled"] = True              # cancelled before any stage runs
    events = list(ingest.stream_ingest())
    assert events and events[0]["type"] == "cancelled"        # no subprocess spawned
