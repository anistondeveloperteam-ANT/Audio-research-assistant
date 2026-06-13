"""PDF upload dedup: re-adding the same PDF is skipped by CONTENT hash (not filename), so it is
never re-parsed/re-embedded. Filesystem only — no Oracle, no embeddings."""
import webapp.ingest as ingest


def test_save_pdf_dedups_identical_content(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "PAPERS_DIR", tmp_path)
    data = b"%PDF-1.4 hello world"

    assert ingest.save_pdf("paper.pdf", data)["status"] == "saved"
    # same content, same name -> duplicate
    assert ingest.save_pdf("paper.pdf", data)["status"] == "duplicate"
    # same content, DIFFERENT name -> still duplicate (dedup is by content hash, not filename)
    assert ingest.save_pdf("renamed.pdf", data)["status"] == "duplicate"
    assert len(list(tmp_path.glob("*.pdf"))) == 1            # only one copy on disk

    # different content is saved
    assert ingest.save_pdf("other.pdf", b"%PDF-1.4 different")["status"] == "saved"
    assert len(list(tmp_path.glob("*.pdf"))) == 2


def test_save_pdf_rejects_non_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "PAPERS_DIR", tmp_path)
    assert ingest.save_pdf("x.pdf", b"not a pdf")["status"] == "error"
    assert ingest.save_pdf("x.pdf", b"")["status"] == "error"
