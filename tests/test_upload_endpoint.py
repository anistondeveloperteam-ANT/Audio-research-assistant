"""/api/upload accepts MANY PDFs in one request (TestClient, temp papers dir, no Oracle)."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_AUTH", "false")
    monkeypatch.setenv("AUTH_SECRET_KEY", "x" * 64)
    monkeypatch.setenv("ENABLE_LOCAL_RAG", "false")
    import webapp.ingest as ingest
    monkeypatch.setattr(ingest, "PAPERS_DIR", tmp_path)
    import webapp.server as server
    return TestClient(server.app), tmp_path


def test_upload_multiple_pdfs_in_one_request(client):
    c, papers = client
    files = [
        ("files", ("a.pdf", b"%PDF-1.4 alpha", "application/pdf")),
        ("files", ("b.pdf", b"%PDF-1.4 beta", "application/pdf")),
        ("files", ("a-copy.pdf", b"%PDF-1.4 alpha", "application/pdf")),   # duplicate of a.pdf
        ("files", ("bad.txt", b"not a pdf", "application/pdf")),           # invalid content
    ]
    r = c.post("/api/upload", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 4
    assert body["saved"] == 2 and body["duplicate"] == 1 and body["error"] == 1
    assert len(list(papers.glob("*.pdf"))) == 2
    assert len(body["results"]) == 4


def test_upload_single_pdf_still_works(client):
    c, papers = client
    r = c.post("/api/upload", files=[("files", ("solo.pdf", b"%PDF-1.4 solo", "application/pdf"))])
    body = r.json()
    assert body["total"] == 1 and body["saved"] == 1
    assert (papers / "solo.pdf").exists()
