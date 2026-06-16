"""Unit tests for scraper.scrape_pdfs. Fully offline -- the network is mocked with a
fake requests session, so no real HTTP is ever performed."""
import pytest
import requests

from scraper.scrape_pdfs import (
    DownloadResult,
    download_pdf,
    filename_from_response,
    filename_from_url,
    looks_like_pdf,
    pdf_title,
    read_urls,
    sanitize_filename,
    scrape,
    unique_path,
)

PDF_BYTES = b"%PDF-1.7\n%fake pdf body for testing\n"


def _pdf_with_title(title: str) -> bytes:
    """Build a tiny real PDF whose metadata /Title is `title` (needs PyMuPDF)."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page()
    doc.set_metadata({"title": title})
    try:
        return doc.tobytes()
    finally:
        doc.close()


# ----------------------------------------------------------------------
# Fakes -- stand in for requests.Session / Response (no network)
# ----------------------------------------------------------------------
class FakeResponse:
    def __init__(self, body=b"", headers=None, error=None):
        self.body = body
        self.headers = headers or {}
        self._error = error
        self.closed = False

    def raise_for_status(self):
        if self._error is not None:
            raise self._error

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i:i + chunk_size]

    def close(self):
        self.closed = True


class FakeSession:
    """Returns the given response (or raises it if it's an Exception) for every GET."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


# ----------------------------------------------------------------------
# read_urls
# ----------------------------------------------------------------------
def test_read_urls_ignores_comments_blanks_and_dedupes(tmp_path):
    f = tmp_path / "urls.txt"
    f.write_text(
        "# a comment\n"
        "https://x/a.pdf\n"
        "\n"
        "   https://x/b.pdf  \n"
        "https://x/a.pdf\n",  # duplicate -> dropped
        encoding="utf-8",
    )
    assert read_urls(f) == ["https://x/a.pdf", "https://x/b.pdf"]


# ----------------------------------------------------------------------
# sanitize_filename
# ----------------------------------------------------------------------
def test_sanitize_filename_replaces_illegal_chars_and_trailing_dots():
    assert sanitize_filename('a/b:c?"d*.pdf') == "a_b_c__d_.pdf"
    assert sanitize_filename("name.   ") == "name"  # trailing dots/spaces stripped


# ----------------------------------------------------------------------
# filename_from_url
# ----------------------------------------------------------------------
def test_filename_from_url_handles_plain_url():
    assert filename_from_url("https://host.com/papers/Beamforming.pdf") == "Beamforming.pdf"


def test_filename_from_url_appends_pdf_when_missing_extension():
    # arXiv-style path with no .pdf suffix
    assert filename_from_url("https://arxiv.org/pdf/2305.12345") == "2305.12345.pdf"


def test_filename_from_url_decodes_percent_encoding():
    assert filename_from_url("https://x/My%20Paper.pdf") == "My Paper.pdf"


def test_filename_from_url_falls_back_to_host_when_no_path():
    assert filename_from_url("https://host.com") == "host.com.pdf"


# ----------------------------------------------------------------------
# filename_from_response
# ----------------------------------------------------------------------
def test_filename_from_response_prefers_content_disposition():
    headers = {"Content-Disposition": 'attachment; filename="Real Paper.pdf"'}
    assert filename_from_response("https://x/download?id=1", headers) == "Real Paper.pdf"


def test_filename_from_response_handles_rfc5987_filename_star():
    headers = {"Content-Disposition": "attachment; filename*=UTF-8''My%20Paper.pdf"}
    assert filename_from_response("https://x/dl", headers) == "My Paper.pdf"


def test_filename_from_response_falls_back_to_url():
    assert filename_from_response("https://x/paper.pdf", {}) == "paper.pdf"


# ----------------------------------------------------------------------
# pdf_title + title-based naming
# ----------------------------------------------------------------------
def test_pdf_title_reads_embedded_metadata():
    assert pdf_title(_pdf_with_title("My Great Paper")) == "My Great Paper"


def test_pdf_title_empty_for_non_pdf_bytes():
    assert pdf_title(b"<html>not a pdf</html>") == ""


def test_filename_from_response_prefers_pdf_title_over_url():
    body = _pdf_with_title("Voice Cloning Survey")
    # No Content-Disposition: the embedded title should win over the arXiv-id URL.
    assert filename_from_response("https://arxiv.org/pdf/2505.00579", {}, body) == \
        "Voice Cloning Survey.pdf"


def test_filename_from_response_url_names_flag_ignores_title():
    body = _pdf_with_title("Voice Cloning Survey")
    assert filename_from_response(
        "https://arxiv.org/pdf/2505.00579", {}, body, use_pdf_title=False
    ) == "2505.00579.pdf"


# ----------------------------------------------------------------------
# looks_like_pdf
# ----------------------------------------------------------------------
def test_looks_like_pdf_true_for_magic_bytes():
    assert looks_like_pdf(b"%PDF-1.4 ...rest...")


def test_looks_like_pdf_tolerates_leading_bom():
    assert looks_like_pdf(b"\xef\xbb\xbf%PDF-1.5")


def test_looks_like_pdf_false_for_html():
    assert not looks_like_pdf(b"<!DOCTYPE html><html>not found</html>")


# ----------------------------------------------------------------------
# unique_path
# ----------------------------------------------------------------------
def test_unique_path_suffixes_on_collision(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"x")
    assert unique_path(tmp_path, "a.pdf").name == "a (1).pdf"
    assert unique_path(tmp_path, "b.pdf").name == "b.pdf"  # no collision


# ----------------------------------------------------------------------
# download_pdf (mocked network)
# ----------------------------------------------------------------------
def test_download_pdf_saves_valid_pdf(tmp_path):
    session = FakeSession(
        FakeResponse(body=PDF_BYTES, headers={"Content-Type": "application/pdf"})
    )
    result = download_pdf("https://x/paper.pdf", tmp_path, session=session)

    assert result.status == "ok"
    saved = tmp_path / "paper.pdf"
    assert saved.exists()
    assert saved.read_bytes() == PDF_BYTES
    assert result.num_bytes == len(PDF_BYTES)


def test_download_pdf_rejects_non_pdf_body(tmp_path):
    session = FakeSession(
        FakeResponse(body=b"<html>404</html>", headers={"Content-Type": "text/html"})
    )
    result = download_pdf("https://x/paper.pdf", tmp_path, session=session)

    assert result.status == "failed"
    assert "not a pdf" in result.message.lower()
    assert not (tmp_path / "paper.pdf").exists()  # nothing written


def test_download_pdf_reports_http_error(tmp_path):
    session = FakeSession(FakeResponse(error=requests.HTTPError("404 Not Found")))
    result = download_pdf("https://x/missing.pdf", tmp_path, session=session)

    assert result.status == "failed"
    assert "request failed" in result.message.lower()


def test_download_pdf_uses_content_disposition_name(tmp_path):
    headers = {
        "Content-Type": "application/pdf",
        "Content-Disposition": 'attachment; filename="Nice Name.pdf"',
    }
    session = FakeSession(FakeResponse(body=PDF_BYTES, headers=headers))
    result = download_pdf("https://x/download?id=9", tmp_path, session=session)

    assert result.status == "ok"
    assert (tmp_path / "Nice Name.pdf").exists()


def test_download_pdf_names_file_from_pdf_title(tmp_path):
    body = _pdf_with_title("Knowledge Distillation for Speech Denoising")
    session = FakeSession(FakeResponse(body=body, headers={"Content-Type": "application/pdf"}))
    result = download_pdf("https://arxiv.org/pdf/2505.03442", tmp_path, session=session)

    assert result.status == "ok"
    assert (tmp_path / "Knowledge Distillation for Speech Denoising.pdf").exists()


def test_download_pdf_url_names_flag_uses_arxiv_id(tmp_path):
    body = _pdf_with_title("Knowledge Distillation for Speech Denoising")
    session = FakeSession(FakeResponse(body=body, headers={"Content-Type": "application/pdf"}))
    result = download_pdf(
        "https://arxiv.org/pdf/2505.03442", tmp_path, session=session, use_pdf_title=False
    )

    assert result.status == "ok"
    assert (tmp_path / "2505.03442.pdf").exists()


def test_download_pdf_skip_existing_avoids_network(tmp_path):
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-old")
    session = FakeSession(FakeResponse(body=PDF_BYTES))
    result = download_pdf("https://x/paper.pdf", tmp_path, session=session, skip_existing=True)

    assert result.status == "skipped"
    assert session.calls == []  # skipped before any request
    assert (tmp_path / "paper.pdf").read_bytes() == b"%PDF-old"  # untouched


def test_download_pdf_writes_unique_name_when_not_overwriting(tmp_path):
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-old")
    session = FakeSession(
        FakeResponse(body=PDF_BYTES, headers={"Content-Type": "application/pdf"})
    )
    result = download_pdf("https://x/paper.pdf", tmp_path, session=session)

    assert result.status == "ok"
    assert (tmp_path / "paper (1).pdf").exists()
    assert (tmp_path / "paper.pdf").read_bytes() == b"%PDF-old"  # original kept


# ----------------------------------------------------------------------
# scrape (batch)
# ----------------------------------------------------------------------
def test_scrape_returns_one_result_per_url_and_reports_progress(tmp_path):
    session = FakeSession(
        FakeResponse(body=PDF_BYTES, headers={"Content-Type": "application/pdf"})
    )
    seen = []
    results = scrape(
        ["https://x/a.pdf", "https://x/b.pdf"],
        tmp_path,
        session=session,
        on_progress=lambda i, total, r: seen.append((i, total, r.status)),
    )

    assert [r.status for r in results] == ["ok", "ok"]
    assert all(isinstance(r, DownloadResult) for r in results)
    assert seen == [(1, 2, "ok"), (2, 2, "ok")]
