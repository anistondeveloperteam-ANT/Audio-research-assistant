"""Low-memory / toggle gating for the heavy Docling parser. This is the fix for the 'add papers'
ingest crashes ('exit code 3221225477' / MemoryError): on a memory-starved host Docling is skipped
and the crash-proof PyMuPDF text path indexes the document instead. Fully offline — no Docling, no fitz."""
from pathlib import Path

import backend.ingestion.pdf_parser as pp


def _pm(*texts):
    pages = [{"page": i + 1, "text": t, "parser": "pymupdf"} for i, t in enumerate(texts)]
    return {"parser": "pymupdf", "pages": pages, "page_count": len(texts),
            "raw_markdown": "", "tables": [], "equations": []}


def test_docling_enabled_toggle(monkeypatch):
    monkeypatch.delenv("ENABLE_DOCLING", raising=False)
    assert pp.docling_enabled() is True
    monkeypatch.setenv("ENABLE_DOCLING", "false")
    assert pp.docling_enabled() is False


def test_docling_min_free_mb_default_and_override(monkeypatch):
    monkeypatch.delenv("DOCLING_MIN_FREE_MB", raising=False)
    assert pp.docling_min_free_mb() == 1500
    monkeypatch.setenv("DOCLING_MIN_FREE_MB", "4096")
    assert pp.docling_min_free_mb() == 4096
    monkeypatch.setenv("DOCLING_MIN_FREE_MB", "notanint")
    assert pp.docling_min_free_mb() == 1500


def test_available_memory_mb_is_positive():
    assert pp._available_memory_mb() > 0                  # real read; never zero/negative


def test_should_run_docling_skips_when_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_DOCLING", "false")
    run, why = pp._should_run_docling()
    assert run is False and "ENABLE_DOCLING" in why


def test_should_run_docling_skips_on_low_memory(monkeypatch):
    monkeypatch.setenv("ENABLE_DOCLING", "true")
    monkeypatch.setenv("DOCLING_MIN_FREE_MB", "1500")
    monkeypatch.setattr(pp, "_available_memory_mb", lambda: 500)   # starved host
    run, why = pp._should_run_docling()
    assert run is False and "low memory" in why


def test_should_run_docling_runs_when_memory_is_ample(monkeypatch):
    monkeypatch.setenv("ENABLE_DOCLING", "true")
    monkeypatch.delenv("DOCLING_MIN_FREE_MB", raising=False)
    monkeypatch.setattr(pp, "_available_memory_mb", lambda: 32000)
    run, why = pp._should_run_docling()
    assert run is True and why == ""


def test_docling_min_free_zero_never_skips_on_memory(monkeypatch):
    monkeypatch.setenv("ENABLE_DOCLING", "true")
    monkeypatch.setenv("DOCLING_MIN_FREE_MB", "0")
    monkeypatch.setattr(pp, "_available_memory_mb", lambda: 10)    # almost nothing free
    run, why = pp._should_run_docling()
    assert run is True


def test_parse_pdf_skips_docling_under_memory_pressure(monkeypatch):
    monkeypatch.setattr(pp, "estimate_page_count", lambda p: 1)
    monkeypatch.setattr(pp, "parse_with_pymupdf", lambda p: _pm("x" * 400))
    called = {"n": 0}
    monkeypatch.setattr(pp, "_docling_safe",
                        lambda p: called.__setitem__("n", called["n"] + 1) or {"raw_markdown": "z"})
    monkeypatch.setattr(pp, "_available_memory_mb", lambda: 500)   # below the 1500 MB floor
    monkeypatch.setenv("DOCLING_MIN_FREE_MB", "1500")

    res = pp.parse_pdf(Path("x.pdf"))
    assert called["n"] == 0                       # the heavy parser was NEVER invoked -> no crash
    assert res["parser"] == "pymupdf"             # indexed via the crash-proof text path
    assert res["pages_indexed"] == 1
