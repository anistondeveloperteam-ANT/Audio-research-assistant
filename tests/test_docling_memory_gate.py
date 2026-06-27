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
    assert pp.docling_enabled() is False                  # opt-IN: OFF by default (CPU cost too high)
    monkeypatch.setenv("ENABLE_DOCLING", "true")
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


# ---- conditional Docling: run the heavy parser ONLY on table-rich papers ----
_PROSE = ("This paper studies dereverberation. We motivate the approach and review prior work. "
          "The method is described in prose with no result tables here at all, just discussion.")
_TABLE_RICH = ("Table 1 compares methods. Table 2 reports ablations. Our model improves PESQ and STOI "
               "and SI-SDR over baselines; SNR and latency are also reported. | A | B | C | D |")


def test_table_richness_separates_prose_from_tables():
    prose = pp._table_richness([{"text": _PROSE}])
    rich = pp._table_richness([{"text": _TABLE_RICH}])
    assert rich >= pp.docling_table_trigger() > prose      # table paper crosses the trigger, prose doesn't


def test_docling_conditional_default_and_override(monkeypatch):
    monkeypatch.delenv("DOCLING_CONDITIONAL", raising=False)
    assert pp.docling_conditional() is True
    monkeypatch.setenv("DOCLING_CONDITIONAL", "false")
    assert pp.docling_conditional() is False


def test_should_run_docling_conditional_skips_prose(monkeypatch):
    monkeypatch.setenv("ENABLE_DOCLING", "true")
    monkeypatch.delenv("DOCLING_CONDITIONAL", raising=False)      # conditional ON (default)
    monkeypatch.setattr(pp, "_available_memory_mb", lambda: 32000)
    run, why = pp._should_run_docling([{"text": _PROSE}])
    assert run is False and "conditional" in why                 # prose -> skip the heavy parse


def test_should_run_docling_conditional_runs_on_tables(monkeypatch):
    monkeypatch.setenv("ENABLE_DOCLING", "true")
    monkeypatch.delenv("DOCLING_CONDITIONAL", raising=False)
    monkeypatch.setattr(pp, "_available_memory_mb", lambda: 32000)
    run, why = pp._should_run_docling([{"text": _TABLE_RICH}])
    assert run is True and why == ""                             # tables -> run Docling (keep accuracy)


def test_should_run_docling_conditional_off_runs_on_prose(monkeypatch):
    monkeypatch.setenv("ENABLE_DOCLING", "true")
    monkeypatch.setenv("DOCLING_CONDITIONAL", "false")           # always-run mode
    monkeypatch.setattr(pp, "_available_memory_mb", lambda: 32000)
    run, why = pp._should_run_docling([{"text": _PROSE}])
    assert run is True and why == ""


def test_parse_pdf_skips_docling_on_prose(monkeypatch):
    monkeypatch.setenv("ENABLE_DOCLING", "true")
    monkeypatch.delenv("DOCLING_CONDITIONAL", raising=False)
    monkeypatch.setattr(pp, "estimate_page_count", lambda p: 1)
    monkeypatch.setattr(pp, "parse_with_pymupdf", lambda p: _pm(_PROSE))
    monkeypatch.setattr(pp, "_available_memory_mb", lambda: 32000)   # ample memory
    called = {"n": 0}
    monkeypatch.setattr(pp, "_docling_safe",
                        lambda p: called.__setitem__("n", called["n"] + 1) or {"raw_markdown": "z"})
    res = pp.parse_pdf(Path("x.pdf"))
    assert called["n"] == 0                       # prose paper -> Docling skipped despite ample memory
    assert res["parser"] == "pymupdf"


def test_parse_pdf_runs_docling_on_table_rich(monkeypatch):
    monkeypatch.setenv("ENABLE_DOCLING", "true")
    monkeypatch.delenv("DOCLING_CONDITIONAL", raising=False)
    monkeypatch.setattr(pp, "estimate_page_count", lambda p: 1)
    monkeypatch.setattr(pp, "parse_with_pymupdf", lambda p: _pm(_TABLE_RICH))
    monkeypatch.setattr(pp, "_available_memory_mb", lambda: 32000)
    called = {"n": 0}
    monkeypatch.setattr(pp, "_docling_safe",
                        lambda p: called.__setitem__("n", called["n"] + 1) or None)
    pp.parse_pdf(Path("x.pdf"))
    assert called["n"] == 1                       # table-rich paper -> Docling runs (structure preserved)


# ---- Step 5: fast clean tables via PyMuPDF find_tables (no ML), gated on table-richness ----
def test_pymupdf_tables_enabled_default_and_override(monkeypatch):
    monkeypatch.delenv("ENABLE_PYMUPDF_TABLES", raising=False)
    assert pp.pymupdf_tables_enabled() is True
    monkeypatch.setenv("ENABLE_PYMUPDF_TABLES", "false")
    assert pp.pymupdf_tables_enabled() is False


def test_parse_pdf_extracts_pymupdf_tables_when_enabled(monkeypatch):
    monkeypatch.setenv("ENABLE_DOCLING", "false")               # no Docling tables
    monkeypatch.delenv("ENABLE_PYMUPDF_TABLES", raising=False)
    monkeypatch.setattr(pp, "estimate_page_count", lambda p: 1)
    monkeypatch.setattr(pp, "parse_with_pymupdf", lambda p: _pm(_TABLE_RICH))
    monkeypatch.setattr(pp, "extract_pymupdf_tables", lambda p: ["|A|B|\n|---|---|\n|1|2|"])
    res = pp.parse_pdf(Path("x.pdf"))
    assert res["tables"] == ["|A|B|\n|---|---|\n|1|2|"]          # find_tables grids included


def test_parse_pdf_runs_find_tables_even_without_table_references(monkeypatch):
    # No "Table N" in the text, but the paper may still HAVE a table -> find_tables must still run
    # (the detector is the gate). Mock returns [] (no real grid) -> tables empty, but it WAS called.
    monkeypatch.setenv("ENABLE_DOCLING", "false")
    monkeypatch.delenv("ENABLE_PYMUPDF_TABLES", raising=False)
    monkeypatch.setattr(pp, "estimate_page_count", lambda p: 1)
    monkeypatch.setattr(pp, "parse_with_pymupdf", lambda p: _pm(_PROSE))
    called = {"n": 0}
    monkeypatch.setattr(pp, "extract_pymupdf_tables",
                        lambda p: called.__setitem__("n", called["n"] + 1) or [])
    res = pp.parse_pdf(Path("x.pdf"))
    assert called["n"] == 1 and res["tables"] == []             # ran (no gate), found nothing


def test_parse_pdf_pymupdf_tables_off_when_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_DOCLING", "false")
    monkeypatch.setenv("ENABLE_PYMUPDF_TABLES", "false")
    monkeypatch.setattr(pp, "estimate_page_count", lambda p: 1)
    monkeypatch.setattr(pp, "parse_with_pymupdf", lambda p: _pm(_TABLE_RICH))
    called = {"n": 0}
    monkeypatch.setattr(pp, "extract_pymupdf_tables",
                        lambda p: called.__setitem__("n", called["n"] + 1) or ["x"])
    res = pp.parse_pdf(Path("x.pdf"))
    assert called["n"] == 0 and res["tables"] == []             # disabled -> not run


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
