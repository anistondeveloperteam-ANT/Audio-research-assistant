"""PDF parsing + OCR device/coverage behaviour — fully mocked (no Docling, no real OCR, no fitz)."""
import os
import sys
import types
from pathlib import Path

import backend.ingestion.ocr_fallback as ocr
import backend.ingestion.pdf_parser as pp


def _pages(*texts):
    return [{"page": i + 1, "text": t, "parser": "pymupdf"} for i, t in enumerate(texts)]


def _pm(*texts):
    return {"parser": "pymupdf", "pages": _pages(*texts), "page_count": len(texts),
            "raw_markdown": "", "tables": [], "equations": []}


# ---- OCR engine is forced onto the CPU (never grabs the GPU) ----------
def test_paddleocr_is_forced_to_cpu(monkeypatch):
    captured = {}

    class FakePaddle:
        def __init__(self, **kw):
            captured.update(kw)

        def ocr(self, *a, **k):
            return []

    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCR=FakePaddle))
    monkeypatch.setattr(ocr, "_paddle_ocr", None)        # reset the cached instance
    ocr._get_paddle()
    assert captured.get("use_gpu") is False               # CPU, not GPU


def test_ocr_pages_only_renders_requested_pages(monkeypatch):
    rendered = []
    monkeypatch.setattr(ocr, "_render_page_image",
                        lambda pdf, p, out, dpi=200: rendered.append(p) or Path(f"p{p}.png"))
    monkeypatch.setattr(ocr, "_ocr_image_cpu", lambda img: "recovered text " * 5)
    out = ocr.ocr_pages(Path("x.pdf"), [6, 7, 8])
    assert rendered == [6, 7, 8]                           # only the text-poor pages, nothing else
    assert set(out) == {6, 7, 8} and all(out.values())


def test_ocr_pages_empty_input_is_noop(monkeypatch):
    monkeypatch.setattr(ocr, "_render_page_image", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert ocr.ocr_pages(Path("x.pdf"), []) == {}


def test_ocr_pages_one_page_failure_does_not_abort_rest(monkeypatch):
    monkeypatch.setattr(ocr, "_render_page_image", lambda pdf, p, out, dpi=200: Path(f"p{p}.png"))

    def flaky(img):
        if "p7" in str(img):
            raise RuntimeError("boom")
        return "text " * 10

    monkeypatch.setattr(ocr, "_ocr_image_cpu", flaky)
    out = ocr.ocr_pages(Path("x.pdf"), [6, 7, 8])
    assert set(out) == {6, 8}                              # page 7 failed, 6 & 8 still returned


# ---- Docling: OCR disabled + OOM-safe layout ---------------------------
def test_docling_pipeline_disables_ocr():
    opts = types.SimpleNamespace()
    pp._configure_pipeline_options(opts)
    assert opts.do_ocr is False                            # RapidOCR never initialised


def test_oom_error_detection():
    assert pp._is_oom_error(RuntimeError("std::bad_alloc")) is True
    assert pp._is_oom_error(RuntimeError("CUDA out of memory")) is True
    assert pp._is_oom_error(ValueError("a normal error")) is False


def test_force_cpu_parsing_hides_cuda_by_default(monkeypatch):
    monkeypatch.delenv("DOCLING_DEVICE", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert pp._should_hide_cuda() is True
    pp.force_cpu_parsing()
    # MUST be "-1" — an empty string does NOT hide the GPU (torch stays cuda_available=True).
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"


def test_docling_page_batch_default_and_override(monkeypatch):
    monkeypatch.delenv("DOCLING_PAGE_BATCH", raising=False)
    assert pp._docling_page_batch() == 1                  # low peak memory + faster by default
    monkeypatch.setenv("DOCLING_PAGE_BATCH", "4")
    assert pp._docling_page_batch() == 4
    monkeypatch.setenv("DOCLING_PAGE_BATCH", "bad")
    assert pp._docling_page_batch() == 1                  # bad value -> safe default


def test_suppress_native_stderr_is_safe(capsys):
    # No-op-safe context manager: the body runs and nothing raises (stderr is captured here).
    ran = []
    with pp._suppress_native_stderr():
        ran.append(True)
    assert ran == [True]


def test_docling_device_cuda_keeps_gpu_visible(monkeypatch):
    monkeypatch.setenv("DOCLING_DEVICE", "cuda")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert pp._should_hide_cuda() is False
    pp.force_cpu_parsing()
    assert "CUDA_VISIBLE_DEVICES" not in os.environ        # opt-in: GPU layout allowed


def test_docling_safe_falls_back_to_none_on_failure(monkeypatch):
    def boom(pdf, dev):
        raise RuntimeError("docling not installed")
    monkeypatch.setattr(pp, "_docling_convert", boom)
    assert pp._docling_safe(Path("x.pdf")) is None         # genuine failure -> PyMuPDF text path


# ---- parse_pdf: skip OCR on digital pages, coverage, per-page rescue ---
def test_ocr_skipped_when_pages_have_text(monkeypatch):
    monkeypatch.setenv("ENABLE_OCR", "true")               # even enabled: text pages must skip OCR
    monkeypatch.setattr(pp, "estimate_page_count", lambda p: 2)
    monkeypatch.setattr(pp, "parse_with_pymupdf", lambda p: _pm("x" * 400, "y" * 400))
    monkeypatch.setattr(pp, "_docling_safe", lambda p: {"raw_markdown": "z" * 800, "tables": [], "equations": []})
    called = {"n": 0}
    monkeypatch.setattr(pp, "ocr_pages", lambda *a, **k: called.__setitem__("n", called["n"] + 1) or {})

    res = pp.parse_pdf(Path("x.pdf"))
    assert called["n"] == 0                                # OCR never invoked on born-digital pages
    assert res["pages_indexed"] == 2 and res["pages_missing"] == []
    assert res["warnings"] == []


def test_pages_needing_ocr_helper():
    assert pp._pages_needing_ocr(_pages("plenty of text " * 10, "", "  "), min_chars=50) == [2, 3]


def test_missing_page_warning_when_ocr_off(monkeypatch):
    monkeypatch.setenv("ENABLE_OCR", "false")              # scanned page not OCR'd -> reported missing
    monkeypatch.setattr(pp, "estimate_page_count", lambda p: 3)
    monkeypatch.setattr(pp, "parse_with_pymupdf", lambda p: _pm("x" * 400, "y" * 400, ""))
    monkeypatch.setattr(pp, "_docling_safe", lambda p: None)
    monkeypatch.setattr(pp, "ocr_pages", lambda *a, **k: {})
    res = pp.parse_pdf(Path("x.pdf"))
    assert res["pages_missing"] == [3]
    assert res["pages_indexed"] == 2
    assert any("NOT indexed" in w for w in res["warnings"])


def test_per_page_cpu_ocr_rescues_scanned_page(monkeypatch):
    monkeypatch.setenv("ENABLE_OCR", "true")
    monkeypatch.setattr(pp, "estimate_page_count", lambda p: 2)
    monkeypatch.setattr(pp, "parse_with_pymupdf", lambda p: _pm("x" * 400, ""))   # page 2 empty
    monkeypatch.setattr(pp, "_docling_safe", lambda p: None)
    monkeypatch.setattr(pp, "ocr_pages", lambda path, nums, **k: {2: "recovered text " * 10})
    res = pp.parse_pdf(Path("x.pdf"))
    assert res["ocr_pages"] == [2]                         # page 2 OCR'd on CPU and rescued
    assert res["pages_missing"] == [] and res["pages_indexed"] == 2


# ---- ingest summary surfaces coverage + the missing-page warning -------
def test_ingest_summary_reports_pages_and_warning():
    from backend.ingestion import ingest_papers as ip
    parsed = {"parser": "docling", "pages_indexed": 5, "pages_total": 8,
              "warnings": ["WARNING: 3 page(s) failed/empty and are NOT indexed: [6, 7, 8]"]}
    line = ip.coverage_line("Deep.pdf", parsed, 40)
    assert "pages_indexed=5/8" in line
    assert "parsed in 8.3s" in ip.coverage_line("Deep.pdf", parsed, 40, 8.3)   # per-PDF parse time
    warns = ip.coverage_warnings("Deep.pdf", parsed)
    assert warns and warns[0].startswith("Deep.pdf:") and "NOT indexed" in warns[0]
    # a fully-indexed paper produces no warnings
    assert ip.coverage_warnings("ok.pdf", {"warnings": []}) == []
