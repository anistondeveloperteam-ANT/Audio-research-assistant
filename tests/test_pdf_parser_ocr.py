"""PDF parsing + OCR device/coverage behaviour — fully mocked (no Docling, no real OCR, no fitz)."""
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


def test_docling_oom_falls_back_to_cpu(monkeypatch):
    calls = []

    def fake_convert(pdf_path, device):
        calls.append(device)
        if device != "cpu":
            raise RuntimeError("std::bad_alloc")
        return {"raw_markdown": "# ok\n" + "a" * 500, "tables": [], "equations": []}

    monkeypatch.setattr(pp, "_docling_convert", fake_convert)
    monkeypatch.setattr(pp, "docling_device", lambda: "cuda")
    emptied = {"n": 0}
    monkeypatch.setattr(pp, "_empty_cuda_cache", lambda: emptied.__setitem__("n", emptied["n"] + 1))
    out = pp._docling_safe(Path("x.pdf"))
    assert calls == ["cuda", "cpu"]                        # GPU failed -> retried on CPU
    assert emptied["n"] == 1 and out["raw_markdown"]        # cache emptied, CPU result used


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
    warns = ip.coverage_warnings("Deep.pdf", parsed)
    assert warns and warns[0].startswith("Deep.pdf:") and "NOT indexed" in warns[0]
    # a fully-indexed paper produces no warnings
    assert ip.coverage_warnings("ok.pdf", {"warnings": []}) == []
