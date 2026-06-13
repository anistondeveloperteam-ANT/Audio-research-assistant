"""PDF parsing + OCR device/coverage behaviour — fully mocked (no Docling, no real OCR, no fitz)."""
import sys
import types
from pathlib import Path

import backend.ingestion.ocr_fallback as ocr


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
