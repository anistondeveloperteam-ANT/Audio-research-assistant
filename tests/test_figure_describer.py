"""Figure understanding (ENABLE_FIGURE_UNDERSTANDING): figures are described by a multimodal LLM and
indexed as searchable 'figure' chunks. The renderer (fitz), the provider, and the vision call are all
mocked, so these tests are fast, offline, and need no real PDF/key/GPU."""
from pathlib import Path

import backend.ingestion.figure_describer as fd


_PARSED = {
    "parser": "pymupdf", "page_count": 3,
    "pages": [
        {"page": 1, "text": "Intro text with no figures here."},
        {"page": 2, "text": "Some body.\nFigure 1: Validation loss versus training epochs.\nMore body."},
        {"page": 3, "text": "Fig. 2. PESQ scores across the three systems compared."},
    ],
}


def test_enabled_default_off(monkeypatch):
    monkeypatch.delenv("ENABLE_FIGURE_UNDERSTANDING", raising=False)
    assert fd.enabled() is False
    monkeypatch.setenv("ENABLE_FIGURE_UNDERSTANDING", "true")
    assert fd.enabled() is True


def test_figure_captions_finds_each_with_page():
    caps = fd._figure_captions(_PARSED)
    assert [(c["page"], c["num"]) for c in caps] == [(2, "1"), (3, "2")]
    assert "Validation loss" in caps[0]["caption"]


def test_figure_chunks_empty_when_disabled(monkeypatch):
    monkeypatch.delenv("ENABLE_FIGURE_UNDERSTANDING", raising=False)
    assert fd.figure_chunks(Path("x.pdf"), _PARSED) == []


def test_figure_chunks_empty_when_no_vision_provider(monkeypatch):
    monkeypatch.setenv("ENABLE_FIGURE_UNDERSTANDING", "true")
    monkeypatch.setattr(fd, "_vision_provider", lambda: None)        # no key / text-only model
    assert fd.figure_chunks(Path("x.pdf"), _PARSED) == []


def test_figure_chunks_happy_path(monkeypatch):
    monkeypatch.setenv("ENABLE_FIGURE_UNDERSTANDING", "true")
    monkeypatch.setattr(fd, "_vision_provider", lambda: object())
    monkeypatch.setattr(fd, "_render_page_png", lambda path, page, px: b"\x89PNG-bytes")
    monkeypatch.setattr(fd, "_describe",
                        lambda prov, cap, png: "It shows loss falling from 2.1 to 0.4 over 50 epochs.")
    chunks = fd.figure_chunks(Path("x.pdf"), _PARSED)
    assert len(chunks) == 2
    c = chunks[0]
    assert c["section"] == "Figure" and c["chunk_type"] == "figure" and c["page_start"] == 2
    assert "Validation loss" in c["text"] and "falling from 2.1" in c["text"]


def test_figure_chunks_skips_unrenderable(monkeypatch):
    monkeypatch.setenv("ENABLE_FIGURE_UNDERSTANDING", "true")
    monkeypatch.setattr(fd, "_vision_provider", lambda: object())
    monkeypatch.setattr(fd, "_render_page_png", lambda path, page, px: None)   # render fails
    monkeypatch.setattr(fd, "_describe", lambda prov, cap, png: "desc")
    assert fd.figure_chunks(Path("x.pdf"), _PARSED) == []


def test_describe_builds_vision_message_and_is_failsafe():
    class _FakeVision:
        def __init__(self):
            self.messages = None
            self.kwargs = None

        def stream_chat(self, messages, system="", **k):
            self.messages = messages
            self.kwargs = k
            return ["The chart shows a downward loss curve."]

    fp = _FakeVision()
    out = fd._describe(fp, "Figure 1: loss vs epochs", b"PNGDATA")
    assert out == "The chart shows a downward loss curve."
    assert fp.kwargs.get("timeout", 0) >= 5            # a hard timeout is passed (anti-hang guard)
    content = fp.messages[0]["content"]
    kinds = {part["type"] for part in content}
    assert kinds == {"text", "image_url"}
    img = next(p for p in content if p["type"] == "image_url")["image_url"]["url"]
    assert img.startswith("data:image/png;base64,")

    class _Boom:
        def stream_chat(self, *a, **k):
            raise RuntimeError("vision down")

    assert fd._describe(_Boom(), "cap", b"x") == ""              # fail-safe -> empty
