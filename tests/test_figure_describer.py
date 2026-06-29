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
    monkeypatch.setenv("FIGURE_CACHE", "false")                  # don't touch the real cache file
    monkeypatch.setattr(fd, "_vision_provider", lambda: object())
    monkeypatch.setattr(fd, "_render_page_png", lambda path, page, px, num=None: b"\x89PNG-bytes")
    monkeypatch.setattr(fd, "_describe",
                        lambda prov, cap, ref, png: "It shows loss falling from 2.1 to 0.4 over 50 epochs.")
    chunks = fd.figure_chunks(Path("x.pdf"), _PARSED)
    assert len(chunks) == 2
    c = chunks[0]
    assert c["section"] == "Figure" and c["chunk_type"] == "figure" and c["page_start"] == 2
    assert "Validation loss" in c["text"] and "falling from 2.1" in c["text"]


def test_figure_chunks_skips_unrenderable(monkeypatch):
    monkeypatch.setenv("ENABLE_FIGURE_UNDERSTANDING", "true")
    monkeypatch.setenv("FIGURE_CACHE", "false")
    monkeypatch.setattr(fd, "_vision_provider", lambda: object())
    monkeypatch.setattr(fd, "_render_page_png", lambda path, page, px, num=None: None)   # render fails
    monkeypatch.setattr(fd, "_describe", lambda prov, cap, ref, png: "desc")
    assert fd.figure_chunks(Path("x.pdf"), _PARSED) == []


def test_reference_text_pulls_the_papers_own_sentences():
    ref = fd._reference_text(_PARSED, "1")
    assert "Validation loss" in ref                              # the sentence mentioning 'Figure 1'
    assert fd._reference_text(_PARSED, "9") == ""                # no mention -> empty


def test_figure_chunks_uses_cache_and_skips_the_llm(monkeypatch):
    # A cached figure must NOT trigger an LLM call (this is the describe-once cost win).
    monkeypatch.setenv("ENABLE_FIGURE_UNDERSTANDING", "true")
    monkeypatch.setenv("FIGURE_CACHE", "true")
    monkeypatch.setattr(fd, "_vision_provider", lambda: object())
    monkeypatch.setattr(fd, "_render_page_png", lambda path, page, px, num=None: b"\x89PNG-bytes")
    caps = fd._figure_captions(_PARSED)
    cache = {fd._cache_key(b"\x89PNG-bytes", c["caption"]): f"cached desc for fig {c['num']}" for c in caps}
    monkeypatch.setattr(fd, "_load_cache", lambda: cache)
    monkeypatch.setattr(fd, "_save_cache", lambda c: None)

    def _boom(*a, **k):
        raise AssertionError("LLM was called despite a cache hit")

    monkeypatch.setattr(fd, "_describe", _boom)
    chunks = fd.figure_chunks(Path("x.pdf"), _PARSED)
    assert len(chunks) == 2
    assert "cached desc for fig 1" in chunks[0]["text"]


def test_describe_builds_vision_message_and_is_failsafe(monkeypatch):
    monkeypatch.setenv("FIGURE_DESCRIBE_RETRIES", "0")           # don't retry the failure case

    class _FakeVision:
        def __init__(self):
            self.messages = None
            self.kwargs = None

        def stream_chat(self, messages, system="", **k):
            self.messages = messages
            self.kwargs = k
            return ["The chart shows a downward loss curve."]

    fp = _FakeVision()
    out = fd._describe(fp, "Figure 1: loss vs epochs", "the paper says loss drops", b"PNGDATA")
    assert out == "The chart shows a downward loss curve."
    assert fp.kwargs.get("timeout", 0) >= 5            # a hard timeout is passed (anti-hang guard)
    content = fp.messages[0]["content"]
    kinds = {part["type"] for part in content}
    assert kinds == {"text", "image_url"}
    text_part = next(p for p in content if p["type"] == "text")["text"]
    assert "the paper says loss drops" in text_part    # reference text grounds the prompt
    img = next(p for p in content if p["type"] == "image_url")["image_url"]["url"]
    assert img.startswith("data:image/png;base64,")

    class _Boom:
        def stream_chat(self, *a, **k):
            raise RuntimeError("vision down")

    assert fd._describe(_Boom(), "cap", "ref", b"x") == ""       # fail-safe -> empty
