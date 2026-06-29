"""figure_describer.py — optional FIGURE UNDERSTANDING (ENABLE_FIGURE_UNDERSTANDING, default OFF).

A text-only RAG can't read charts/figures — only their captions. When enabled, each figure is rendered
to an image and described by the configured MULTIMODAL LLM (default Gemini 2.5 Flash — vision, free, and
CLOUD, so there is NO local VLM and NO extra GPU pressure on a small card). The description is stored as
a searchable 'figure' chunk and embedded like any other text, so a question about what a figure shows
now retrieves a real description instead of just the caption.

Design:
  - Captions ('Figure N: ...') and their page are detected from the parsed pages.
  - The page is rendered to a PNG (PyMuPDF) and sent, with the caption, to the vision model.
  - Each description becomes a chunk (section='Figure', chunk_type='figure') via make_chunk.
  - FAIL-SAFE everywhere: no vision provider / no key / render or LLM error -> that figure is skipped
    (or all are), and the paper still indexes normally. Never raises into ingestion.
  - Bounded: at most FIGURE_MAX_PER_PAPER figures/paper; render capped to FIGURE_RENDER_MAX_PX.
"""
from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# A figure caption anchor: 'Figure N:' / 'Fig. N.' anywhere in the page text (NOT line-anchored —
# PyMuPDF often flattens a caption into the middle of a paragraph line). The [:.] + following space
# is the caption signal that distinguishes 'Figure 1: ...' from a bare reference like 'see Figure 1'.
_FIG_RE = re.compile(r"(?:figure|fig)\.?\s*(\d+)\s*[:.]\s", re.I)
_CAPTION_CHARS = 280

_SYSTEM = (
    "You describe a figure from a research paper for SEARCH AND CITATION. In 2-4 sentences, state what "
    "the figure shows, its axes/variables, the main trend or comparison, and any specific values or "
    "findings a reader would cite. Be concrete and factual; never invent numbers you cannot read. "
    "Output only the description, no preamble."
)


def enabled() -> bool:
    return (os.getenv("ENABLE_FIGURE_UNDERSTANDING", "false") or "false").strip().lower() in (
        "1", "true", "yes", "on")


def _model() -> str:
    return (os.getenv("FIGURE_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash").strip()


def _max_figures() -> int:
    try:
        return max(1, int(os.getenv("FIGURE_MAX_PER_PAPER", "12")))
    except (TypeError, ValueError):
        return 12


def _render_max_px() -> int:
    try:
        return max(256, int(os.getenv("FIGURE_RENDER_MAX_PX", "1100")))
    except (TypeError, ValueError):
        return 1100


def _describe_timeout() -> float:
    """Hard per-figure timeout (seconds) for the vision call, so one stalled request can't hang the
    whole ingest (the failure mode this guards against). The figure is skipped on timeout."""
    try:
        return max(5.0, float(os.getenv("FIGURE_DESCRIBE_TIMEOUT", "45")))
    except (TypeError, ValueError):
        return 45.0


def _figure_captions(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """[{page, num, caption}] for each distinct 'Figure N: ...' caption found in the parsed pages."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for p in parsed.get("pages", []):
        text = p.get("text") or ""
        for m in _FIG_RE.finditer(text):
            num = m.group(1)
            if num in seen:
                continue
            seen.add(num)
            caption = re.sub(r"\s+", " ", text[m.start():m.start() + _CAPTION_CHARS]).strip()
            out.append({"page": int(p.get("page", 1)), "num": num, "caption": caption})
    return out


def _render_page_png(pdf_path: Path, page_no: int, max_px: int) -> Optional[bytes]:
    """Render one page (1-indexed) to PNG bytes, scaled so the longest side is ~max_px. None on error."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_no - 1]
            longest = max(page.rect.width, page.rect.height, 1.0)
            scale = min(max_px / longest, 2.0)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            return pix.tobytes("png")
        finally:
            doc.close()
    except Exception:
        return None


def _describe(provider: Any, caption: str, png: bytes) -> str:
    """Ask the vision model to describe the figure. '' on any error (fail-safe)."""
    b64 = base64.b64encode(png).decode("ascii")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"Caption: {caption}\nDescribe the figure this caption refers to."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    }]
    try:
        parts: List[str] = []
        for tok in provider.stream_chat(messages, system=_SYSTEM, max_tokens=300, temperature=0.0,
                                        timeout=_describe_timeout()):
            if isinstance(tok, str):
                parts.append(tok)
        return " ".join("".join(parts).split()).strip()
    except Exception:
        return ""


def _vision_provider():
    """A multimodal provider for FIGURE_MODEL, or None if unavailable (no key / text-only model)."""
    try:
        from backend.llm.streaming_provider import get_provider
        provider = get_provider(_model())
    except Exception:
        return None
    if provider is None or not getattr(provider, "is_available", False):
        return None
    return provider


def figure_chunks(pdf_path: Path, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Describe each figure via the vision model and return them as 'figure' chunks (make_chunk schema).
    Returns [] when disabled / no vision provider / no figures / on any error — never raises."""
    if not enabled():
        return []
    captions = _figure_captions(parsed)[:_max_figures()]
    if not captions:
        return []
    provider = _vision_provider()
    if provider is None:
        return []
    from backend.ingestion.document_chunker import make_chunk

    parser = parsed.get("parser", "pymupdf")
    out: List[Dict[str, Any]] = []
    max_px = _render_max_px()
    for cap in captions:
        png = _render_page_png(Path(pdf_path), cap["page"], max_px)
        if not png:
            continue
        desc = _describe(provider, cap["caption"], png)
        if not desc:
            continue
        chunk = make_chunk(f"{cap['caption']}\n{desc}", "Figure", cap["page"], cap["page"], parser=parser)
        chunk["chunk_type"] = "figure"
        out.append(chunk)
    return out
