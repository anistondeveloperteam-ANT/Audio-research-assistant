"""figure_describer.py — production figure understanding (ENABLE_FIGURE_UNDERSTANDING, default OFF).

A text-only RAG can't read charts/figures, only their captions. When enabled, each figure is rendered
and described by a MULTIMODAL LLM (FIGURE_MODEL, e.g. gemini-2.5-flash or pixtral-large-latest), and the
description is stored as a searchable 'figure' chunk embedded like any text.

Built for cost + quality (premium-production):
  - DESCRIBE-ONCE CACHE: descriptions are cached on disk keyed by a content hash of (rendered image +
    caption). A figure is sent to the LLM at most ONCE, ever — re-ingests / re-indexes reuse the cache
    for $0. This is what makes the recurring cost go away (the call is paid once, not every run).
  - GROUNDED PROMPT: the model gets the image PLUS the caption PLUS the paper's own sentences that
    reference that figure, so the description is accurate and doesn't invent values.
  - RATE-LIMIT RESILIENT: per-figure timeout + retry-with-backoff on 429, so a run reaches full coverage
    instead of silently skipping figures.
  - FAIL-SAFE + BOUNDED: any unrecoverable error skips that figure (paper still indexes); capped per
    paper; never raises into ingestion.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
_CACHE_FILE = ROOT / "data" / "extracted" / "figure_cache.json"

_FIG_RE = re.compile(r"(?:figure|fig)\.?\s*(\d+)\s*[:.]\s", re.I)
_CAPTION_CHARS = 280
_REFERENCE_CHARS = 600

_SYSTEM = (
    "You describe a figure from a research paper for SEARCH AND CITATION. In 2-4 sentences, state what "
    "the figure shows, its axes/variables, the main trend or comparison, and any specific values or "
    "findings a reader would cite. GROUND numbers in the caption and paper text provided; never invent "
    "values you cannot read. Output only the description, no preamble."
)


def enabled() -> bool:
    return (os.getenv("ENABLE_FIGURE_UNDERSTANDING", "false") or "false").strip().lower() in (
        "1", "true", "yes", "on")


def _cache_enabled() -> bool:
    return (os.getenv("FIGURE_CACHE", "true") or "true").strip().lower() not in ("0", "false", "no", "off")


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
    """Hard per-attempt timeout (s) so one stalled request can't hang ingest."""
    try:
        return max(5.0, float(os.getenv("FIGURE_DESCRIBE_TIMEOUT", "45")))
    except (TypeError, ValueError):
        return 45.0


def _max_retries() -> int:
    """Retries on a rate-limit (429) before giving up on a figure (skipped, retried on a later run)."""
    try:
        return max(0, int(os.getenv("FIGURE_DESCRIBE_RETRIES", "4")))
    except (TypeError, ValueError):
        return 4


# ---------------------------------------------------------------------------
# Describe-once cache (disk)
# ---------------------------------------------------------------------------
def _cache_key(png: bytes, caption: str) -> str:
    h = hashlib.sha256()
    h.update(png)
    h.update(b"\x00")
    h.update(caption.encode("utf-8", "ignore"))
    return h.hexdigest()


def _load_cache() -> Dict[str, str]:
    try:
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: Dict[str, str]) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Captions, reference text, rendering
# ---------------------------------------------------------------------------
def _figure_captions(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """[{page, num, caption}] for each distinct 'Figure N: ...' caption (scans full page text, so it
    survives PyMuPDF flattening the caption into a paragraph line)."""
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


def _reference_text(parsed: Dict[str, Any], num: str, limit: int = _REFERENCE_CHARS) -> str:
    """The paper's own sentences that mention 'Figure N' — authoritative context to ground the model."""
    pat = re.compile(rf"\b(?:figure|fig)\.?\s*{re.escape(num)}\b", re.I)
    out: List[str] = []
    for p in parsed.get("pages", []):
        for sent in re.split(r"(?<=[.!?])\s+", p.get("text") or ""):
            if pat.search(sent):
                s = re.sub(r"\s+", " ", sent).strip()
                if s and s not in out:
                    out.append(s)
    return " ".join(out)[:limit]


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


# ---------------------------------------------------------------------------
# Vision describe (grounded + retrying)
# ---------------------------------------------------------------------------
def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (exc.__class__.__name__ == "RateLimitError"
            or "429" in msg or "rate limit" in msg or "quota" in msg or "resource_exhausted" in msg)


def _describe(provider: Any, caption: str, reference: str, png: bytes) -> str:
    """Describe the figure from the image + caption + the paper's reference sentences. Retries on a
    rate-limit with backoff; '' on timeout / non-retryable error / exhausted retries (fail-safe)."""
    b64 = base64.b64encode(png).decode("ascii")
    prompt = f"Caption: {caption}\n"
    if reference:
        prompt += f"What the paper's text says about this figure: {reference}\n"
    prompt += "Describe the figure this caption refers to."
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    }]
    backoff = 2.0
    for attempt in range(_max_retries() + 1):
        try:
            parts: List[str] = []
            for tok in provider.stream_chat(messages, system=_SYSTEM, max_tokens=300, temperature=0.0,
                                            timeout=_describe_timeout()):
                if isinstance(tok, str):
                    parts.append(tok)
            return " ".join("".join(parts).split()).strip()
        except Exception as exc:                          # noqa: BLE001
            if _is_rate_limit(exc) and attempt < _max_retries():
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            return ""
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
    """Describe each figure (cache-first, grounded) and return them as 'figure' chunks. Returns [] when
    disabled / no vision provider / no figures / on any error — never raises."""
    if not enabled():
        return []
    captions = _figure_captions(parsed)[:_max_figures()]
    if not captions:
        return []
    provider = _vision_provider()
    if provider is None:
        return []
    from backend.ingestion.document_chunker import make_chunk

    cache = _load_cache() if _cache_enabled() else {}
    dirty = False
    parser = parsed.get("parser", "pymupdf")
    out: List[Dict[str, Any]] = []
    max_px = _render_max_px()
    for cap in captions:
        png = _render_page_png(Path(pdf_path), cap["page"], max_px)
        if not png:
            continue
        key = _cache_key(png, cap["caption"])
        desc = cache.get(key) if _cache_enabled() else None
        if not desc:                                       # cache miss -> the (paid) LLM call, once
            reference = _reference_text(parsed, cap["num"])
            desc = _describe(provider, cap["caption"], reference, png)
            if desc and _cache_enabled():
                cache[key] = desc
                dirty = True
        if not desc:
            continue
        chunk = make_chunk(f"{cap['caption']}\n{desc}", "Figure", cap["page"], cap["page"], parser=parser)
        chunk["chunk_type"] = "figure"
        out.append(chunk)
    if dirty:
        _save_cache(cache)
    return out
