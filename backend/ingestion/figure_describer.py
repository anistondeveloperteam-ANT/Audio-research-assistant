"""figure_describer.py — production figure understanding (ENABLE_FIGURE_UNDERSTANDING, default OFF).

A text-only RAG can't read charts/figures, only captions. When enabled, each figure is cropped to its
region and described by a DEDICATED, SELF-HOSTED vision model — default Qwen2-VL-2B (native transformers,
no remote-code fragility, ~4.8 GB VRAM so it fits a 6 GB card, free, no API/quota). The description is
stored as a searchable 'figure' chunk and embedded like any text.

Pipeline (premium-production, all local + free):
  - CROP to the figure region (PyMuPDF block layout, pdffigures2-style) -> a clean focused image.
  - DESCRIBE-ONCE CACHE: cached on disk by content hash of (cropped image + caption); each figure hits
    the model at most ONCE ever, so re-ingests / re-indexes are free.
  - GROUNDED PROMPT: the model gets the image + caption + the paper's own sentences about that figure.
  - FAIL-SAFE: model can't load (e.g. GPU busy) / any error -> that figure is skipped (paper still
    indexes). Run figure description as a batch job when the GPU is free (the cache makes it one-time).

`FIGURE_BACKEND=local` (default) uses the local VLM; `=llm` uses a cloud multimodal provider
(FIGURE_MODEL) for those who prefer it. Both share the crop + cache + grounding.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
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


def _backend() -> str:
    return (os.getenv("FIGURE_BACKEND", "local") or "local").strip().lower()


def _cache_enabled() -> bool:
    return (os.getenv("FIGURE_CACHE", "true") or "true").strip().lower() not in ("0", "false", "no", "off")


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


# ---- local VLM (default backend) ----
def _vlm_model_id() -> str:
    return (os.getenv("FIGURE_VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct") or "Qwen/Qwen2-VL-2B-Instruct").strip()


def _vlm_max_pixels() -> int:
    """Cap on vision tokens (px) — bounds VRAM so a 2B VLM fits a 6 GB card."""
    try:
        return max(64 * 28 * 28, int(os.getenv("FIGURE_VLM_MAX_PIXELS", str(900 * 28 * 28))))
    except (TypeError, ValueError):
        return 900 * 28 * 28


def _vlm_max_tokens() -> int:
    try:
        return max(64, int(os.getenv("FIGURE_VLM_MAX_TOKENS", "220")))
    except (TypeError, ValueError):
        return 220


# ---- cloud LLM (optional backend) ----
def _model() -> str:
    return (os.getenv("FIGURE_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash").strip()


def _describe_timeout() -> float:
    try:
        return max(5.0, float(os.getenv("FIGURE_DESCRIBE_TIMEOUT", "45")))
    except (TypeError, ValueError):
        return 45.0


def _max_retries() -> int:
    try:
        return max(0, int(os.getenv("FIGURE_DESCRIBE_RETRIES", "4")))
    except (TypeError, ValueError):
        return 4


# ---------------------------------------------------------------------------
# Describe-once cache
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
# Captions, reference text, region crop, render
# ---------------------------------------------------------------------------
def _figure_captions(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
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
    pat = re.compile(rf"\b(?:figure|fig)\.?\s*{re.escape(num)}\b", re.I)
    out: List[str] = []
    for p in parsed.get("pages", []):
        for sent in re.split(r"(?<=[.!?])\s+", p.get("text") or ""):
            if pat.search(sent):
                s = re.sub(r"\s+", " ", sent).strip()
                if s and s not in out:
                    out.append(s)
    return " ".join(out)[:limit]


def _figure_region(page, num: str):
    """Bounding box of the figure for 'Figure N' (visual content above its caption), from the page's
    block layout. None when it can't be located confidently (caller renders the whole page)."""
    import fitz
    try:
        blocks = page.get_text("dict").get("blocks", [])
    except Exception:
        return None
    capre = re.compile(rf"^\s*(?:figure|fig)\.?\s*{re.escape(str(num))}\b", re.I)
    cap = None
    for b in blocks:
        if b.get("type") != 0:
            continue
        txt = "".join(s.get("text", "") for ln in b.get("lines", []) for s in ln.get("spans", []))
        if capre.match(txt.strip()):
            cap = fitz.Rect(b["bbox"])
            break
    if cap is None:
        return None
    ph = page.rect.height
    vis = [fitz.Rect(b["bbox"]) for b in blocks if b.get("type") == 1]
    try:
        for dr in page.get_drawings():
            r = dr.get("rect")
            if r and r.width > 24 and r.height > 24:
                vis.append(fitz.Rect(r))
    except Exception:
        pass
    above = [r for r in vis if r.y1 <= cap.y0 + 4 and 0 < (cap.y0 - r.y0) < ph * 0.78
             and r.height > 30 and r.width > 40]
    if not above:
        return None
    x0 = min(r.x0 for r in above)
    y0 = min(r.y0 for r in above)
    x1 = max(r.x1 for r in above)
    fig = fitz.Rect(x0, y0, x1, cap.y1) + (-8, -8, 8, 8)
    if fig.width < page.rect.width * 0.15 or fig.height < ph * 0.08:
        return None
    return fig & page.rect


def _render_page_png(pdf_path: Path, page_no: int, max_px: int, num: Optional[str] = None) -> Optional[bytes]:
    try:
        import fitz
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_no - 1]
            clip = _figure_region(page, num) if num is not None else None
            box = clip or page.rect
            scale = min(max_px / max(box.width, box.height, 1.0), 2.0)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip)
            return pix.tobytes("png")
        finally:
            doc.close()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
def _build_prompt(caption: str, reference: str) -> str:
    p = f"Caption: {caption}\n" if caption else ""
    if reference:
        p += f"What the paper's text says about this figure: {reference}\n"
    p += ("Describe the figure this caption refers to: what it shows, the axes/variables, the main trend "
          "or comparison, and any specific values. Be concise and factual; do not invent values.")
    return p


# ---------------------------------------------------------------------------
# Backend: local VLM (Qwen2-VL by default)
# ---------------------------------------------------------------------------
_VLM = None
_VLM_TRIED = False
_VLM_LOCK = threading.Lock()


def _local_vlm():
    """Load the local vision model ONCE (cached). Returns (model, processor, device) or None on failure
    (e.g. GPU out of memory because the server holds it) so the caller fails safe."""
    global _VLM, _VLM_TRIED
    with _VLM_LOCK:
        if _VLM_TRIED:
            return _VLM
        _VLM_TRIED = True
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if dev == "cuda" else torch.float32
            model = Qwen2VLForConditionalGeneration.from_pretrained(_vlm_model_id(), torch_dtype=dtype).to(dev).eval()
            proc = AutoProcessor.from_pretrained(_vlm_model_id(),
                                                 min_pixels=256 * 28 * 28, max_pixels=_vlm_max_pixels())
            _VLM = (model, proc, dev)
        except Exception:
            _VLM = None
        return _VLM


def _describe_local(caption: str, reference: str, png: bytes) -> str:
    """Describe a figure with the local VLM. '' on any error (fail-safe). Frees CUDA memory after EVERY
    figure (success or failure) so memory can't accumulate across a long batch and degrade later figures
    — the cause of caption-rich papers getting no descriptions mid-run."""
    vlm = _local_vlm()
    if vlm is None:
        return ""
    model, proc, dev = vlm
    import torch                                            # before try so `finally` can clear the cache
    try:
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(png)).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": _SYSTEM + "\n\n" + _build_prompt(caption, reference)}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=_vlm_max_tokens(), do_sample=False)
        gen = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return " ".join(proc.batch_decode(gen, skip_special_tokens=True)[0].split()).strip()
    except Exception:
        return ""
    finally:
        try:
            if dev == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Backend: cloud LLM (optional, FIGURE_BACKEND=llm)
# ---------------------------------------------------------------------------
def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (exc.__class__.__name__ == "RateLimitError"
            or "429" in msg or "rate limit" in msg or "quota" in msg or "resource_exhausted" in msg)


def _describe_llm(provider: Any, caption: str, reference: str, png: bytes) -> str:
    b64 = base64.b64encode(png).decode("ascii")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": _build_prompt(caption, reference)},
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
    try:
        from backend.llm.streaming_provider import get_provider
        provider = get_provider(_model())
    except Exception:
        return None
    if provider is None or not getattr(provider, "is_available", False):
        return None
    return provider


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def figure_chunks(pdf_path: Path, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Describe each figure (crop -> cache-first -> local VLM, grounded) and return 'figure' chunks.
    Returns [] when disabled / backend unavailable / no figures / on any error — never raises."""
    if not enabled():
        return []
    captions = _figure_captions(parsed)[:_max_figures()]
    if not captions:
        return []
    backend = _backend()
    provider = _vision_provider() if backend == "llm" else None
    if backend == "llm" and provider is None:
        return []
    from backend.ingestion.document_chunker import make_chunk

    cache = _load_cache() if _cache_enabled() else {}
    dirty = False
    parser = parsed.get("parser", "pymupdf")
    out: List[Dict[str, Any]] = []
    max_px = _render_max_px()
    for cap in captions:
        png = _render_page_png(Path(pdf_path), cap["page"], max_px, cap["num"])
        if not png:
            continue
        key = _cache_key(png, cap["caption"])
        desc = cache.get(key) if _cache_enabled() else None
        if not desc:
            reference = _reference_text(parsed, cap["num"])
            if backend == "llm":
                desc = _describe_llm(provider, cap["caption"], reference, png)
            else:
                desc = _describe_local(cap["caption"], reference, png)
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
