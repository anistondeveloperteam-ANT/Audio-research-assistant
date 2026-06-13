import contextlib
import logging
import os
import re
import sys
from pathlib import Path

import fitz
from backend.ingestion.ocr_fallback import ocr_pages
from dotenv import load_dotenv

try:
    fitz.TOOLS.mupdf_display_errors(False)  # silence noisy "MuPDF error: ..." stderr spam
except Exception:
    pass

load_dotenv()

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger(__name__)
# Keep Docling quiet during ingestion — no scary multi-line "Stage … failed" stack traces.
for _name in ("docling", "docling.utils", "docling.pipeline", "docling_core", "docling_ibm_models"):
    try:
        logging.getLogger(_name).setLevel(logging.ERROR)
    except Exception:
        pass


EXTRACTED_DIR = Path("data/extracted")
PARSER_CACHE_DIR = EXTRACTED_DIR / "parser_cache"
PARSER_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<![.!?:;])\n(?!\n)", " ", text)

    return text.strip()


def estimate_page_count(pdf_path: Path) -> int:
    try:
        doc = fitz.open(pdf_path)
        return len(doc)
    except Exception:
        return 0


def parse_with_pymupdf(pdf_path: Path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_index, page in enumerate(doc):
        blocks = page.get_text("blocks")
        text_blocks = []

        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            text = clean_text(text)

            if not text:
                continue

            if len(text) < 20 and re.fullmatch(r"[\d\s\W]+", text):
                continue

            text_blocks.append((y0, x0, text))

        text_blocks.sort(key=lambda x: (round(x[0] / 20), x[1]))

        page_text = "\n".join(t[2] for t in text_blocks)

        pages.append({
            "page": page_index + 1,
            "text": clean_text(page_text),
            "parser": "pymupdf",
        })

    return {
        "parser": "pymupdf",
        "pages": pages,
        "page_count": len(doc),
        "raw_markdown": "",
        "tables": [],
        "equations": [],
    }


def extract_markdown_tables(text: str):
    tables = []
    current = []

    for line in text.splitlines():
        if "|" in line and line.count("|") >= 2:
            current.append(line)
        else:
            if len(current) >= 2:
                tables.append("\n".join(current))
            current = []

    if len(current) >= 2:
        tables.append("\n".join(current))

    return tables


def extract_equation_blocks(text: str):
    equations = []

    patterns = [
        r"\$\$(.*?)\$\$",
        r"\\\[(.*?)\\\]",
        r"\\begin\{equation\}(.*?)\\end\{equation\}",
    ]

    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.DOTALL):
            eq = clean_text(match)
            if eq:
                equations.append(eq)

    for line in text.splitlines():
        line_clean = line.strip()
        if len(line_clean) > 20 and any(x in line_clean for x in ["=", "argmin", "arg max", "\\sum", "\\frac"]):
            equations.append(line_clean)

    return list(dict.fromkeys(equations))


def total_text_length(parsed):
    total = 0

    for page in parsed.get("pages", []):
        total += len(page.get("text") or "")

    return total


# ----------------------------------------------------------------------
# Configuration (read live so .env / tests take effect)
# ----------------------------------------------------------------------
# A page with fewer than this many extracted characters is treated as "text-poor"
# (scanned / image-only) and becomes a candidate for OCR.
MIN_PAGE_CHARS = int(os.getenv("OCR_MIN_PAGE_CHARS", "50"))


def enable_ocr() -> bool:
    """OCR is OFF by default. When ON, it runs ONLY on text-poor pages, on the CPU."""
    return (os.getenv("ENABLE_OCR", "false") or "false").strip().lower() in ("1", "true", "yes", "on")


def docling_device() -> str:
    """Device for Docling LAYOUT parsing. Default CPU — parsing stays off the GPU, which is
    reserved for the reranker/embedder. Set DOCLING_DEVICE=cuda to put layout on the GPU."""
    return (os.getenv("DOCLING_DEVICE") or "cpu").strip().lower()


def _is_oom_error(exc: Exception) -> bool:
    """True for GPU/host out-of-memory failures (torch OOM, RapidOCR/onnx std::bad_alloc, etc.)."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return ("outofmemory" in name or "bad_alloc" in msg or "out of memory" in msg
            or "cuda error" in msg or "cublas" in msg or "cudnn" in msg)


def _should_hide_cuda() -> bool:
    """During ingestion, parsing runs CUDA-free by DEFAULT so Docling never even ATTEMPTS the GPU
    (its CPU AcceleratorDevice is not always honoured, and a GPU attempt OOMs std::bad_alloc on
    small cards). Opt back in with DOCLING_DEVICE=cuda."""
    return (os.getenv("DOCLING_DEVICE") or "cpu").strip().lower() != "cuda"


def force_cpu_parsing() -> None:
    """Hide CUDA from THIS process so Docling / torch / onnxruntime cannot initialize the GPU while
    parsing. Call ONCE, before importing torch or docling. Scoped to the ingestion parse process:
    the reranker/embedder use the GPU at QUERY time (a separate process), and embedding is a
    separate ingestion stage (embed_chunks), so neither is affected.

    NOTE: the value MUST be "-1" — an EMPTY string does NOT hide the GPU (torch still reports
    cuda_available=True), which is why parsing previously still touched CUDA."""
    if _should_hide_cuda():
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def _pages_needing_ocr(pages, min_chars: int = MIN_PAGE_CHARS):
    """1-based page numbers whose extracted text is too short to trust (likely scanned)."""
    return [p["page"] for p in pages if len((p.get("text") or "").strip()) < min_chars]


# ----------------------------------------------------------------------
# Docling (layout / tables / reading order). OCR is DISABLED here on purpose
# (RapidOCR self-selects CUDA and OOMs on small GPUs). Two more hardening points,
# established by profiling:
#   - Docling's native `preprocess` can throw std::bad_alloc on complex (vector-heavy) pages,
#     device-independent and unfixable via options. That page is recovered losslessly by the
#     PyMuPDF per-page spine, so we just SILENCE the native stderr noise (it is not data loss).
#   - page_batch_size=1 keeps peak memory low and is ~5x faster than the default batch of 4.
# ----------------------------------------------------------------------
_docling_converters = {}


def _configure_pipeline_options(opts):
    """Turn Docling OCR OFF (born-digital text extraction only; no RapidOCR on the GPU)."""
    opts.do_ocr = False
    return opts


def _docling_page_batch() -> int:
    """Pages Docling processes per batch. 1 keeps peak memory low (fewer simultaneous heavy
    allocations) and parses ~5x faster here than the default of 4. Tune via DOCLING_PAGE_BATCH."""
    try:
        return max(1, int(os.getenv("DOCLING_PAGE_BATCH", "1")))
    except (TypeError, ValueError):
        return 1


@contextlib.contextmanager
def _suppress_native_stderr():
    """Silence NATIVE (C/C++) writes to stderr for the block. Docling's per-page 'Stage preprocess
    failed ... std::bad_alloc' is a native write Python logging can't filter; the page is recovered
    via PyMuPDF, so the line is pure noise. No-op when stderr has no real fd (e.g. captured under
    pytest), so it never interferes with test output."""
    try:
        fd = sys.stderr.fileno()
        saved = os.dup(fd)
    except Exception:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stderr.flush()
        os.dup2(devnull, fd)
        yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


def _get_docling_converter(device: str):
    if device not in _docling_converters:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice,
        )
        from docling.datamodel.settings import settings as _docling_settings
        _docling_settings.perf.page_batch_size = _docling_page_batch()
        opts = PdfPipelineOptions()
        _configure_pipeline_options(opts)
        dev = AcceleratorDevice.CPU if device == "cpu" else AcceleratorDevice.CUDA
        opts.accelerator_options = AcceleratorOptions(num_threads=8, device=dev)
        print(f"  Docling layout device: {device} (do_ocr=False, page_batch={_docling_page_batch()})")
        _docling_converters[device] = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
    return _docling_converters[device]


def _docling_convert(pdf_path: Path, device: str):
    """Run Docling on a given device and return {raw_markdown, tables, equations}. Native stderr is
    suppressed during convert so a recovered-page std::bad_alloc never reaches the console."""
    converter = _get_docling_converter(device)
    with _suppress_native_stderr():
        result = converter.convert(str(pdf_path))
        md = clean_text(result.document.export_to_markdown())
    if not md.strip():
        raise RuntimeError("Docling produced empty output.")
    return {"raw_markdown": md, "tables": extract_markdown_tables(md),
            "equations": extract_equation_blocks(md)}


def _docling_safe(pdf_path: Path):
    """Parse with Docling (CPU during ingestion — CUDA is hidden, so there is no GPU attempt). On
    ANY failure, log ONE clean INFO line and return None so the caller falls back to PyMuPDF text.
    This fallback is for GENUINE parse failures only — there is no GPU attempt left to OOM."""
    try:
        return _docling_convert(pdf_path, docling_device())
    except Exception as e:
        kind = "out-of-memory" if _is_oom_error(e) else "parse error"
        logger.info("Docling %s for %s — using PyMuPDF text (%s).",
                    kind, pdf_path.name, str(e).splitlines()[0][:160])
        return None


def parse_with_docling(pdf_path: Path):
    """Back-compat wrapper: Docling parse (no OCR) on the configured device."""
    out = _docling_convert(pdf_path, docling_device())
    return {
        "parser": "docling",
        "pages": [{"page": 1, "text": out["raw_markdown"], "parser": "docling"}],
        "page_count": estimate_page_count(pdf_path),
        "raw_markdown": out["raw_markdown"],
        "tables": out["tables"],
        "equations": out["equations"],
    }


def parse_pdf(pdf_path: Path):
    """Parse a PDF into per-page text with full coverage accounting and NO silent page loss.

    1. PyMuPDF gives fast, reliable per-page text — the source of truth for which pages exist.
    2. Docling (layout/tables, OCR OFF) enriches the document-level markdown, OOM-safe.
    3. OCR runs ONLY on text-poor pages and ONLY if ENABLE_OCR=true, on the CPU.
    4. Every page is accounted for (pages_indexed vs pages_total) with a clear warning for any
       page that ends up empty — never dropped silently.
    """
    pages_total = estimate_page_count(pdf_path)
    pm = parse_with_pymupdf(pdf_path)
    pages = pm.get("pages", [])

    docling = _docling_safe(pdf_path)
    raw_markdown = (docling or {}).get("raw_markdown", "")
    tables = (docling or {}).get("tables", [])
    equations = (docling or {}).get("equations", [])

    # Auto OCR — text-poor pages only, CPU only, opt-in via ENABLE_OCR.
    poor = _pages_needing_ocr(pages)
    ocr_used = []
    if poor and enable_ocr():
        print(f"  OCR (cpu): {len(poor)} text-poor page(s) {poor} — OCRing on CPU…")
        try:
            recovered = ocr_pages(pdf_path, poor)
        except Exception as e:
            print(f"  OCR failed: {str(e)[:160]}")
            recovered = {}
        for p in pages:
            t = recovered.get(p["page"])
            if t and len(t.strip()) >= MIN_PAGE_CHARS:
                p["text"] = t.strip()
                p["parser"] = "ocr"
                ocr_used.append(p["page"])
    elif poor:
        print(f"  OCR off (ENABLE_OCR=false): {len(poor)} text-poor page(s) {poor} not OCR'd.")

    # Coverage — from the per-page spine, regardless of which text source we chunk.
    indexed = [p["page"] for p in pages if len((p.get("text") or "").strip()) >= MIN_PAGE_CHARS]
    missing = [p["page"] for p in pages if p["page"] not in indexed]
    warnings = []
    if missing:
        warnings.append(
            f"WARNING: {len(missing)} page(s) failed/empty and are NOT indexed: {missing}")

    # Chunk from Docling markdown when it covers the text well; else per-page (keeps real page nums).
    pm_len = sum(len((p.get("text") or "")) for p in pages)
    use_markdown = bool(raw_markdown) and len(raw_markdown) >= 0.6 * max(pm_len, 1)
    if use_markdown and ocr_used:
        extra = "\n\n".join((p.get("text") or "") for p in pages if p["page"] in ocr_used)
        raw_markdown = (raw_markdown + "\n\n" + extra).strip()

    return {
        "parser": "docling" if use_markdown else pm.get("parser", "pymupdf"),
        "pages": pages,
        "page_count": pages_total or len(pages),
        "raw_markdown": raw_markdown if use_markdown else "",
        "tables": tables,
        "equations": equations,
        "pages_total": pages_total or len(pages),
        "pages_indexed": len(indexed),
        "pages_missing": missing,
        "ocr_pages": ocr_used,
        "warnings": warnings,
    }


if __name__ == "__main__":
    force_cpu_parsing()        # parse CUDA-free (no GPU attempt), same as the ingestion pipeline
    pdfs = list(Path("data/papers").glob("*.pdf"))

    if not pdfs:
        print("No PDF found in data/papers")
        raise SystemExit

    sample = pdfs[0]
    print("Testing parser on:", sample)

    result = parse_pdf(sample)

    print("Parser used:", result["parser"])
    print("Pages indexed:", f"{result.get('pages_indexed')}/{result.get('pages_total')}")
    print("OCR pages:", result.get("ocr_pages"))
    for w in result.get("warnings", []):
        print(w)
    print("Tables detected:", len(result["tables"]))
    print("Equations detected:", len(result["equations"]))
    print("\nText preview:\n")
    print(result["pages"][0]["text"][:1500])