import os
import re
from pathlib import Path

import fitz
from backend.ingestion.ocr_fallback import ocr_pdf_fallback
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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


def safe_name(path: Path) -> str:
    name = path.stem
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    return name[:120]


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


def build_ocr_parsed_result(pdf_path: Path, ocr_result):
    return {
        "parser": f"ocr_{ocr_result['engine']}",
        "pages": [
            {
                "page": 1,
                "text": ocr_result.get("text") or "",
                "parser": f"ocr_{ocr_result['engine']}",
            }
        ],
        "page_count": estimate_page_count(pdf_path),
        "raw_markdown": "",
        "tables": [],
        "equations": [],
    }


_docling_converter = None


def _get_docling_converter():
    """Lazily build and cache the Docling converter (loads layout models once),
    on the GPU when DEVICE allows."""
    global _docling_converter
    if _docling_converter is None:
        from docling.document_converter import DocumentConverter
        try:
            from docling.document_converter import PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice,
            )
            want = (os.getenv("DEVICE", "auto") or "auto").strip().lower()
            device = AcceleratorDevice.CPU if want == "cpu" else AcceleratorDevice.CUDA
            opts = PdfPipelineOptions()
            opts.accelerator_options = AcceleratorOptions(num_threads=8, device=device)
            _docling_converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            )
        except Exception:
            # Older/different Docling API — fall back to its default (auto device).
            _docling_converter = DocumentConverter()
    return _docling_converter


def parse_with_docling(pdf_path: Path):
    """Parse via Docling's Python API (layout, tables, reading order)."""
    converter = _get_docling_converter()
    result = converter.convert(str(pdf_path))
    md_text = clean_text(result.document.export_to_markdown())
    if not md_text.strip():
        raise RuntimeError("Docling produced empty output.")

    return {
        "parser": "docling",
        "pages": [{"page": 1, "text": md_text, "parser": "docling"}],
        "page_count": estimate_page_count(pdf_path),
        "raw_markdown": md_text,
        "tables": extract_markdown_tables(md_text),
        "equations": extract_equation_blocks(md_text),
    }


def parse_pdf(pdf_path: Path):
    # Best-quality parser first: Docling (layout, tables, reading order).
    # Falls back to fast PyMuPDF only if Docling errors, so ingestion never fails.
    parsed = None
    try:
        parsed = parse_with_docling(pdf_path)
    except Exception as e:
        print(f"Docling failed for {pdf_path.name}; falling back to PyMuPDF. {str(e)[:300]}")

    if parsed is None:
        parsed = parse_with_pymupdf(pdf_path)

    # OCR fallback only if normal extraction is weak (scanned / image-only PDF).
    if total_text_length(parsed) < 500:
        print(f"Weak extracted text detected for {pdf_path.name}. Trying OCR fallback...")
        ocr_result = ocr_pdf_fallback(pdf_path, max_pages=10)

        if ocr_result.get("text") and len(ocr_result["text"]) > total_text_length(parsed):
            print(f"OCR fallback used: {ocr_result['engine']}")
            parsed = build_ocr_parsed_result(pdf_path, ocr_result)
        else:
            print("OCR fallback did not improve extraction.")

    return parsed


if __name__ == "__main__":
    pdfs = list(Path("data/papers").glob("*.pdf"))

    if not pdfs:
        print("No PDF found in data/papers")
        raise SystemExit

    sample = pdfs[0]
    print("Testing parser on:", sample)

    result = parse_pdf(sample)

    print("Parser used:", result["parser"])
    print("Pages:", result["page_count"])
    print("Tables detected:", len(result["tables"]))
    print("Equations detected:", len(result["equations"]))
    print("\nText preview:\n")
    print(result["pages"][0]["text"][:1500])