import os
import re
import shutil
import subprocess
from pathlib import Path

import fitz
from ocr_fallback import ocr_pdf_fallback
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PARSER_MODE = os.getenv("PARSER_MODE", "auto").lower()
ENABLE_DOCLING = os.getenv("ENABLE_DOCLING", "true").lower() == "true"
ENABLE_MARKER = os.getenv("ENABLE_MARKER", "false").lower() == "true"

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


def parse_with_docling(pdf_path: Path):
    if not shutil.which("docling"):
        raise RuntimeError("Docling CLI not found. Install with: pip install docling")

    out_dir = PARSER_CACHE_DIR / safe_name(pdf_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docling",
        "--to", "md",
        "--output", str(out_dir),
        str(pdf_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1200,
        env={
            **os.environ,
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TRANSFORMERS_VERBOSITY": "error",
            "TOKENIZERS_PARALLELISM": "false",
        },
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)

    md_files = list(out_dir.glob("*.md"))

    if not md_files:
        raise RuntimeError("Docling did not create markdown output.")

    md_text = md_files[0].read_text(encoding="utf-8", errors="ignore")
    md_text = clean_text(md_text)

    return {
        "parser": "docling",
        "pages": [
            {
                "page": 1,
                "text": md_text,
                "parser": "docling",
            }
        ],
        "page_count": estimate_page_count(pdf_path),
        "raw_markdown": md_text,
        "tables": extract_markdown_tables(md_text),
        "equations": extract_equation_blocks(md_text),
    }

def parse_with_marker(pdf_path: Path):
    # Deep parser mode using marker-pdf.
    # Your marker_single CLI expects: marker_single [OPTIONS] FPATH
    if not ENABLE_MARKER:
        raise RuntimeError('Marker disabled. Set ENABLE_MARKER=true in .env')

    marker_cmd = shutil.which('marker_single')
    if not marker_cmd:
        raise RuntimeError('marker_single command not found. Install with: pip install marker-pdf')

    out_dir = PARSER_CACHE_DIR / (safe_name(pdf_path) + '_marker')
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_abs = str(pdf_path.resolve())
    out_abs = str(out_dir.resolve())

    command_variants = [
        [marker_cmd, pdf_abs, '--output_dir', out_abs],
        [marker_cmd, pdf_abs, '--output_dir', out_abs, '--output_format', 'markdown'],
        [marker_cmd, pdf_abs, '--output', out_abs],
        [marker_cmd, '--output_dir', out_abs, pdf_abs],
        [marker_cmd, '--output_format', 'markdown', '--output_dir', out_abs, pdf_abs],
        [marker_cmd, pdf_abs],
    ]

    last_error = ''
    generated_files = []

    for cmd in command_variants:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                cwd=str(out_dir),
                env={
                    **os.environ,
                    'HF_HUB_DISABLE_SYMLINKS_WARNING': '1',
                    'HF_HUB_DISABLE_PROGRESS_BARS': '1',
                    'HF_HUB_DISABLE_TELEMETRY': '1',
                    'TRANSFORMERS_VERBOSITY': 'error',
                    'TOKENIZERS_PARALLELISM': 'false',
                },
            )

            if result.returncode != 0:
                last_error = (result.stderr or result.stdout or '')[:3000]
                continue

            generated_files = (
                list(out_dir.rglob('*.md')) +
                list(out_dir.rglob('*.markdown')) +
                list(out_dir.rglob('*.txt'))
            )

            if generated_files:
                break

            last_error = 'Marker succeeded but no markdown/txt output was found. stdout=' + result.stdout[:1000] + ' stderr=' + result.stderr[:1000]

        except Exception as exc:
            last_error = str(exc)

    if not generated_files:
        manual_cmd = 'marker_single "' + pdf_abs + '" --output_dir "' + out_abs + '"'
        message = (
            'Marker did not create markdown output.' + chr(10) +
            'Output folder: ' + str(out_dir) + chr(10) +
            'Last error: ' + str(last_error) + chr(10) + chr(10) +
            'Manual test command:' + chr(10) + manual_cmd
        )
        raise RuntimeError(message)

    best_file = max(generated_files, key=lambda p: p.stat().st_size)
    md_text = best_file.read_text(encoding='utf-8', errors='ignore')
    md_text = clean_text(md_text)

    return {
        'parser': 'marker',
        'pages': [
            {
                'page': 1,
                'text': md_text,
                'parser': 'marker',
            }
        ],
        'page_count': estimate_page_count(pdf_path),
        'raw_markdown': md_text,
        'tables': extract_markdown_tables(md_text),
        'equations': extract_equation_blocks(md_text),
    }

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


def parse_pdf(pdf_path: Path):
    mode = PARSER_MODE

    parsed = None

    if mode == "pymupdf":
        parsed = parse_with_pymupdf(pdf_path)

    elif mode == "docling":
        parsed = parse_with_docling(pdf_path)

    elif mode == "marker":
        parsed = parse_with_marker(pdf_path)

    elif mode == "auto":
        if ENABLE_DOCLING:
            try:
                parsed = parse_with_docling(pdf_path)
            except Exception as e:
                print(f"Docling failed for {pdf_path.name}. Falling back to PyMuPDF.")
                print(str(e)[:500])

        if parsed is None:
            parsed = parse_with_pymupdf(pdf_path)

    else:
        parsed = parse_with_pymupdf(pdf_path)

    # OCR fallback only if normal extraction is weak.
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