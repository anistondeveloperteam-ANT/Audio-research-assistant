from pathlib import Path
import fitz

try:
    fitz.TOOLS.mupdf_display_errors(False)  # silence noisy "MuPDF error: ..." stderr spam
except Exception:
    pass


TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def pdf_pages_to_images(pdf_path: Path, out_dir: Path, dpi=220, max_pages=10):
    """
    Convert PDF pages to PNG images for OCR fallback.
    This is only used when normal PDF text extraction is weak.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    image_paths = []

    for i, page in enumerate(doc):
        if i >= max_pages:
            break

        pix = page.get_pixmap(dpi=dpi)
        out = out_dir / f"page_{i + 1}.png"
        pix.save(str(out))
        image_paths.append(out)

    return image_paths


_paddle_ocr = None


def _get_paddle():
    """Build (once) a PaddleOCR forced onto the CPU. PaddleOCR otherwise self-selects the GPU
    when paddlepaddle-gpu is installed (its own device config, separate from torch) — which we must
    NOT do here: OCR runs on CPU, the GPU is reserved for the reranker/embedder."""
    global _paddle_ocr
    if _paddle_ocr is None:
        from paddleocr import PaddleOCR
        try:
            _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)  # force CPU
        except TypeError:
            # Newer PaddleOCR dropped use_gpu in favour of `device`; pin to CPU there instead.
            try:
                _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", device="cpu")
            except TypeError:
                _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
    return _paddle_ocr


def ocr_with_paddle(image_paths):
    """
    PaddleOCR fallback (CPU). Good for scanned/image-heavy PDFs.
    """
    try:
        ocr = _get_paddle()
    except Exception as e:
        return "", f"PaddleOCR not available: {e}"

    try:
        texts = []

        for img in image_paths:
            result = ocr.ocr(str(img), cls=True)

            if not result:
                continue

            for page in result:
                if not page:
                    continue

                for line in page:
                    try:
                        texts.append(line[1][0])
                    except Exception:
                        pass

        return "\n".join(texts).strip(), None

    except Exception as e:
        return "", f"PaddleOCR failed: {e}"


def ocr_with_tesseract(image_paths):
    """
    Tesseract fallback.
    Requires Tesseract installed on Windows.
    """
    try:
        import pytesseract
        from PIL import Image

        # Direct path avoids Windows PATH problems.
        if Path(TESSERACT_EXE).exists():
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

    except Exception as e:
        return "", f"Tesseract wrapper not available: {e}"

    try:
        texts = []

        for img in image_paths:
            text = pytesseract.image_to_string(Image.open(img), lang="eng")
            texts.append(text)

        return "\n".join(texts).strip(), None

    except Exception as e:
        return "", f"Tesseract failed: {e}"


def ocr_pdf_fallback(pdf_path: Path, max_pages=10):
    """
    OCR fallback controller:
    1. Convert PDF pages to images.
    2. Try PaddleOCR.
    3. If weak/failed, try Tesseract.
    """
    out_dir = Path("data/extracted/ocr_cache") / pdf_path.stem
    image_paths = pdf_pages_to_images(pdf_path, out_dir, dpi=220, max_pages=max_pages)

    paddle_text, paddle_error = ocr_with_paddle(image_paths)

    if paddle_text and len(paddle_text.strip()) > 200:
        return {
            "engine": "paddleocr",
            "text": paddle_text,
            "error": None,
        }

    tess_text, tess_error = ocr_with_tesseract(image_paths)

    if tess_text and len(tess_text.strip()) > 200:
        return {
            "engine": "tesseract",
            "text": tess_text,
            "error": None,
        }

    return {
        "engine": "none",
        "text": "",
        "error": paddle_error or tess_error or "No OCR text extracted",
    }


def _render_page_image(pdf_path: Path, page_no: int, out_dir: Path, dpi: int = 200) -> Path:
    """Render ONE 1-based page to a PNG (only the pages that actually need OCR)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    try:
        pix = doc[page_no - 1].get_pixmap(dpi=dpi)
        out = out_dir / f"page_{page_no}.png"
        pix.save(str(out))
        return out
    finally:
        doc.close()


def _ocr_image_cpu(img_path: Path) -> str:
    """OCR a single image on the CPU: PaddleOCR (forced CPU) then Tesseract. '' if neither works."""
    text, _ = ocr_with_paddle([img_path])
    if text and len(text.strip()) > 20:
        return text.strip()
    text, _ = ocr_with_tesseract([img_path])
    return (text or "").strip()


def ocr_pages(pdf_path: Path, page_numbers, dpi: int = 200) -> dict:
    """OCR ONLY the given 1-based pages, on the CPU. Returns {page_no: text} for pages that yielded
    text. OCR engines are optional — an absent/failing engine just yields no text for that page (the
    caller then records it as a missing page), and one page's failure never aborts the rest."""
    wanted = sorted({int(p) for p in (page_numbers or [])})
    if not wanted:
        return {}
    out_dir = Path("data/extracted/ocr_cache") / pdf_path.stem
    result = {}
    for p in wanted:
        try:
            img = _render_page_image(pdf_path, p, out_dir, dpi=dpi)
        except Exception as e:
            print(f"  OCR: could not render page {p}: {str(e)[:120]}")
            continue
        try:
            text = _ocr_image_cpu(img)
        except Exception as e:
            print(f"  OCR: page {p} failed: {str(e)[:120]}")
            continue
        if text:
            result[p] = text
    return result


if __name__ == "__main__":
    pdfs = list(Path("data/papers").glob("*.pdf"))

    if not pdfs:
        print("No PDFs found in data/papers")
        raise SystemExit

    sample = pdfs[0]
    print("Testing OCR fallback on:", sample)

    result = ocr_pdf_fallback(sample, max_pages=2)

    print("Engine:", result["engine"])
    print("Error:", result["error"])
    print("\nPreview:")
    print(result["text"][:1200])