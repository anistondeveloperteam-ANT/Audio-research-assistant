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


def ocr_with_paddle(image_paths):
    """
    PaddleOCR fallback.
    Good for scanned/image-heavy PDFs.
    """
    try:
        from paddleocr import PaddleOCR
    except Exception as e:
        return "", f"PaddleOCR not available: {e}"

    try:
        ocr = PaddleOCR(use_angle_cls=True, lang="en")
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