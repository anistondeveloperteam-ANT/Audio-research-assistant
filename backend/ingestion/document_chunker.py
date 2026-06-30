"""
advanced_chunker.py  --  Batch 4 (Chunker Improvements)

Only THIS file changes in Batch 4. The improvements:

  H9.1 SECTION_PATTERNS expanded from 17 to ~45 entries.
       Adds Network Architecture, Loss Function, Training Details,
       Ablation Study, Datasets, Implementation Details, Future Work,
       Limitations, System Overview, Proposed Algorithm, etc.

  H9.2 detect_section() now case-insensitive and tolerates leading
       'The'/'A'/'An' plus Roman / arabic / parenthesised numbering.
       Matches more real-world section headings.

  H9.3 extract_figure_captions() finds 'Figure N: ...', 'Fig. N: ...',
       'Table N: ...' captions and makes them their own chunks. In
       research papers these captions often pack method names AND specific
       metric values, so they're high-signal evidence for retrieval.

  H9.4 extract_algorithm_blocks() preserves 'Algorithm N: ...' blocks
       as single chunks so sentence-splitting can't fragment them.

  Lower min thresholds:
       MIN_SENTENCE_CHARS:  20 -> 10
       MIN_CHUNK_CHARS:    200 -> 150
       Short technical statements like 'Set alpha = 0.5.' now survive.

  KEY_CONCEPTS covers domain-neutral terms across ML, math/stats,
  systems, and evaluation (general research, not field-specific).

Backward compatible:
  - chunk dict schema unchanged (same keys)
  - chunk_type values unchanged (so Batch 1's chunk_type_boost still
    fires on equation / algorithm / table_or_metrics)
  - chunk_parsed_document(parsed) signature unchanged

To take effect on existing papers: re-ingest with `python pipeline.py`.
"""

import os
import re
from dotenv import load_dotenv

from backend.ingestion.semantic_chunker import semantic_split

load_dotenv()

CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1800"))
CHUNK_OVERLAP_SENTENCES = int(os.getenv("CHUNK_OVERLAP_SENTENCES", "2"))
MIN_SENTENCE_CHARS = int(os.getenv("MIN_SENTENCE_CHARS", "10"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "150"))
# bge-large embeds at most ~512 tokens (~2000 chars); a bigger table chunk gets silently truncated, so
# the tail of a large table never enters the vector. Split long tables into header-preserving row groups.
TABLE_CHUNK_MAX_CHARS = int(os.getenv("TABLE_CHUNK_MAX_CHARS", "1600"))


SECTION_PATTERNS = [
    # Standard front matter
    "abstract", "introduction", "related work", "background",
    "prior work", "literature review",
    # Problem / model formulation
    "signal model", "signal processing pipeline",
    "problem formulation", "problem statement", "system overview",
    "system model",
    # Proposed approach variants
    "proposed method", "proposed approach", "proposed algorithm",
    "proposed model", "proposed system", "proposed framework",
    "method", "methods", "methodology", "approach",
    # Model architecture variants (very common in DNN papers)
    "model", "model architecture", "network architecture",
    "architecture", "design",
    # Algorithm / loss / training
    "algorithm", "algorithms",
    "loss function", "training loss", "objective function",
    "training", "training details", "training procedure",
    "training setup", "inference",
    # Engineering details
    "implementation", "implementation details",
    "datasets", "dataset", "data", "experimental data",
    # Experiments / results
    "experimental setup", "experiments", "experimental results",
    "results", "results and discussion", "evaluation",
    "evaluation setup", "evaluation methodology",
    "ablation", "ablation study", "ablation studies",
    "discussion", "analysis",
    # Closing
    "limitations", "future work", "future directions",
    "conclusion", "conclusions", "concluding remarks",
    "references", "appendix",
    "acknowledgment", "acknowledgement",
    "acknowledgments", "acknowledgements",
]


# Domain-neutral technical terms used to tag chunks with the high-signal "concepts" they
# mention (stored in the chunks.concept_tags column, weighted in keyword/BM25 search). General
# across research domains -- ML, math/stats, systems, and evaluation -- not tied to any field.
KEY_CONCEPTS = [
    # Machine learning / models
    "neural network", "deep learning", "transformer", "self-attention", "attention",
    "convolutional", "CNN", "RNN", "LSTM", "GRU", "encoder", "decoder", "embedding",
    "fine-tuning", "pretraining", "transfer learning", "reinforcement learning",
    "supervised", "unsupervised", "self-supervised", "diffusion model", "GAN",
    # Training / optimization
    "gradient descent", "backpropagation", "optimizer", "learning rate",
    "regularization", "dropout", "batch normalization", "loss function",
    "cross-entropy", "overfitting", "hyperparameter",
    # Math / statistics
    "probability", "distribution", "Bayesian", "variance", "covariance matrix",
    "eigenvalue", "matrix", "vector", "optimization", "convex", "gradient",
    "regression", "classification", "clustering", "dimensionality reduction", "PCA",
    # Methods / systems
    "algorithm", "complexity", "heuristic", "approximation", "parallel",
    "distributed", "scalability", "latency", "throughput", "cache",
    # Evaluation
    "benchmark", "baseline", "ablation", "accuracy", "precision", "recall",
    "cross-validation", "dataset", "ground truth", "state-of-the-art", "evaluation metric",
]


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_section_line(line: str) -> str:
    """Normalize a candidate section header for matching."""
    line_clean = line.strip().lower()
    # Strip numbering: roman, arabic, parenthesised
    line_clean = re.sub(r"^[ivx]+\.\s+", "", line_clean)
    line_clean = re.sub(r"^\d+(\.\d+)*\s*\.?\s+", "", line_clean)
    line_clean = re.sub(r"^\(\d+\)\s+", "", line_clean)
    # Strip leading article
    line_clean = re.sub(r"^(the|a|an)\s+", "", line_clean)
    # Strip trailing punctuation / hash / dash
    line_clean = line_clean.strip(":.-# ")
    return line_clean


# Sort patterns longest-first so 'training details' beats 'training' and
# 'implementation details' beats 'implementation' during prefix matching.
_SECTION_PATTERNS_SORTED = sorted(SECTION_PATTERNS, key=len, reverse=True)


def detect_section(line: str):
    if not line:
        return None
    line_clean = _normalize_section_line(line)
    if len(line_clean) > 80 or len(line_clean) < 3:
        return None

    for sec in _SECTION_PATTERNS_SORTED:
        if line_clean == sec or line_clean.startswith(sec + " "):
            return sec.title()
    return None


def split_sentences(text: str):
    text = clean_text(text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= MIN_SENTENCE_CHARS]


def detect_concepts(text: str):
    low = text.lower()
    found = []
    for concept in KEY_CONCEPTS:
        if concept.lower() in low:
            found.append(concept)
    return sorted(set(found))


def has_equation(text: str) -> int:
    """A real equation — a numbered equation tag like '(3)' at a line end, or LaTeX/math operators.
    NOT the mere presence of '=' or common words like 'matrix'/'vector', which appear throughout prose
    and previously over-tagged ordinary text as equations."""
    if re.search(r"\(\d+(?:\.\d+)?\)\s*$", text, re.MULTILINE):
        return 1
    low = text.lower()
    math_markers = ["\\sum", "\\frac", "\\int", "\\prod", "\\nabla", "\\partial",
                    "\\alpha", "\\beta", "\\theta", "argmin", "arg min", "argmax", "arg max"]
    return int(any(m in low for m in math_markers))


def has_table(text: str) -> int:
    """A real table — a pipe/markdown grid or a numeric tabular block — NOT merely the word 'table' or
    a metric name (PESQ/SNR/…) mentioned in prose, which previously tagged ~1/3 of ALL chunks as tables
    and made the retrieval chunk-type boost misfire."""
    if text.count("|") >= 8:                       # markdown/pipe grid: >= ~2 rows x several columns
        return 1
    numeric_rows = sum(1 for ln in text.splitlines()
                       if len(ln) < 120 and len(re.findall(r"\b\d+(?:\.\d+)?\b", ln)) >= 4)
    return 1 if numeric_rows >= 3 else 0           # a dense numeric block (results grid as text)


def has_algorithm(text: str) -> int:
    """A real algorithm/pseudocode block — an 'Algorithm N' header, the word 'pseudocode', or paired
    Input:/Output: markers. NOT common words like 'training'/'inference'/'procedure' in prose. (Genuine
    'Algorithm N:' blocks are also force-tagged separately by extract_algorithm_blocks.)"""
    low = text.lower()
    if re.search(r"\balgorithm\s+\d+", low):
        return 1
    if "pseudocode" in low:
        return 1
    return int("input:" in low and "output:" in low)


def classify_chunk(text: str):
    if has_table(text):
        return "table_or_metrics"
    if has_algorithm(text):
        return "algorithm"
    if has_equation(text):
        return "equation"
    return "text"


def make_chunk(text, section, page_start, page_end, parser="unknown"):
    return {
        "section": section,
        "text": clean_text(text),
        "page_start": page_start,
        "page_end": page_end,
        "chunk_type": classify_chunk(text),
        "has_equation": has_equation(text),
        "has_algorithm": has_algorithm(text),
        "has_table": has_table(text),
        "concepts": detect_concepts(text),
        "parser": parser,
    }


def chunk_text(text: str, section="Unknown", page_start=1, page_end=1, parser="unknown"):
    # Topic-coherent split (SEMANTIC_CHUNKER=true) — boundaries follow meaning, not a fixed char
    # budget. Returns None when off/unavailable, so we fall through to the legacy packer below.
    segments = semantic_split(text)
    if segments:
        out = [make_chunk(seg, section, page_start, page_end, parser)
               for seg in segments if len(seg) >= MIN_CHUNK_CHARS]
        if out:
            return out

    sentences = split_sentences(text)
    chunks = []
    current = []
    for sentence in sentences:
        candidate = " ".join(current + [sentence])
        if len(candidate) > CHUNK_MAX_CHARS and current:
            chunk_body = " ".join(current).strip()
            chunks.append(make_chunk(chunk_body, section, page_start, page_end, parser))
            current = current[-CHUNK_OVERLAP_SENTENCES:] if len(current) > CHUNK_OVERLAP_SENTENCES else current[:]
        current.append(sentence)
    if current:
        chunk_body = " ".join(current).strip()
        if len(chunk_body) >= MIN_CHUNK_CHARS:
            chunks.append(make_chunk(chunk_body, section, page_start, page_end, parser))
    return chunks


def split_by_markdown_sections(text: str):
    sections = []
    current_name = "Unknown"
    current_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if current_lines:
                sections.append((current_name, "\n".join(current_lines)))
                current_lines = []
            header = stripped.strip("#").strip() or "Unknown"
            # Normalize to canonical section name if we recognize it
            detected = detect_section(header)
            current_name = detected if detected else header
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_name, "\n".join(current_lines)))
    if not sections:
        sections.append(("Unknown", text))
    return sections


# ======================================================================
# NEW in Batch 4: figure / table caption extraction
# ======================================================================

_FIGURE_CAPTION_RE = re.compile(
    r"^\s*(?:Figure|Fig\.?|Table)\s*\d+[:.]\s*(.{15,})",
    re.IGNORECASE,
)


def extract_figure_captions(text: str, page_start: int, page_end: int, parser: str):
    """Find Figure / Table captions and make them their own chunks."""
    if not text:
        return []
    captions = []
    seen = set()
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not _FIGURE_CAPTION_RE.match(line):
            i += 1
            continue

        # Capture this line plus up to 2 continuation lines until blank
        # line or next caption header.
        caption_lines = [line]
        j = 1
        while j <= 3 and (i + j) < len(lines):
            nxt = lines[i + j]
            if not nxt.strip():
                break
            if _FIGURE_CAPTION_RE.match(nxt):
                break
            caption_lines.append(nxt)
            j += 1

        caption_text = "\n".join(caption_lines).strip()
        key = caption_text[:120].lower()
        if key not in seen and len(caption_text) >= 40:
            seen.add(key)
            section = "Figure caption"
            if re.match(r"^\s*table", caption_text, re.IGNORECASE):
                section = "Table caption"
            captions.append(
                make_chunk(caption_text, section, page_start, page_end, parser)
            )
        i += len(caption_lines)
    return captions


# ======================================================================
# NEW in Batch 4: algorithm block preservation
# ======================================================================

_ALGORITHM_HEADER_RE = re.compile(
    r"^\s*Algorithm\s+\d+[:.]?\s*.*",
    re.IGNORECASE,
)


def extract_algorithm_blocks(text: str, page_start: int, page_end: int, parser: str):
    """
    Find 'Algorithm N: ...' headers and capture the following lines
    until a blank line or another section header, keeping the block
    intact as one chunk.
    """
    if not text:
        return []
    blocks = []
    seen = set()
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not _ALGORITHM_HEADER_RE.match(line):
            i += 1
            continue

        block_lines = [line]
        j = 1
        while (i + j) < len(lines) and j < 40:
            nxt = lines[i + j]
            if not nxt.strip():
                break
            if _ALGORITHM_HEADER_RE.match(nxt):
                break
            # Stop at a clear new section header line
            if detect_section(nxt):
                break
            block_lines.append(nxt)
            j += 1

        block_text = "\n".join(block_lines).strip()
        key = block_text[:120].lower()
        if key not in seen and len(block_text) >= 60:
            seen.add(key)
            chunk = make_chunk(block_text, "Algorithm block", page_start, page_end, parser)
            # Force classification (these are pseudo-code by definition)
            chunk["chunk_type"] = "algorithm"
            chunk["has_algorithm"] = 1
            blocks.append(chunk)
        i += len(block_lines)
    return blocks


def _split_markdown_table(md: str, max_chars: int = TABLE_CHUNK_MAX_CHARS) -> list:
    """Split a long markdown table into row groups that each fit the embedder's window, repeating the
    header row(s) in every group so each piece stays self-describing. Short tables pass through as one."""
    md = (md or "").strip()
    if len(md) <= max_chars:
        return [md] if md else []
    lines = md.split("\n")
    header, body_start = [], 0
    if lines and "|" in lines[0]:
        header.append(lines[0])
        body_start = 1
        # a markdown separator row is only |, -, :, space -> keep it as part of the header
        if len(lines) > 1 and set(lines[1].replace("|", "").replace("-", "").replace(":", "").strip()) == set():
            header.append(lines[1])
            body_start = 2
    head_len = sum(len(line) + 1 for line in header)
    groups, cur, cur_len = [], list(header), head_len
    for line in lines[body_start:]:
        if len(line) > max_chars:                   # a single giant row can't be split by rows
            line = line[:max_chars]                 # hard-cap it so no chunk exceeds the embed window
        if cur_len + len(line) + 1 > max_chars and len(cur) > len(header):
            groups.append("\n".join(cur))
            cur, cur_len = list(header), head_len
        cur.append(line)
        cur_len += len(line) + 1
    if len(cur) > len(header):
        groups.append("\n".join(cur))
    return groups or [md]


# ======================================================================
# Main entry point
# ======================================================================

def chunk_parsed_document(parsed):
    parser = parsed.get("parser", "unknown")
    page_count = parsed.get("page_count", 1)
    all_chunks = []

    if parsed.get("raw_markdown"):
        full_text = parsed["raw_markdown"]
        sections = split_by_markdown_sections(full_text)
        for section, section_text in sections:
            all_chunks.extend(
                chunk_text(section_text, section=section,
                           page_start=1, page_end=page_count, parser=parser)
            )
        # Figure / table captions and algorithm blocks from full markdown
        all_chunks.extend(extract_figure_captions(full_text, 1, page_count, parser))
        all_chunks.extend(extract_algorithm_blocks(full_text, 1, page_count, parser))

    else:
        pages = parsed["pages"]
        current_section = "Unknown"
        section_buffer = []
        start_page = 1
        last_page = 1

        def flush():
            nonlocal section_buffer  # the other names are only read, not reassigned here
            section_text = "\n".join(section_buffer).strip()
            if section_text:
                all_chunks.extend(
                    chunk_text(section_text, current_section,
                               start_page, last_page, parser=parser)
                )
                all_chunks.extend(extract_figure_captions(
                    section_text, start_page, last_page, parser))
                all_chunks.extend(extract_algorithm_blocks(
                    section_text, start_page, last_page, parser))
            section_buffer = []

        for page in pages:
            page_num = page["page"]
            lines = page["text"].splitlines()
            for line in lines:
                sec = detect_section(line)
                if sec and section_buffer:
                    flush()
                    current_section = sec
                    start_page = page_num
                if sec:
                    current_section = sec
                section_buffer.append(line)
                last_page = page_num
        if section_buffer:
            flush()

    # Parser-level extracted tables / equations (kept from original). Long tables are split into
    # header-preserving row groups so each piece fits the embedder's window (no silent truncation).
    for table in parsed.get("tables", []):
        for part in _split_markdown_table(table):
            all_chunks.append(make_chunk(part, "Table", 1, page_count, parser=parser))
    for equation in parsed.get("equations", []):
        all_chunks.append(make_chunk(equation, "Equation", 1, page_count, parser=parser))

    # Final dedupe by (section, first 200 chars of text)
    seen = set()
    deduped = []
    for ch in all_chunks:
        sig = (ch.get("section", ""), ch.get("text", "")[:200])
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(ch)

    return deduped
