"""
prompt_quality.py  --  Batch 1 (Critical Bug Fixes)

Fix vs original:
  BUG 2 -- Original source_score() boosted beamforming keywords and
           subtracted 4.0 from sources mentioning denoising /
           dereverberation / PESQ / STOI / AEC unless they also
           mentioned beamforming. This tanked evidence quality for
           every non-beamforming question.

           Original compact_evidence_body() only kept sentences that
           contained beamforming-related words, so for non-beamforming
           questions it returned essentially empty evidence.

This rewrite:
  - Extracts question terms from the actual user question.
  - Scores sources by question-term matches plus a general set of
    high-quality-evidence indicators (no negative penalty for any
    audio DSP subdomain).
  - compact_evidence_body() picks the highest-scoring sentences
    against the actual question, not a fixed keyword list, and
    preserves their original order.

Public API kept unchanged:
  enhance_result_prompt(result) -> dict
  enhance_prompt_files(prompt_path, context_path) -> dict
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
EXTRACTED_DIR = ROOT / "data" / "extracted"
LATEST_PROMPT = EXTRACTED_DIR / "latest_manual_prompt.txt"
LATEST_CONTEXT = EXTRACTED_DIR / "latest_context.txt"
ENHANCED_PROMPT = EXTRACTED_DIR / "latest_manual_prompt_enhanced.txt"
PROMPT_QUALITY_REPORT = EXTRACTED_DIR / "prompt_quality_report.json"


# Topic-agnostic stopwords for question term extraction.
_STOPWORDS = {
    "what", "which", "when", "where", "why", "how", "the", "this", "that",
    "would", "could", "should", "with", "from", "into", "your", "user",
    "question", "answer", "explain", "want", "remain", "across", "wide",
    "type", "give", "tell", "about", "and", "for", "are", "you", "any",
    "can", "use", "all", "but", "not", "have", "has", "does", "did",
    "between", "compare", "comparison", "best", "good", "better",
    "please", "kindly", "much", "many", "very", "really", "just",
}


# General "useful evidence" indicators. ALL audio DSP subdomains
# are represented here; nothing is penalised, everything is a
# small positive signal.
_GENERAL_USEFUL_KEYWORDS = [
    # Methods / structures
    "algorithm", "method", "approach", "technique", "model", "architecture",
    "pipeline", "framework", "system",
    # Math / structure
    "equation", "formula", "matrix", "vector", "covariance", "transform",
    # Performance / metrics
    "pesq", "stoi", "sdr", "snr", "si-sdr", "mos", "wer", "latency",
    "throughput", "accuracy", "performance", "complexity",
    # Audio fundamentals
    "frequency", "sample rate", "spectral", "spectrogram", "time-frequency",
    "magnitude", "phase", "amplitude", "signal", "noise", "speech",
    # Common DSP / AI subdomains (positive signals for all)
    "beamforming", "mvdr", "lcmv", "gsc", "doa", "microphone", "array",
    "speech enhancement", "denoising", "noise suppression",
    "dereverberation", "wpe", "echo cancellation", "aec",
    "deepfilternet", "rnnoise", "dnn", "cnn", "rnn", "transformer",
    "training", "inference", "real-time", "low-latency", "embedded",
    "voice activity", "source separation", "spectral subtraction",
    "wiener", "kalman", "lms", "rls",
]


def read_text(path: Path, default: str = "") -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        pass
    return default


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="ignore")


def clean_blob(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"!\[Image\]\(data:image/[^)]+\)", "[image omitted]", text, flags=re.IGNORECASE)
    text = re.sub(r"data:image/[A-Za-z0-9+/=;:,._-]{500,}", "[image omitted]", text)
    text = text.replace("<!-- formula-not-decoded -->", "[equation not decoded]")
    text = re.sub(r"#{8,}", "\n", text)
    text = re.sub(r"={8,}", "\n", text)
    text = re.sub(r"-{8,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_user_question(raw_prompt: str, raw_context: str) -> str:
    for text in [raw_prompt, raw_context]:
        m = re.search(r"USER QUESTION:\s*(.+?)(?:\n\n|RETRIEVED|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            q = clean_blob(m.group(1)).strip()
            if q:
                return q

        m = re.search(r"USER QUESTION\s*\n+(.+?)(?:\n\n|\nRETRIEVED|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            q = clean_blob(m.group(1)).strip()
            q = re.sub(r"^[:\- ]+", "", q).strip()
            if q:
                return q

    return "Answer the user's question using the retrieved uploaded-paper evidence."


def split_sources(text: str) -> List[Dict[str, Any]]:
    text = clean_blob(text)
    blocks: List[Dict[str, Any]] = []

    matches = list(re.finditer(r"\[SOURCE\s+(\d+)\]", text, flags=re.IGNORECASE))
    for i, match in enumerate(matches):
        source_no = int(match.group(1))
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()

        paper = ""
        section = ""
        pages = ""
        source_type = ""
        concepts = ""

        for line in block.splitlines()[:14]:
            lower = line.lower().strip()
            if lower.startswith("paper:"):
                paper = line.split(":", 1)[-1].strip()
            elif lower.startswith("section:"):
                section = line.split(":", 1)[-1].strip()
            elif lower.startswith("pages:"):
                pages = line.split(":", 1)[-1].strip()
            elif lower.startswith("type:"):
                source_type = line.split(":", 1)[-1].strip()
            elif lower.startswith("concepts:"):
                concepts = line.split(":", 1)[-1].strip()

        body_lines = []
        for line in block.splitlines():
            low = line.lower().strip()
            if low.startswith("[source"):
                continue
            if low.startswith(("topic:", "paper:", "section:", "pages:", "type:", "concepts:")):
                continue
            body_lines.append(line)

        body = clean_blob("\n".join(body_lines))

        blocks.append({
            "source_no": source_no,
            "paper": paper or "Unknown uploaded paper",
            "section": section or "Unknown section",
            "pages": pages or "Unknown pages",
            "type": source_type or "evidence",
            "concepts": concepts,
            "body": body,
        })

    return blocks


def question_terms(question: str) -> List[str]:
    """
    Extract topical content terms from the user's question.
    Topic-agnostic: not biased toward any audio DSP subdomain.

    Returns multi-word phrases first (highest signal), then acronyms,
    then single content words. Duplicates removed, original order kept.
    """
    q = (question or "").lower()

    phrase_patterns = [
        "noise suppression", "noise reduction", "speech enhancement",
        "real time", "real-time", "low latency", "low-latency",
        "acoustic echo cancellation", "direction of arrival",
        "deep filtering", "microphone array", "beamforming",
        "dereverberation", "speech quality", "speech intelligibility",
        "voice activity detection", "source separation",
        "spectral subtraction", "wiener filter", "kalman filter",
        "echo cancellation", "noise cancellation", "covariance matrix",
        "steering vector", "blocking matrix",
    ]
    phrases = [p for p in phrase_patterns if p in q]

    acronyms = [a.lower() for a in re.findall(r"\b[A-Z][A-Z0-9\-]{1,}\b", question or "")]

    words = re.findall(r"[a-z0-9][a-z0-9\-]{2,}", q)
    singles = [w for w in words if w not in _STOPWORDS]

    seen = set()
    out = []
    for term in phrases + acronyms + singles:
        if term not in seen:
            seen.add(term)
            out.append(term)

    return out


def source_score(source: Dict[str, Any], question: str, q_terms: List[str]) -> float:
    """
    Topic-agnostic source scoring. Uses extracted question terms
    plus a general 'useful evidence' signal list. No negative
    penalty for any audio DSP subdomain.
    """
    text = " ".join([
        str(source.get("paper", "")),
        str(source.get("section", "")),
        str(source.get("concepts", "")),
        str(source.get("body", "")),
    ]).lower()

    score = 0.0

    # Primary signal: question terms found in source body / metadata
    for t in q_terms:
        if t in text:
            score += 3.0 if " " in t else 2.0

    # Secondary signal: general "useful evidence" keywords (boost only)
    for keyword in _GENERAL_USEFUL_KEYWORDS:
        if keyword in text:
            score += 0.3

    # Chunk type bonuses (dense / structured evidence)
    stype = str(source.get("type", "")).lower()
    if "equation" in stype:
        score += 0.6
    if "algorithm" in stype:
        score += 0.6
    if "table" in stype or "metrics" in stype:
        score += 0.5

    # Section bonuses
    section = str(source.get("section", "")).lower()
    if any(s in section for s in ["method", "algorithm", "proposed", "system model", "approach"]):
        score += 0.5
    if any(s in section for s in ["results", "experiment", "evaluation"]):
        score += 0.4
    if any(s in section for s in ["abstract", "introduction"]):
        score += 0.2

    # Section penalties (skip non-evidence sections)
    if any(s in section for s in ["references", "acknowledgment", "acknowledgements", "appendix"]):
        score -= 5.0

    # Body length quality signals
    body = str(source.get("body", ""))
    if len(body) < 80:
        score -= 2.0
    if "[image omitted]" in body and len(body) > 2500:
        score -= 1.0

    return score


def select_sources(sources: List[Dict[str, Any]], question: str, max_sources: int = 10) -> List[Dict[str, Any]]:
    q_terms = question_terms(question)
    scored = []
    seen = set()

    for s in sources:
        key = (s.get("paper"), s.get("section"), s.get("pages"), str(s.get("body", ""))[:180])
        if key in seen:
            continue
        seen.add(key)
        s = dict(s)
        s["score"] = source_score(s, question, q_terms)
        scored.append(s)

    scored.sort(key=lambda x: x["score"], reverse=True)
    selected = [s for s in scored if s["score"] > 0][:max_sources]
    if len(selected) < min(4, len(scored)):
        selected = scored[:min(max_sources, len(scored))]
    return selected[:max_sources]


def compact_evidence_body(body: str, question: str, q_terms: List[str], max_chars: int = 1100) -> str:
    """
    Pick the highest-scoring sentences against the actual question
    (not a hardcoded keyword list), then restore their original order
    in the body. max_chars increased from 900 -> 1100 to preserve
    more context per source.
    """
    body = clean_blob(body)
    if len(body) <= max_chars:
        return body

    sentences = re.split(r"(?<=[.!?])\s+", body)

    scored_sents = []
    for sent in sentences:
        low = sent.lower()
        score = 0
        for t in q_terms:
            if t in low:
                score += 3 if " " in t else 2
        for k in _GENERAL_USEFUL_KEYWORDS:
            if k in low:
                score += 1
        if score > 0:
            scored_sents.append((score, sent.strip()))

    if not scored_sents:
        return body[:max_chars].rsplit(" ", 1)[0] + "..."

    # Take highest-scoring first, but pack in original document order
    scored_sents.sort(key=lambda x: -x[0])
    chosen = []
    total = 0
    for _, sent in scored_sents:
        if total + len(sent) + 1 > max_chars:
            continue
        chosen.append(sent)
        total += len(sent) + 1

    if not chosen:
        return body[:max_chars].rsplit(" ", 1)[0] + "..."

    # Sort chosen sentences by their original position in the body
    positions = {sent: body.find(sent) for sent in chosen}
    chosen.sort(key=lambda s: positions.get(s, 0))

    return " ".join(chosen)


def build_enhanced_prompt(question: str, sources: List[Dict[str, Any]]) -> str:
    q_terms = question_terms(question)
    evidence_lines = []
    source_map_lines = []

    for idx, s in enumerate(sources, 1):
        old_no = s.get("source_no", idx)
        citation = f"[SOURCE {idx}]"
        paper = s.get("paper", "Unknown uploaded paper")
        section = s.get("section", "Unknown section")
        pages = s.get("pages", "Unknown pages")
        concepts = s.get("concepts", "")
        body = compact_evidence_body(s.get("body", ""), question, q_terms)

        source_map_lines.append(f"- {citation} {paper} | {section} | pages {pages} | original source {old_no}")

        meta = f"{citation}\nPaper: {paper}\nSection: {section}\nPages: {pages}"
        if concepts:
            meta += f"\nConcepts: {concepts}"
        evidence_lines.append(meta + "\nEvidence:\n" + body.strip())

    evidence_pack = "\n\n".join(evidence_lines) if evidence_lines else "No retrieved evidence was available."
    source_map = "\n".join(source_map_lines) if source_map_lines else "- No source map available"

    prompt = f"""You are a high-end Audio DSP and AI research assistant.

Task:
Answer the user question using ONLY the retrieved uploaded-paper evidence below.

User question:
{question}

Important evidence rules:
- Use only the evidence in this prompt.
- Do not use outside/general knowledge as the source of truth.
- Cite every technical claim with [SOURCE n].
- If the evidence does not support a part of the answer, write: "The uploaded papers do not provide enough evidence for this part."
- If the user asks for "latest", answer as "latest among the uploaded papers", not latest in the world.
- Be direct, technical, and easy to understand.
- Do not mention retrieval routes, chunking, embeddings, BM25, vector search, or backend internals.

Output format:
1. Direct answer
2. Evidence-backed explanation
3. Practical interpretation
4. Limitations / missing evidence
5. Final recommendation

Source map:
{source_map}

Retrieved uploaded-paper evidence:
{evidence_pack}

Now write the final answer.
"""
    return prompt.strip() + "\n"


def enhance_prompt_files(prompt_path: Path = LATEST_PROMPT, context_path: Path = LATEST_CONTEXT) -> Dict[str, Any]:
    raw_prompt = read_text(prompt_path)
    raw_context = read_text(context_path)

    raw = raw_prompt + "\n\n" + raw_context
    question = extract_user_question(raw_prompt, raw_context)
    sources = split_sources(raw)
    selected = select_sources(sources, question, max_sources=int(os.getenv("ENHANCED_PROMPT_MAX_SOURCES", "10")))
    enhanced = build_enhanced_prompt(question, selected)

    if prompt_path.exists():
        backup = prompt_path.with_name(prompt_path.name + f".backup_prompt_quality_{int(time.time())}")
        try:
            backup.write_text(raw_prompt, encoding="utf-8", errors="ignore")
        except Exception:
            pass

    write_text(prompt_path, enhanced)
    write_text(ENHANCED_PROMPT, enhanced)

    report = {
        "question": question,
        "raw_source_count": len(sources),
        "selected_source_count": len(selected),
        "selected_sources": [
            {
                "new_source_no": i + 1,
                "original_source_no": s.get("source_no"),
                "paper": s.get("paper"),
                "section": s.get("section"),
                "pages": s.get("pages"),
                "score": round(float(s.get("score", 0)), 3),
            }
            for i, s in enumerate(selected)
        ],
        "enhanced_prompt_path": str(ENHANCED_PROMPT),
        "main_prompt_path": str(prompt_path),
    }

    PROMPT_QUALITY_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def enhance_result_prompt(result: Dict[str, Any] | None = None) -> Dict[str, Any]:
    result = dict(result or {})
    prompt_path = Path(result.get("manual_prompt_path") or LATEST_PROMPT)
    context_path = Path(result.get("context_path") or LATEST_CONTEXT)

    report = enhance_prompt_files(prompt_path, context_path)

    result["manual_prompt_path"] = str(prompt_path)
    result["enhanced_prompt_path"] = str(ENHANCED_PROMPT)
    result["prompt_quality_report"] = str(PROMPT_QUALITY_REPORT)
    result["selected_source_count"] = report.get("selected_source_count")
    return result


if __name__ == "__main__":
    report = enhance_prompt_files()
    print("PROMPT QUALITY COMPLETE")
    print("Question:", report["question"])
    print("Raw sources:", report["raw_source_count"])
    print("Selected sources:", report["selected_source_count"])
    print("Enhanced prompt:", report["enhanced_prompt_path"])
