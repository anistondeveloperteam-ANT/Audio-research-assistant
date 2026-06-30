"""
Corpus coverage report (RAGFlow-inspired): what's actually in the index — papers, chunks,
per-paper breakdown, chunk-type / section / topic distribution, duplicate papers, failed
ingestions, and under-represented topics. Use it to add PDFs that BROADEN coverage instead
of randomly inflating chunk count.

    python -m backend.evaluation.corpus_report              # full report (md + json + summary)
    python -m backend.evaluation.corpus_report --inspect    # list papers + chunk counts
    python -m backend.evaluation.corpus_report --inspect 3  # dump the chunks of paper id 3

Also available as `python pipeline.py --corpus-report`.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "extracted"
OUT_JSON = OUT_DIR / "corpus_coverage_report.json"
OUT_MD = OUT_DIR / "corpus_coverage_report.md"

# Broad topic map (deliberately NOT audio-only) — the goal is wide searchable coverage across
# audio, speech, signal processing, ML, evaluation, datasets, and research methods.
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "speech enhancement / denoising": ["speech enhancement", "denois", "noise reduction", "noise suppression"],
    "beamforming / mic arrays": ["beamform", "mvdr", "lcmv", "gsc", "microphone array", "spatial filter", "doa", "direction of arrival"],
    "source separation": ["source separation", "blind source", " ica ", "permutation invariant", "pit "],
    "speech recognition (ASR)": ["speech recognition", " asr", "acoustic model", " ctc", "transducer", "wav2vec"],
    "speaker / diarization": ["speaker", "diariz", "verification", "x-vector", "d-vector"],
    "TTS / vocoder / synthesis": ["text-to-speech", " tts", "vocoder", "synthesis", "wavenet", "melgan"],
    "evaluation metrics": ["pesq", "stoi", " sdr", "si-sdr", " mos", " wer", " cer", "perceptual quality"],
    "deep learning architectures": ["transformer", "attention", "convolutional", " cnn", " rnn", "lstm", " gru", "u-net", "diffusion", " gan"],
    "datasets / corpora": ["dataset", "corpus", "librispeech", "wsj", "voxceleb", "dns challenge", "musan", "wham"],
    "signal processing / features": ["stft", "fourier", "spectrogram", "mfcc", "mel-", "filter bank", "cepstr", "windowing"],
    "ML training / optimization": ["training", "loss function", "optimiz", "gradient", "regulariz", "augmentation", "fine-tun"],
    "self-supervised / representation": ["self-supervised", "representation learning", "pretrain", "contrastive", "embedding"],
}


def _connect():
    from backend.retrieval.vector_retriever import connect
    return connect()


def _read(v) -> str:
    from backend.retrieval.vector_retriever import read_lob
    return (read_lob(v) or "") if v is not None else ""


def gather() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(papers_with_chunk_counts, chunks). Reads CLOBs eagerly before the connection closes."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT p.id, p.title, p.page_count, COUNT(c.id) "
        "FROM papers p LEFT JOIN chunks c ON c.paper_id = p.id "
        "GROUP BY p.id, p.title, p.page_count ORDER BY p.id")
    papers = [{"id": r[0], "title": _read(r[1]).strip(), "page_count": r[2] or 0,
               "chunk_count": r[3] or 0} for r in cur.fetchall()]
    cur.execute(
        "SELECT c.paper_id, c.chunk_type, c.section_name, c.concept_tags, c.chunk_text "
        "FROM chunks c")
    chunks = [{"paper_id": r[0], "chunk_type": (r[1] or "text"),
               "section": _read(r[2]).strip() or "Unknown",
               "concepts": _read(r[3]), "text": _read(r[4])} for r in cur.fetchall()]
    cur.close()
    conn.close()
    return papers, chunks


def _topic_hits(text: str) -> List[str]:
    t = " " + text.lower() + " "
    return [topic for topic, kws in TOPIC_KEYWORDS.items() if any(k in t for k in kws)]


def build_report() -> Dict[str, Any]:
    papers, chunks = gather()
    n_papers, n_chunks = len(papers), len(chunks)

    by_paper_types: Dict[Any, Counter] = {}
    for c in chunks:
        by_paper_types.setdefault(c["paper_id"], Counter())[c["chunk_type"]] += 1

    topic_chunks: Counter = Counter()
    topic_papers: Dict[str, set] = {t: set() for t in TOPIC_KEYWORDS}
    for c in chunks:
        blob = f"{c['concepts']} {c['section']} {c['text'][:400]}"
        for topic in _topic_hits(blob):
            topic_chunks[topic] += 1
            topic_papers[topic].add(c["paper_id"])

    title_counts = Counter(p["title"].lower().strip() for p in papers if p["title"])
    duplicates = sorted(t for t, n in title_counts.items() if n > 1)
    failed = [p for p in papers if p["chunk_count"] == 0]
    missing_topics = [t for t in TOPIC_KEYWORDS if not topic_papers[t]]
    weak_topics = [t for t in TOPIC_KEYWORDS if 0 < len(topic_papers[t]) <= 1]

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "totals": {"papers": n_papers, "chunks": n_chunks,
                   "avg_chunks_per_paper": round(n_chunks / n_papers, 1) if n_papers else 0,
                   "topics_covered": sum(1 for t in TOPIC_KEYWORDS if topic_papers[t]),
                   "topics_total": len(TOPIC_KEYWORDS)},
        "papers": [{**p, "chunk_types": dict(by_paper_types.get(p["id"], {}))} for p in papers],
        "chunk_type_distribution": dict(Counter(c["chunk_type"] for c in chunks)),
        "section_distribution": dict(Counter(c["section"] for c in chunks).most_common(20)),
        "topic_coverage": {t: {"papers": len(topic_papers[t]), "chunks": topic_chunks[t]}
                           for t in TOPIC_KEYWORDS},
        "duplicate_titles": duplicates,
        "failed_ingestions": [{"id": p["id"], "title": p["title"]} for p in failed],
        "missing_topics": missing_topics,
        "underrepresented_topics": weak_topics,
    }


def to_markdown(rep: Dict[str, Any]) -> str:
    t = rep["totals"]
    L = ["# Corpus Coverage Report", "", f"_{rep['generated_at']}_", "",
         f"- **Papers:** {t['papers']}", f"- **Chunks:** {t['chunks']}",
         f"- **Avg chunks/paper:** {t['avg_chunks_per_paper']}",
         f"- **Topic coverage:** {t['topics_covered']}/{t['topics_total']} broad domains have >=1 paper", ""]
    if rep["failed_ingestions"]:
        L += ["## :warning: Failed ingestions (0 chunks)", ""]
        L += [f"- [{p['id']}] {p['title']}" for p in rep["failed_ingestions"]] + [""]
    if rep["duplicate_titles"]:
        L += ["## :warning: Duplicate paper titles", ""] + [f"- {d}" for d in rep["duplicate_titles"]] + [""]
    L += ["## Per-paper", "", "| id | title | pages | chunks | types |", "|---|---|---|---|---|"]
    for p in rep["papers"]:
        types = ", ".join(f"{k}:{v}" for k, v in p["chunk_types"].items())
        L.append(f"| {p['id']} | {p['title'][:60]} | {p['page_count']} | {p['chunk_count']} | {types} |")
    L += ["", "## Chunk-type distribution", ""]
    L += [f"- {k}: {v}" for k, v in rep["chunk_type_distribution"].items()]
    L += ["", "## Topic coverage (broad domains)", "", "| topic | papers | chunks |", "|---|---|---|"]
    for topic, d in rep["topic_coverage"].items():
        flag = " :x:" if d["papers"] == 0 else ""
        L.append(f"| {topic}{flag} | {d['papers']} | {d['chunks']} |")
    if rep["missing_topics"]:
        L += ["", "## :x: Missing topics — add PDFs here to broaden coverage", ""]
        L += [f"- {x}" for x in rep["missing_topics"]]
    if rep["underrepresented_topics"]:
        L += ["", "## :warning: Under-represented topics (only 1 paper)", ""]
        L += [f"- {x}" for x in rep["underrepresented_topics"]]
    return "\n".join(L) + "\n"


def run_report() -> Dict[str, Any]:
    rep = build_report()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    OUT_MD.write_text(to_markdown(rep), encoding="utf-8")
    t = rep["totals"]
    print(f"Corpus: {t['papers']} papers, {t['chunks']} chunks ({t['avg_chunks_per_paper']}/paper)")
    print(f"Topic coverage: {t['topics_covered']}/{t['topics_total']} broad domains have >=1 paper")
    if rep["missing_topics"]:
        more = "..." if len(rep["missing_topics"]) > 6 else ""
        print(f"Missing topics ({len(rep['missing_topics'])}): " + ", ".join(rep["missing_topics"][:6]) + more)
    if rep["failed_ingestions"]:
        print(f"WARNING: {len(rep['failed_ingestions'])} paper(s) ingested with 0 chunks")
    if rep["duplicate_titles"]:
        print(f"WARNING: {len(rep['duplicate_titles'])} duplicate title(s)")
    print(f"Report written: {OUT_MD}")
    return rep


def inspect(paper_id: int | None = None) -> None:
    papers, chunks = gather()
    if paper_id is None:
        for p in papers:
            print(f"[{p['id']}] {p['title'][:70]} — {p['chunk_count']} chunks")
        return
    pcs = [c for c in chunks if c["paper_id"] == paper_id]
    title = next((p["title"] for p in papers if p["id"] == paper_id), "?")
    print(f"Paper [{paper_id}] {title} — {len(pcs)} chunks\n")
    for i, c in enumerate(pcs, 1):
        print(f"--- chunk {i} | type={c['chunk_type']} | section={c['section']} ---")
        print((c["text"] or "").strip()[:400])
        print()


def main(argv: List[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--inspect":
        inspect(int(args[1]) if len(args) > 1 else None)
        return 0
    run_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
