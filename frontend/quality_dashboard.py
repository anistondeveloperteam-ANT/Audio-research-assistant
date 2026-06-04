
"""
quality_dashboard.py
Market/product quality dashboard for Audio Research Paper Assistant.

Run:
    cd /d C:\AI\audio-research-assistant
    .venv\Scripts\activate
    streamlit run frontend\quality_dashboard.py

Purpose:
- Shows retrieval evaluation score, time, weak questions, and top sources.
- Lets you run evaluation from UI.
- No manual JSON/code editing required.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
REPORT_FILE = ROOT / "data" / "extracted" / "retrieval_eval_report.json"


st.set_page_config(
    page_title="Audio RAG Quality Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; color: #f8fafc; }
[data-testid="stSidebar"] { background: #171923; }
.block-container { max-width: 1250px; padding-top: 2rem; }
.card {
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.04);
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.good { color: #22c55e; font-weight: 700; }
.warn { color: #f59e0b; font-weight: 700; }
.bad { color: #ef4444; font-weight: 700; }
.small { color: #9ca3af; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)


def run_eval() -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", "backend.evaluation.evaluate_retrieval"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    return result.returncode == 0, output


def load_report() -> Dict[str, Any]:
    if not REPORT_FILE.exists():
        return {}
    try:
        return json.loads(REPORT_FILE.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def status_label(score: float) -> str:
    if score >= 0.90:
        return "Excellent"
    if score >= 0.80:
        return "Good"
    if score >= 0.65:
        return "Needs tuning"
    return "Weak"


def status_class(score: float) -> str:
    if score >= 0.80:
        return "good"
    if score >= 0.65:
        return "warn"
    return "bad"


st.title("📊 Audio Research Assistant Quality Dashboard")
st.caption("Use this after indexing papers to measure retrieval quality and find weak areas before improving the model.")

col_a, col_b = st.columns([1, 3])

with col_a:
    if st.button("Run Evaluation", type="primary", use_container_width=True):
        with st.status("Running retrieval evaluation...", expanded=False) as status:
            ok, output = run_eval()
            if ok:
                status.update(label="Evaluation complete", state="complete")
                st.success("Evaluation completed.")
            else:
                status.update(label="Evaluation failed", state="error")
                st.error("Evaluation failed.")
            with st.expander("Evaluation logs"):
                st.code(output)

with col_b:
    st.info("Target before paid API: average score ≥ 0.88 and average retrieval time under 5 seconds.")

report = load_report()

if not report:
    st.warning("No report found yet. Click Run Evaluation.")
    st.stop()

avg_score = float(report.get("average_score", 0))
avg_time = float(report.get("average_time_seconds", 0))
question_count = int(report.get("question_count", len(report.get("results", []))))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Average Score", f"{avg_score:.3f}")
m2.metric("Average Time", f"{avg_time:.2f} sec")
m3.metric("Questions", question_count)
m4.markdown(
    f'<div class="card"><div class="{status_class(avg_score)}">{status_label(avg_score)}</div><div class="small">Current baseline</div></div>',
    unsafe_allow_html=True,
)

st.divider()

results: List[Dict[str, Any]] = report.get("results", [])
weak = sorted(results, key=lambda x: x.get("score", 0))

st.subheader("Lowest scoring questions first")

for item in weak:
    score = float(item.get("score", 0))
    q = item.get("question", "")
    time_s = item.get("time_seconds", 0)
    hits = item.get("hits", [])
    misses = item.get("misses", [])
    sources = item.get("top_sources", [])

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### {q}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Score", f"{score:.3f}")
        c2.metric("Time", f"{time_s} sec")
        c3.metric("Missing Terms", len(misses))

        st.markdown(f"**Hits:** {', '.join(hits) if hits else 'None'}")
        st.markdown(f"**Misses:** {', '.join(misses) if misses else 'None'}")

        with st.expander("Top sources"):
            for i, src in enumerate(sources, 1):
                title = src.get("title") or "Unknown"
                section = src.get("section") or "Unknown"
                pages = src.get("pages")
                concepts = src.get("concepts") or ""
                preview = src.get("preview") or ""

                st.markdown(f"**{i}. {title}**")
                st.caption(f"Section: {section} | Pages: {pages} | Concepts: {concepts}")
                if preview:
                    st.write(preview[:700])
                st.divider()

        st.markdown("</div>", unsafe_allow_html=True)

st.divider()

st.subheader("Recommendation")
if avg_score >= 0.88:
    st.success("Retrieval is strong. Next improvement should be answer generation, UI polish, and paid API integration later.")
elif avg_score >= 0.80:
    st.info("Retrieval is good. Next improvement should tune weak questions, source balance, and source preview quality.")
else:
    st.warning("Retrieval needs tuning before adding more UI or API features.")
