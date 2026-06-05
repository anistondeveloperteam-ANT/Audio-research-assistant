"""
hyde_generator.py  --  Batch 3 (Smart Query Layer)

Local HyDE-style query expansion. No LLM, no API calls.

HyDE (Hypothetical Document Embeddings, Gao et al. 2022) is a retrieval
technique: generate a hypothetical answer with an LLM, embed THAT, then
do vector search. The intuition is that retrieval embeddings work best
on document-vs-document similarity, not question-vs-document. By turning
the question into a passage that LOOKS like an answer, vector search
sees better recall.

We approximate this with template + lexicon expansion. Given a question:
  1. Detect intent (best_method / compare / how_to / metrics / ...)
  2. Detect topic (beamforming / dereverberation / AEC / ...)
  3. Fill an intent template with topic-specific filler vocabulary
  4. Append the original question at the end (preserves exact-term signal)

The result is a doc-like string passed to vector_search instead of the
raw question. The embedding model sees terms like "MVDR, LCMV, and
frequency-invariant beamforming" in addition to whatever the user typed,
which dramatically improves recall on broadly-phrased questions.

Tunable via env var ENABLE_HYDE (default: true).
"""

from __future__ import annotations

import re
from typing import List, Tuple


# Intent templates -- each describes the kind of paragraph a paper
# would have about this question type.
INTENT_TEMPLATES = {
    "best_method": (
        "This work compares {topic_terms} methods. We evaluate {algos} "
        "using {metrics} on standard datasets. We discuss assumptions, "
        "strengths, weaknesses, and practical recommendations."
    ),
    "compare": (
        "We compare {algos} with respect to {metrics}. Each algorithm "
        "has distinct assumptions about {assumptions}. Results show "
        "tradeoffs between performance and complexity."
    ),
    "how_to": (
        "The proposed approach for {topic_terms} consists of the "
        "following steps. First, {step1}. Then, {step2}. Finally, "
        "the output is evaluated using {metrics}."
    ),
    "limitations": (
        "While {algos} achieve strong performance, they have several "
        "limitations: sensitivity to {weakness1}, computational cost "
        "for {weakness2}, and degradation under {weakness3}."
    ),
    "metrics": (
        "We evaluate {algos} using {metrics}. Higher PESQ and STOI "
        "indicate better speech quality. SDR and SI-SDR measure "
        "signal-level fidelity. Latency and complexity matter for "
        "real-time deployment."
    ),
    "implementation": (
        "Implementation of {topic_terms} requires attention to "
        "{assumptions}. Real-time constraints demand low latency and "
        "bounded complexity. We use {algos} and report {metrics}."
    ),
    "summary": (
        "{topic_terms} addresses improving speech quality. We review "
        "{algos} and discuss their key contributions, results, and "
        "limitations."
    ),
    "definition": (
        "{topic_terms} is a signal-processing technique for speech "
        "and audio. It relies on {assumptions} and is typically "
        "implemented using {algos}. Common evaluation uses {metrics}."
    ),
    "general": (
        "{topic_terms} involves {algos}. Typical assumptions include "
        "{assumptions}. Performance is measured using {metrics}."
    ),
}


# Topic-specific filler vocabulary. Drawn from common audio DSP papers.
TOPIC_FILLERS = {
    "beamforming": {
        "algos": "MVDR, LCMV, GSC, and frequency-invariant beamforming",
        "assumptions": "stationary noise, known steering vector, far-field source",
        "metrics": "directivity index, SNR improvement, SI-SDR, PESQ",
        "weakness1": "steering vector mismatch",
        "weakness2": "large arrays",
        "weakness3": "moving sources",
    },
    "noise suppression": {
        "algos": "spectral subtraction, Wiener filtering, RNNoise, DeepFilterNet, DNN-based denoising",
        "assumptions": "additive noise model, short-time stationarity",
        "metrics": "PESQ, STOI, SI-SDR, SNR improvement",
        "weakness1": "non-stationary noise",
        "weakness2": "low-SNR conditions",
        "weakness3": "speech distortion",
    },
    "dereverberation": {
        "algos": "WPE, weighted prediction error, deep filtering, neural dereverberation",
        "assumptions": "linear convolution, time-invariant room",
        "metrics": "PESQ, STOI, SRMR, cepstral distance",
        "weakness1": "moving sources",
        "weakness2": "very long T60",
        "weakness3": "reverberation tail estimation",
    },
    "acoustic echo cancellation": {
        "algos": "NLMS, RLS, Kalman, deep AEC models",
        "assumptions": "linear echo path, known reference signal",
        "metrics": "ERLE, PESQ during double-talk",
        "weakness1": "double-talk",
        "weakness2": "nonlinear distortion",
        "weakness3": "long echo paths",
    },
    "doa": {
        "algos": "MUSIC, ESPRIT, SRP-PHAT, neural DOA estimators",
        "assumptions": "narrowband or wideband signal model, known array geometry",
        "metrics": "RMSE in degrees, angular resolution",
        "weakness1": "low SNR",
        "weakness2": "coherent sources",
        "weakness3": "reverberation",
    },
    "speech enhancement": {
        "algos": "spectral masking, Wiener filtering, DNN, RNN, and Transformer-based models",
        "assumptions": "additive noise, short-time stationarity, speaker generalization",
        "metrics": "PESQ, STOI, SI-SDR, MOS",
        "weakness1": "unseen noise types",
        "weakness2": "speaker mismatch",
        "weakness3": "low-resource deployment",
    },
    "default": {
        "algos": "various signal-processing and deep-learning methods",
        "assumptions": "standard audio-processing assumptions",
        "metrics": "PESQ, STOI, SDR, SNR",
        "weakness1": "out-of-distribution data",
        "weakness2": "high complexity",
        "weakness3": "real-time constraints",
    },
}


_TOPIC_TRIGGERS: List[Tuple[str, List[str]]] = [
    ("beamforming",                ["beamforming", "beamformer", "mvdr", "lcmv", "gsc", "steering vector"]),
    ("doa",                        ["direction of arrival", "doa", "music algorithm", "esprit", "srp-phat", "source localization"]),
    ("dereverberation",            ["dereverberation", "wpe", "reverberation", "room acoustic", "rir"]),
    ("acoustic echo cancellation", ["acoustic echo", "aec", "echo cancellation", "double-talk"]),
    ("noise suppression",          ["noise suppression", "noise reduction", "denoising", "denoise", "deepfilternet", "rnnoise"]),
    ("speech enhancement",         ["speech enhancement", "speech quality", "intelligibility", "voice quality"]),
]


def detect_intent(question: str) -> str:
    """Classify the question into a coarse intent route so HyDE can
    generate a hypothetical answer in the right style."""
    q = " " + (question or "").lower() + " "
    if any(x in q for x in [" best ", " suitable ", " recommend ", "which method", "which algorithm"]):
        return "best_method"
    if any(x in q for x in [" compare ", " versus ", " vs ", " difference ", " better than "]):
        return "compare"
    if any(x in q for x in [" limit", " drawback", " weakness", " fail", " problem"]):
        return "limitations"
    if any(x in q for x in [" metric", " evaluate", " benchmark", " measure"]):
        return "metrics"
    if any(x in q for x in [" implement", " deploy", " run "]):
        return "implementation"
    if any(x in q for x in [" how to ", " how can ", " improve ", " remove ", " reduce "]):
        return "how_to"
    if any(x in q for x in [" summarize", " summary", " overview"]):
        return "summary"
    if any(x in q for x in [" what is ", " define ", " explain "]):
        return "definition"
    return "general"


def detect_topic(question: str) -> Tuple[str, str]:
    """Return (filler_key, human-readable topic terms)."""
    q = (question or "").lower()
    for key, triggers in _TOPIC_TRIGGERS:
        if any(t in q for t in triggers):
            return key, key
    return "default", "audio signal processing"


def hyde_expand(question: str) -> str:
    """
    Build a doc-style expansion of the question.
    Returns expansion + original question concatenated.

    Robust to empty / weird inputs -- always returns a string.
    """
    question = (question or "").strip()
    if not question:
        return ""

    try:
        intent = detect_intent(question)
        topic_key, topic_terms = detect_topic(question)

        fillers = dict(TOPIC_FILLERS.get(topic_key, TOPIC_FILLERS["default"]))
        fillers["topic_terms"] = topic_terms
        fillers["step1"] = "the input audio signal is preprocessed and analyzed"
        fillers["step2"] = "the algorithm produces the enhanced output"

        template = INTENT_TEMPLATES.get(intent, INTENT_TEMPLATES["general"])
        try:
            expansion = template.format(**fillers)
        except (KeyError, IndexError):
            expansion = INTENT_TEMPLATES["general"].format(**fillers)

        acronyms = re.findall(r"\b[A-Z][A-Z0-9\-]{1,}\b", question)
        if acronyms:
            expansion += " Specifically: " + ", ".join(sorted(set(acronyms))) + "."

        # Preserve original question for exact-term / acronym match signal
        return expansion + " " + question
    except Exception:
        # If anything in the template path fails, fall back gracefully
        return question


def hyde_queries(question: str) -> List[str]:
    """
    Convenience: return both the original question AND the HyDE
    expansion as a list. Useful for callers that want to run two
    vector searches and fuse them.
    """
    out = [question]
    expanded = hyde_expand(question)
    if expanded and expanded != question:
        out.append(expanded)
    return out
