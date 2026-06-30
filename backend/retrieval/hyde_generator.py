"""
hyde_generator.py  --  local HyDE-style query expansion (no LLM, no API calls).

HyDE (Hypothetical Document Embeddings, Gao et al. 2022) is a retrieval technique:
generate a hypothetical answer, embed THAT, then do vector search. Retrieval
embeddings work best on document-vs-document similarity, so turning the question
into a passage that LOOKS like an answer improves recall.

We approximate it with a domain-neutral template, no LLM. Given a question:
  1. Detect intent (best_method / compare / how_to / metrics / ...).
  2. Fill a generic "answer-shaped" template (no field-specific vocabulary).
  3. Append any uppercase acronyms found, then the original question (preserves
     the exact-term signal the user actually typed).

The result is a doc-like string passed to vector_search instead of the raw
question. It is deliberately domain-agnostic -- the specific signal comes from
the user's own words and acronyms, not from a hardcoded topic lexicon.

Tunable via env var ENABLE_HYDE (default: true).
"""

from __future__ import annotations

import re
from typing import List, Tuple


# Intent templates -- each describes the shape of a paragraph that would answer
# this kind of question. Domain-neutral: no field-specific terms.
INTENT_TEMPLATES = {
    "best_method": (
        "Methods for {topic_terms} are compared here. We evaluate {algos} using "
        "{metrics} on standard datasets and discuss assumptions, strengths, "
        "weaknesses, and practical recommendations."
    ),
    "compare": (
        "We compare {algos} with respect to {metrics}. Each makes distinct "
        "assumptions about {assumptions}, with tradeoffs between performance and cost."
    ),
    "how_to": (
        "An approach to {topic_terms} typically proceeds in steps. First, {step1}. "
        "Then, {step2}. Finally, the output is evaluated using {metrics}."
    ),
    "limitations": (
        "While {algos} perform well, they have limitations: sensitivity to "
        "{weakness1}, the cost of {weakness2}, and degradation under {weakness3}."
    ),
    "metrics": (
        "We evaluate {algos} using {metrics}, reporting accuracy and efficiency "
        "and discussing which measures matter in practice."
    ),
    "implementation": (
        "Implementing solutions for {topic_terms} requires attention to "
        "{assumptions}. Practical constraints demand efficiency and bounded "
        "complexity. We use {algos} and report {metrics}."
    ),
    "summary": (
        "This reviews {topic_terms}. We survey {algos} and discuss key "
        "contributions, results, and limitations."
    ),
    "definition": (
        "{topic_terms} relies on {assumptions} and is typically realized using "
        "{algos}. It is commonly evaluated with {metrics}."
    ),
    "general": (
        "Work on {topic_terms} involves {algos}. Typical assumptions include "
        "{assumptions}. Performance is measured using {metrics}."
    ),
}


# Generic, field-neutral filler vocabulary. The specific signal comes from the
# user's question text + acronyms (appended in hyde_expand), not from here.
GENERIC_FILLERS = {
    "algos": "established and recent methods",
    "assumptions": "standard modeling assumptions",
    "metrics": "standard quantitative metrics",
    "weakness1": "out-of-distribution data",
    "weakness2": "high computational complexity",
    "weakness3": "limited or noisy data",
}


def detect_intent(question: str) -> str:
    """Classify the question into a coarse intent route so HyDE can generate a
    hypothetical answer in the right style."""
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
    """Domain-neutral placeholder. HyDE no longer injects field-specific vocabulary;
    the user's own terms and acronyms carry the specific signal. Returns a generic
    (key, topic_terms) pair."""
    return "default", "this subject"


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
        _topic_key, topic_terms = detect_topic(question)

        fillers = dict(GENERIC_FILLERS)
        fillers["topic_terms"] = topic_terms
        fillers["step1"] = "the inputs are prepared and analyzed"
        fillers["step2"] = "the method produces its output"

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
    Convenience: return both the original question AND the HyDE expansion as a
    list. Useful for callers that want to run two vector searches and fuse them.
    """
    out = [question]
    expanded = hyde_expand(question)
    if expanded and expanded != question:
        out.append(expanded)
    return out
