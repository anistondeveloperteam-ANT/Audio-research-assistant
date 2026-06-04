"""
query_sanity.py  --  Detect nonsense / out-of-scope queries

The goal: prevent the LLM from hallucinating confident answers to
gibberish input like "buoh" or "asdf qwer zxcv".

Approach (no ML, no API calls -- pure heuristics):
  1. Length check: too short = probably not a real question
  2. Vowel ratio: no vowels or all vowels = keyboard mash
  3. Character repetition: "aaaaa" or "abababab" = not real words
  4. Word legitimacy: low ratio of recognizable English words

Returns a SanityResult that tells the caller whether to proceed or
refuse, plus a user-friendly message if refusing.

This is NOT a full safety check. It's a cheap first pass that catches
obvious garbage. Grammatically-correct-but-out-of-scope questions
(e.g. "What's the weather?") will pass this check and rely on the
LLM's stricter system prompt + the low-score warning instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ----------------------------------------------------------------------
# Tunable thresholds
# ----------------------------------------------------------------------

MIN_QUERY_CHARS = 4
MIN_WORD_LEGITIMACY = 0.3   # at least 30% of words should look like real words
MIN_VOWEL_RATIO = 0.10      # English has ~38% vowels; below 10% is suspect
MAX_VOWEL_RATIO = 0.85      # all-vowels is also suspect


# ----------------------------------------------------------------------
# Common English words (small built-in wordlist for the legitimacy check)
# Plus domain-specific terms (audio DSP, beamforming, etc.) so legitimate
# technical questions don't get falsely flagged.
# ----------------------------------------------------------------------

_COMMON_ENGLISH = {
    # Function words
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can",
    "do", "for", "from", "had", "has", "have", "how", "i", "if",
    "in", "is", "it", "me", "my", "no", "not", "of", "on", "or",
    "should", "so", "than", "that", "the", "this", "to", "was",
    "we", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "would", "you", "your", "yes",
    # Common verbs / nouns relevant to research Q&A
    "about", "after", "all", "also", "any", "approach", "best",
    "between", "build", "calculate", "case", "compare", "computation",
    "compute", "concept", "consider", "create", "data", "define",
    "demonstrate", "describe", "design", "detect", "difference",
    "different", "discuss", "does", "doing", "during", "each",
    "effect", "estimate", "evaluate", "example", "experiment",
    "explain", "explore", "feature", "find", "first", "function",
    "generate", "give", "graph", "help", "high", "implement",
    "improve", "into", "learn", "let", "list", "look", "low",
    "make", "many", "matter", "may", "mean", "measure", "method",
    "model", "more", "most", "much", "name", "need", "new", "now",
    "number", "off", "often", "one", "option", "other", "out",
    "output", "over", "paper", "performance", "place", "please",
    "plot", "point", "predict", "problem", "process", "provide",
    "purpose", "question", "rate", "read", "reason", "recent",
    "recommend", "research", "result", "review", "right", "run",
    "same", "say", "see", "set", "show", "simulate", "simulation",
    "since", "small", "some", "source", "specific", "start", "step",
    "study", "such", "summarize", "suppose", "system", "take",
    "tell", "term", "test", "then", "they", "think", "those",
    "through", "time", "tool", "try", "two", "type", "under",
    "understand", "use", "used", "user", "using", "value", "want",
    "way", "weight", "well", "while", "work", "write", "year",
    # Question words / phrasings
    "could", "may", "might", "must", "shall", "ought",
    # Negatives
    "never", "none", "nothing", "nobody", "nowhere",
}

# Audio / DSP / ML domain vocabulary so legitimate technical questions
# don't get flagged as gibberish.
_DOMAIN_VOCAB = {
    "audio", "signal", "processing", "frequency", "spectrum", "fft",
    "stft", "filter", "fir", "iir", "convolution", "correlation",
    "noise", "snr", "db", "decibel", "sampling", "khz", "mhz",
    "speech", "voice", "speaker", "recognition", "enhancement",
    "denoising", "reverberation", "echo", "doa", "direction",
    "arrival", "beamforming", "mvdr", "lcmv", "music", "esprit",
    "phat", "gcc", "array", "microphone", "mic", "spatial",
    "covariance", "matrix", "eigenvalue", "subspace", "steering",
    "vector", "delay", "sum", "sensor", "ula", "uca", "linear",
    "circular", "planar", "geometry", "room", "rir", "impulse",
    "response", "pesq", "stoi", "stft", "mel", "mfcc", "cepstrum",
    "pitch", "formant", "phoneme", "asr", "tts", "synthesis",
    "neural", "network", "transformer", "lstm", "rnn", "cnn",
    "deep", "learning", "machine", "training", "inference",
    "model", "dataset", "benchmark", "evaluation", "metric",
    "loss", "gradient", "optimizer", "epoch", "batch", "tensor",
    "embedding", "attention", "encoder", "decoder", "diffusion",
    "wavenet", "whisper", "openai", "anthropic", "claude", "gpt",
    "ollama", "qwen", "llama", "rag", "retrieval", "embedding",
    "vector", "database", "oracle", "sqlite", "pdf", "paper",
    "publication", "arxiv", "semantic", "scholar", "citation",
    "ieee", "asa", "acoustics", "psychoacoustic", "hrtf", "head",
    "related", "transfer", "binaural", "ambisonic", "surround",
    "stereo", "mono", "channel", "decibel", "loudness", "perception",
}

_LEGIT_WORDS = _COMMON_ENGLISH | _DOMAIN_VOCAB


# ----------------------------------------------------------------------
# Result type
# ----------------------------------------------------------------------

@dataclass
class SanityResult:
    """What the sanity check decided."""
    ok: bool                # True = proceed, False = refuse
    reason: str = ""        # internal reason (for logs)
    user_message: str = ""  # what to show the user if refusing


# ----------------------------------------------------------------------
# Public check
# ----------------------------------------------------------------------

def check_query_sanity(query: Optional[str]) -> SanityResult:
    """Run cheap heuristic checks on the user's query.

    Returns SanityResult(ok=True, ...) if the query looks legitimate
    enough to send to retrieval + the LLM. Otherwise returns a result
    with ok=False and a polite user-facing message.
    """
    if not query:
        return SanityResult(
            ok=False,
            reason="empty",
            user_message="Please type a question.",
        )

    q = query.strip()
    if not q:
        return SanityResult(
            ok=False,
            reason="whitespace_only",
            user_message="Please type a question.",
        )

    # 1) Too short
    if len(q) < MIN_QUERY_CHARS:
        return SanityResult(
            ok=False,
            reason="too_short",
            user_message=(
                "Your question is too short for me to understand. "
                "Could you ask it as a full sentence? "
                "For example: \"What is MVDR beamforming?\""
            ),
        )

    # 2) Only punctuation / digits / non-letters
    letters_only = re.sub(r"[^A-Za-z]", "", q)
    if len(letters_only) < 3:
        return SanityResult(
            ok=False,
            reason="no_letters",
            user_message=(
                "I can't find a question in that. "
                "Could you rephrase using words?"
            ),
        )

    # 3) Vowel ratio sanity
    lower = letters_only.lower()
    n_vowels = sum(1 for c in lower if c in "aeiou")
    vowel_ratio = n_vowels / len(lower)
    if vowel_ratio < MIN_VOWEL_RATIO or vowel_ratio > MAX_VOWEL_RATIO:
        return SanityResult(
            ok=False,
            reason=f"vowel_ratio={vowel_ratio:.2f}",
            user_message=(
                "That doesn't look like a real sentence. "
                "Could you rephrase your question?"
            ),
        )

    # 4) Repeated character / pattern detection
    #    e.g., "aaaaaaaa", "lolololol", "abcabcabc"
    if _is_repeated_pattern(lower):
        return SanityResult(
            ok=False,
            reason="repeated_pattern",
            user_message=(
                "That doesn't look like a real sentence. "
                "Could you rephrase your question?"
            ),
        )

    # 5) Word legitimacy
    #    Split on whitespace/punctuation; check what fraction are
    #    recognizable English or domain words.
    tokens = [t.lower() for t in re.split(r"[^A-Za-z']+", q) if t]
    if not tokens:
        return SanityResult(
            ok=False,
            reason="no_tokens",
            user_message="I couldn't find any words in your question.",
        )

    legitimate = sum(1 for t in tokens if _is_legit_word(t))
    legitimacy = legitimate / len(tokens)

    if legitimacy < MIN_WORD_LEGITIMACY:
        return SanityResult(
            ok=False,
            reason=f"low_legitimacy={legitimacy:.2f}",
            user_message=(
                "I don't recognize most of the words in your question -- "
                "they don't look like English or audio-DSP terms. "
                "Could you rephrase?"
            ),
        )

    # All checks passed
    return SanityResult(ok=True, reason="passed")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _is_repeated_pattern(s: str) -> bool:
    """Detect strings made of one repeated substring of length 1-3.
    'aaaaaa' (period 1), 'ababab' (period 2), 'abcabc' (period 3)."""
    if len(s) < 6:
        return False
    for period in (1, 2, 3):
        unit = s[:period]
        if unit * (len(s) // period) == s[: (len(s) // period) * period]:
            return True
    return False


def _is_legit_word(token: str) -> bool:
    """A token counts as legitimate if it's in the wordlist OR is a
    short common pattern (numbers, single letters used as variables,
    common abbreviations not in the wordlist)."""
    token = token.lower().strip("'")
    if not token:
        return False
    # Numeric tokens always count
    if any(c.isdigit() for c in token):
        return True
    # Single-letter tokens (used as math variables, "x", "y", "n")
    if len(token) == 1:
        return True
    # In the wordlist?
    if token in _LEGIT_WORDS:
        return True
    # Common plurals / past tense / -ing of legit words
    for suffix in ("s", "es", "ed", "ing", "er", "est", "ly", "tion", "sion"):
        if token.endswith(suffix) and token[: -len(suffix)] in _LEGIT_WORDS:
            return True
    return False
