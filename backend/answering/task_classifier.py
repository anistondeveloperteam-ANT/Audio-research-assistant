"""
task_classifier.py  --  Semantic intent + task-type classification.

Replaces brittle keyword matching for the single routing question "should this go
to the autonomous code agent?" and, in the same call, decides HOW the agent must
verify the result. Domain-independent: it recognizes ANY request to write/run/
simulate/benchmark/model/compute, regardless of wording (audio, finance, physics,
ML, data, ...), where the regex `is_code_intent` only catches a few phrasings.

It returns a `TaskClass`:
    code_task : bool   -- route to the code agent (True) or the prose pipeline
    task_type : str    -- how to verify a code task:
        "deterministic"     exact expected-output tests (sorting, math, parsing)
        "simulation"        invariants/properties (pendulum, epidemic, Monte Carlo)
        "numeric_algorithm" domain invariants (FFT/Parseval, beamformer wᴴd=1, BS parity)
        "none"              not a code task
    confidence : float
    source    : str    -- "llm" | "regex" | "cache" (provenance, for logs)

Design (mirrors backend.answering.query_refine):
  * One fast-model LLM call returning strict JSON; cached so repeats are free.
  * High recall: a regex `is_code_intent` hit forces code_task=True (union), so
    obvious code requests are never missed even if the model hesitates.
  * Never breaks: a tight timeout + catch-all fall back to the regex verdict on
    any provider error/timeout/unavailability. No new dependency.

Toggle / tune via .env (read live, never frozen into constants):
    CODE_INTENT_SEMANTIC=true|false   (default true; false = regex only, instant)
    CODE_INTENT_TIMEOUT=3.0           (seconds; fall back to regex after this)
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

from backend.answering.code_intent import is_code_intent, is_reasoning_question

# ----------------------------------------------------------------------
# Types + constants
# ----------------------------------------------------------------------

TASK_TYPES = ("deterministic", "simulation", "numeric_algorithm", "none")


@dataclass(frozen=True)
class TaskClass:
    code_task: bool
    task_type: str
    confidence: float = 0.0
    source: str = "regex"


_MAX_TOKENS = 80
_HARD_CHAR_CAP = 600
_DEFAULT_TIMEOUT = 3.0
_CACHE_MAX = 512

_SYSTEM_PROMPT = (
    "You are the ROUTER for a research assistant. Analyze the user's message and "
    "decide ONE thing accurately: does the user want a PROGRAM produced or run, or "
    "do they want an ANSWER / EXPLANATION?\n"
    "\n"
    "Decide by WHAT THE USER WANTS, never by whether math or numbers are involved.\n"
    "\n"
    "code_task = FALSE (answer by REASONING, no code) when the user wants a result "
    "or explanation they can be handed directly, EVEN IF it needs arithmetic, a "
    "formula, a derivation, or a specific numeric answer. Signals: 'compute', "
    "'calculate', 'derive', 'how much', 'how many', 'what is the value of', 'show "
    "your reasoning', 'explain', 'prove', 'estimate'. A quantitative word problem a "
    "person can work out by hand (arithmetic, unit conversion, algebra, a "
    "closed-form formula) is a REASONING question. The mere presence of numbers, "
    "formulas, or a required numeric answer is NOT a code task.\n"
    "\n"
    "code_task = TRUE (route to the code agent) ONLY when the user wants software "
    "produced or executed: they ask to write / give / build / generate code, a "
    "script, a program, a function, or a snippet; to implement, refactor, debug, or "
    "benchmark something; to run, simulate, or model a system; OR the computation "
    "genuinely NEEDS execution because it is too large / iterative / data-driven for "
    "hand reasoning (process a dataset, run a Monte-Carlo, iterate thousands of "
    "steps, sort a big list). This holds in ANY domain (audio, finance, physics, "
    "ML, data, ...).\n"
    "\n"
    "Examples:\n"
    "  'How much storage for 3 minutes of 44.1 kHz 16-bit stereo audio? Show your "
    "reasoning' -> code_task=false  (a calculation; answer with worked steps + units)\n"
    "  'What is 17 x 23?'  /  'Derive the Black-Scholes pricing formula' -> "
    "code_task=false\n"
    "  'What is MVDR beamforming?'  (a document/explanation question) -> "
    "code_task=false\n"
    "  'Write Python to FFT a chirp'  /  'simulate a damped pendulum'  /  'implement "
    "quicksort'  /  'benchmark mergesort vs quicksort' -> code_task=true\n"
    "\n"
    "If (and ONLY if) it IS a code task, also classify how it should be verified:\n"
    '  "deterministic"     -- a single correct output exists (sorting, parsing, '
    "math, classic algorithms like Dijkstra/quicksort).\n"
    '  "simulation"        -- stochastic / time-stepped / Monte-Carlo / physical '
    "or epidemic / synthetic-signal generation where output varies but properties "
    "must hold (pendulum, SIR spread, random processes).\n"
    '  "numeric_algorithm" -- numerical / DSP / linear-algebra / optimization with '
    "mathematical invariants (FFT/Parseval, beamformer/MVDR, Black-Scholes pricing, "
    "solvers, optimizers).\n"
    "Reply with ONLY one line of strict JSON, no prose:\n"
    '{"code_task": true|false, "task_type": '
    '"deterministic"|"simulation"|"numeric_algorithm"|"none", "confidence": 0.0-1.0}'
)

# Regex hints used ONLY when the LLM is unavailable, to still pick a sane task_type.
_SIM_RE = re.compile(
    r"\b(simulat\w*|monte[\s-]?carlo|epidemic|pandemic|sir|seir|pendulum|"
    r"random\s+walk|stochastic|agent[\s-]?based|particle|n[\s-]?body|diffusion)\b")
_NUMALG_RE = re.compile(
    r"\b(fft|dft|stft|parseval|beamform\w*|mvdr|lcmv|black[\s-]?scholes|option|"
    r"optimi[sz]\w*|solver|eigen\w*|convolv\w*|convolution|filter|gradient|"
    r"linear\s+algebra|matrix|integral|interpolat\w*|pricing|monte)\b")


# ----------------------------------------------------------------------
# In-process LRU cache
# ----------------------------------------------------------------------

_cache: "OrderedDict[str, TaskClass]" = OrderedDict()
_cache_lock = threading.Lock()


def clear_cache() -> None:
    """Empty the classification cache (used by tests)."""
    with _cache_lock:
        _cache.clear()


def _cache_get(key: str) -> Optional[TaskClass]:
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    return None


def _cache_put(key: str, value: TaskClass) -> None:
    with _cache_lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)


# ----------------------------------------------------------------------
# Config (read live so .env / test monkeypatching takes effect)
# ----------------------------------------------------------------------

def _semantic_enabled() -> bool:
    return os.getenv("CODE_INTENT_SEMANTIC", "true").strip().lower() not in (
        "0", "false", "no", "off")


def _timeout() -> float:
    try:
        return max(0.3, float(os.getenv("CODE_INTENT_TIMEOUT", str(_DEFAULT_TIMEOUT))))
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT


# ----------------------------------------------------------------------
# Regex-only verdict (fallback + task-type heuristic)
# ----------------------------------------------------------------------

def _regex_task_type(query: str) -> str:
    """Best-effort task_type from keywords (used when the LLM is unavailable)."""
    s = " " + (query or "").lower() + " "
    if _SIM_RE.search(s):
        return "simulation"
    if _NUMALG_RE.search(s):
        return "numeric_algorithm"
    return "deterministic"


def _regex_verdict(query: str) -> TaskClass:
    if is_code_intent(query):
        return TaskClass(True, _regex_task_type(query), 0.5, "regex")
    return TaskClass(False, "none", 0.5, "regex")


def _normalize_type(code_task: bool, task_type: object) -> str:
    if not code_task:
        return "none"
    t = str(task_type or "").strip().lower()
    return t if t in TASK_TYPES and t != "none" else "deterministic"


# ----------------------------------------------------------------------
# LLM classification
# ----------------------------------------------------------------------

def _parse_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        return None


def _llm_classify(query: str) -> Optional[TaskClass]:
    """One short, timeout-bounded LLM call. Returns a TaskClass or None on
    unavailability / timeout / error / unparseable output (caller falls back)."""
    from backend.llm.streaming_provider import get_provider

    provider = get_provider()
    if not provider.is_available:
        return None

    def _run() -> str:
        parts: List[str] = []
        total = 0
        for tok in provider.stream_chat(
            [{"role": "user", "content": query}],
            system=_SYSTEM_PROMPT,
            max_tokens=_MAX_TOKENS,
            temperature=0.0,
        ):
            if not isinstance(tok, str):
                continue
            parts.append(tok)
            total += len(tok)
            if total > _HARD_CHAR_CAP:
                break
        return "".join(parts)

    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run)
        try:
            raw = fut.result(timeout=_timeout())
        except Exception:                               # noqa: BLE001 - timeout/provider error
            return None

    obj = _parse_json(raw)
    if obj is None:
        return None
    code_task = bool(obj.get("code_task"))
    try:
        conf = float(obj.get("confidence", 0.7))
    except (TypeError, ValueError):
        conf = 0.7
    return TaskClass(code_task, _normalize_type(code_task, obj.get("task_type")), conf, "llm")


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------

def classify(query: Optional[str]) -> TaskClass:
    """Classify a user message into routing + verification intent. Never raises;
    falls back to the regex verdict on disabled/empty input or any LLM failure."""
    q = (query or "").strip()
    regex = _regex_verdict(q)
    # DETERMINISTIC reasoning veto (overrides the semantic model): a clear calculation / reasoning
    # question ("how much/many", "show your reasoning/steps", "explain", ...) with NO code-production
    # signal is ANSWERED BY REASONING, never the code agent — even if the LLM misclassifies it as code.
    # is_reasoning_question already excludes anything is_code_intent flags, so genuine code is untouched.
    if q and is_reasoning_question(q):
        return TaskClass(False, "none", 0.9, "reasoning")
    try:
        if not q or not _semantic_enabled():
            return regex
        key = " ".join(q.lower().split())
        cached = _cache_get(key)
        if cached is not None:
            return cached
        llm = _llm_classify(q)
        if llm is None:
            return regex                                # don't cache transient failures
        # High recall: an obvious-code regex hit forces code_task=True even if the
        # model hesitated; the model's task_type still steers verification.
        code_task = llm.code_task or regex.code_task
        task_type = llm.task_type if llm.code_task else (
            regex.task_type if regex.code_task else "none")
        result = TaskClass(code_task, _normalize_type(code_task, task_type),
                           llm.confidence, "llm")
        _cache_put(key, result)
        return result
    except Exception:                                   # noqa: BLE001 - never break routing
        return regex


def is_code_task(query: Optional[str]) -> bool:
    """Convenience boolean for routing (semantic, with regex fallback)."""
    return classify(query).code_task


def infer_task_type(task: Optional[str]) -> str:
    """Resolve a concrete verification task_type for a known code task. Used by the
    agent when routing didn't already supply one. Falls back to the regex heuristic."""
    tc = classify(task)
    if tc.task_type and tc.task_type != "none":
        return tc.task_type
    return _regex_task_type(task or "")
