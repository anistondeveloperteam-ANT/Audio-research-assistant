"""Shared pytest setup: make the project root importable, and keep the suite fully offline."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The offline suite must NEVER load a real torch model. The external-search reranker otherwise loads
# a real CrossEncoder (BAAI/bge-reranker-v2-m3) whenever the dev's .env has ENABLE_LOCAL_RAG=true —
# slow, and it intermittently segfaults transformers/torch on Windows (a native access violation).
# Force the lexical scorer here (rerank_sources already supports that fallback). Set before any
# backend module imports source_ranker so its module-level USE_CROSS_ENCODER reads this value.
os.environ["EXTERNAL_RERANK_CROSS_ENCODER"] = "false"

# Routing must be deterministic and offline in the suite: disable the semantic
# code-intent classifier so routing falls back to the pure-regex is_code_intent
# (no LLM/network). The classifier's own unit tests opt back in and mock the
# provider, so this never hides a real failure.
os.environ["CODE_INTENT_SEMANTIC"] = "false"

# Compact-memory summarization makes an LLM call when older turns go stale; disable it suite-wide
# so chat tests are deterministic + offline. The compact-memory unit tests opt back in and mock
# the provider, so this never hides a real failure.
os.environ["COMPACT_MEMORY"] = "false"

# Query auto-correct makes a live LLM call when a query has an unrecognized token; disable it
# suite-wide so the chat path stays fully offline + deterministic. test_query_refine opts back in
# and mocks the provider.
os.environ["QUERY_REFINE"] = "false"

# The source-relevance gate makes a live LLM call (judging which retrieved sources address the
# question) before drafting. Disable it suite-wide so retrieval/routing tests are deterministic +
# offline; test_source_relevance_gate opts back in and mocks the provider.
os.environ["SOURCE_RELEVANCE_GATE"] = "false"

# The conclusion-matches-work gate makes a live LLM call (confirming the stated result equals what the
# answer's own derivation yields) during verification. Disable it suite-wide for deterministic +
# offline chat tests; test_conclusion_matches_work opts back in and mocks the provider.
os.environ["AGENTIC_CONSISTENCY_CHECK"] = "false"

# The source router makes a live LLM call (deciding reasoning vs web vs corpus) for non-calc,
# non-freshness questions. Disable it suite-wide so chat tests keep the deterministic retrieve-first
# (fail-open 'corpus') behaviour; test_source_router opts back in and mocks/forces the verdict.
os.environ["SOURCE_ROUTER"] = "false"

# Effort scaling makes a SIMPLE question skip angle-planning and run a single verify pass (scaling work
# to complexity). Disable it suite-wide so existing chat/loop tests keep the legacy full-budget
# behaviour (every question planned + the full loop cap); test_effort_scaling opts back in.
os.environ["EFFORT_SCALING"] = "false"

# Experience memory recalls past LESSONS into the draft prompt and captures new ones each answer.
# Disable it suite-wide so existing chat tests aren't perturbed by an injected lessons block;
# test_experience_memory opts back in.
os.environ["EXPERIENCE_MEMORY"] = "false"

# Corpus growth (Phase 2) recalls LEARNED external findings into retrieval and captures cited verified
# findings into a grown corpus. Disable it suite-wide so existing retrieval/answer tests aren't
# perturbed by injected learned passages; test_acquired_knowledge opts back in (and forces synchronous
# background capture so it stays deterministic).
os.environ["CORPUS_GROWTH"] = "false"

# Self-tuning (Phase 3) is OPT-IN: never apply tuned config in the suite. The tuning OVERRIDE layer is
# always present but a no-op with no overrides set; test_self_tuning opts in and manages its own cache.
os.environ["SELF_TUNING"] = "false"

# The citation-verification gate makes deterministic index lookups (network) + a bounded LLM support
# judge before citations are shown. Disable it suite-wide so chat tests stay offline/deterministic and
# aren't perturbed by citation removals; test_citation_verification opts back in and mocks the lookups +
# the provider.
os.environ["CITATION_VERIFICATION"] = "false"


@pytest.fixture(autouse=True)
def _reset_process_context():
    """Reset the two process-global-ish caches around every test: the Phase 3 tuning override cache and
    the per-request settings contextvar — so a test that binds either can never leak into another."""
    from backend.answering import tuning as _tuning
    from backend.common import request_context as _rc
    _tuning.clear_cache()
    _rc.clear_request_settings()
    yield
    _tuning.clear_cache()
    _rc.clear_request_settings()
