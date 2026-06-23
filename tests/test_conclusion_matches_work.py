"""CONCLUSION-MATCHES-WORK. An answer's final stated result must equal the value its OWN reasoning /
derivation produces. A summary line that drifts from the work is internally self-contradictory — the
verification layer now catches that, reconciles the stated result to the work (single source of truth),
and never labels a contradiction 'verified'. General: numeric, categorical, or qualitative.

Proves:
  (a) a stated result that differs from the derived result is caught and CORRECTED so stated == worked;
  (b) every stated instance agrees after reconciliation (single source of truth);
  (c) unit / convention is made consistent between computation and statement;
  (d) a genuinely consistent answer passes UNCHANGED;
  plus: an un-reconcilable contradiction is NOT verified (gates 'verified'), and the unit tests for
  consistency_check / reconcile_answer / is_truly_verified.

Deterministic + offline: the provider is mocked (content-routed). No network, no Docker, no real LLM.
"""
import pytest

import webapp.chat_logic as cl
from backend.answering import agentic_answer as aa
from backend.memory.store import MemoryStore


@pytest.fixture(autouse=True)
def _enable_consistency(monkeypatch):
    # conftest disables the gate suite-wide for deterministic routing; these tests exercise the gate
    # itself, so opt back in (the provider is mocked, so still offline).
    monkeypatch.setenv("AGENTIC_CONSISTENCY_CHECK", "true")


# ======================================================================================
# Unit: consistency_check + reconcile_answer + the combiner.
# ======================================================================================
class _One:
    is_available = True
    model = "fake"

    def __init__(self, reply):
        self.reply = reply

    def stream_chat(self, messages, system="", **k):
        return [self.reply]


def test_consistency_check_flags_only_an_explicit_contradiction():
    assert aa.consistency_check(_One('{"consistent": false, "derived_result": "30"}'),
                                question="q", answer="a")["consistent"] is False
    assert aa.consistency_check(_One('{"consistent": true}'), question="q", answer="a")["consistent"] is True
    # null (no result to check) and unparseable output BOTH fail open to consistent=True.
    assert aa.consistency_check(_One('{"consistent": null}'), question="q", answer="a")["consistent"] is True
    assert aa.consistency_check(_One("not json"), question="q", answer="a")["consistent"] is True


def test_consistency_check_disabled_fails_open(monkeypatch):
    monkeypatch.setenv("AGENTIC_CONSISTENCY_CHECK", "false")
    assert aa.consistency_check(_One('{"consistent": false}'), question="q", answer="a")["consistent"] is True


def test_reconcile_answer_returns_corrected_text():
    out = aa.reconcile_answer(_One("Total = 30 units."), question="q",
                              answer="...the total is 35 units.", check={"derived_result": "30"})
    assert out == "Total = 30 units."


def test_reconcile_answer_disabled_returns_empty(monkeypatch):
    monkeypatch.setenv("AGENTIC_CONSISTENCY_CHECK", "false")
    assert aa.reconcile_answer(_One("x"), question="q", answer="a") == ""


def test_is_truly_verified_requires_internal_consistency(monkeypatch):
    monkeypatch.setenv("AGENTIC_INDEPENDENT_VERIFY", "true")
    assert aa.is_truly_verified(True, {"agrees": True}, consistent=True)        # all pass -> verified
    assert not aa.is_truly_verified(True, {"agrees": True}, consistent=False)   # contradiction -> never verified
    # Even with the independent layer OFF, a self-contradiction is never verified.
    monkeypatch.setenv("AGENTIC_INDEPENDENT_VERIFY", "false")
    assert aa.is_truly_verified(True, None, consistent=True)
    assert not aa.is_truly_verified(True, None, consistent=False)


def test_consistency_prompts_check_stated_vs_derived_and_units():
    s = aa._CONSISTENCY_SYSTEM.lower()
    assert "stated" in s and "derived" in s and "work" in s and "unit" in s
    assert "single source of truth" in aa._CONSISTENCY_FIX_SYSTEM.lower()


# ======================================================================================
# End-to-end via the reasoning path (any-domain; the provider is content-routed).
# ======================================================================================
class _Trace:
    def set(self, **k):
        return self

    def end(self):
        pass


class _P:
    is_available = True
    model = "fake"

    def __init__(self, *, draft, consistent, derived="", reconciled="", agrees="true"):
        self.draft, self.consistent = draft, consistent
        self.derived, self.reconciled, self.agrees = derived, reconciled, agrees

    def stream_chat(self, messages, system="", **k):
        s = (system or "").lower()
        if "internal-consistency checker" in s:                  # the conclusion-matches-work DETECT
            return ['{"consistent": %s, "derived_result": "%s", "stated_result": "x", '
                    '"issues": ["stated != derived"]}' % (self.consistent, self.derived)]
        if "single source of truth" in s:                        # the RECONCILE rewrite
            return [self.reconciled]
        if "independent checker" in s:                           # the independent re-derivation
            return ['{"agrees": %s, "issues": []}' % self.agrees]
        if "answer-quality judge" in s:                          # the dependent reasoning verify
            return ['{"ok": true, "score": 95}']
        return [self.draft]                                      # the reasoning draft

    def unavailable_message(self):
        return "n/a"


def _run(monkeypatch, tmp_path, provider, question="Compute the total."):
    mem = MemoryStore(tmp_path / "m.db")
    sid = mem.create_session(user_id="local")
    qid = mem.start_question(sid, question)
    monkeypatch.setenv("ENABLE_ANSWER_CACHE", "true")
    monkeypatch.setenv("AGENTIC_CONSISTENCY_CHECK", "true")
    monkeypatch.setenv("AGENTIC_INDEPENDENT_VERIFY", "true")
    monkeypatch.setattr(cl, "get_provider", lambda: provider)
    events = list(cl._reasoning_fallback(question, mem, sid, qid["turn_id"], qid["node_id"], "uC",
                                         True, None, None, _Trace()))
    done = [e for e in events if e["type"] == "done"][0]
    hit = mem.find_cached_answer(user_id="uC", question=question)
    return done, hit


def test_simplified_reasoning_ships_a_consistent_answer_unchanged(monkeypatch, tmp_path):
    # The simplified reasoning/calculation path computes ONCE and ships the answer as-is (no reconcile
    # override, no notes). conclusion-matches-work now lives on the EVIDENCE path (tested via the helper
    # below), not here.
    draft = "Two plus two equals four by basic arithmetic, so the final answer to the question is exactly four (4)."
    done, hit = _run(monkeypatch, tmp_path, _P(draft=draft, consistent="true"))
    assert done["answer"] == draft                                       # untouched, no second-guessing
    assert "reconciled" not in done["answer"].lower()
    assert hit is not None and "(4)" in hit["answer"]                    # verified + cached


# --- conclusion-matches-work helper (the EVIDENCE-path mechanism: detect a stated-vs-derived mismatch
#     and reconcile the stated result to the worked value). Tested directly now that the reasoning path
#     no longer runs it. ---
def test_enforce_reconciles_a_stated_vs_derived_mismatch():
    fixed, ok, corrected, _d = cl._enforce_conclusion_matches_work(
        _P(draft="x", consistent="false", derived="30",
           reconciled="6 x 5 = 30, so the total is 30 units."),
        "q", "6 x 5 = 30, so the total is 35 units.")
    assert corrected and ok and "30 units" in fixed and "35" not in fixed


def test_enforce_reconciles_a_unit_inconsistency():
    fixed, ok, corrected, _d = cl._enforce_conclusion_matches_work(
        _P(draft="x", consistent="false", derived="30 MB",
           reconciled="approximately 30 MB of storage."),
        "q", "approximately 30 GB of storage.")
    assert corrected and "30 MB" in fixed and "30 GB" not in fixed


def test_enforce_flags_an_unreconcilable_contradiction():
    fixed, ok, corrected, _d = cl._enforce_conclusion_matches_work(
        _P(draft="x", consistent="false", derived="30", reconciled=""),
        "q", "the total is 35 units.")
    assert ok is False and corrected is False                            # reconcile failed -> withhold verified
