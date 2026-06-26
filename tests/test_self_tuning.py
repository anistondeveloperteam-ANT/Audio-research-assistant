"""SELF-TUNING — the agent calibrates its own thresholds, but ONLY via an eval gate (Phase 3).

Every threshold getter routes through a zero-latency override layer; the eval-gated tuner adopts a new
value ONLY when it provably improves an offline metric, bounded + reversible + recorded, and APPLIES
only when SELF_TUNING is on (propose-only otherwise — honoring the owner's "no silent self-tuning" rule).

Proves: (a) the override layer changes a real getter's output and clamps to bounds; (b) refresh() loads
from the DB and is TTL-gated/zero-latency; (c) store round-trip for tuned_config + tuning_events;
(d) the tuner ADOPTS an eval-improving change and IGNORES a non-improving one; (e) it respects bounds and
records every trial; (f) propose-only (SELF_TUNING off) changes nothing; apply (on) persists + goes live;
(g) reversibility — clear_overrides reverts; (h) a pinned candidate is not clobbered by refresh().
"""
import pytest

from backend.answering import tuning
from backend.answering import experience as ex
from backend.answering.evidence_grader import crag_strong_min
from backend.evaluation import self_tuner as st
from backend.memory.store import MemoryStore


@pytest.fixture(autouse=True)
def _clean():
    tuning.clear_cache()
    yield
    tuning.clear_cache()


def _mem(tmp_path):
    return MemoryStore(tmp_path / "m.db")


# ---- (a) the override layer changes a real getter and clamps to bounds ----
def test_override_layer_changes_getter_and_clamps():
    assert abs(ex._min_relevance() - 0.62) < 1e-9          # stock default
    tuning.set_cache({"EXPERIENCE_MIN_RELEVANCE": 0.80})
    assert abs(ex._min_relevance() - 0.80) < 1e-9          # override wins
    tuning.set_cache({"EXPERIENCE_MIN_RELEVANCE": 99.0})   # absurd -> clamped to hi (0.90)
    assert abs(ex._min_relevance() - 0.90) < 1e-9
    tuning.set_cache({"CRAG_STRONG_MIN": 0.40})
    assert abs(crag_strong_min() - 0.40) < 1e-9
    tuning.set_cache({"EXPERIENCE_TOP_K": 5})              # integer tunable stays int
    assert ex._top_k() == 5 and isinstance(ex._top_k(), int)


def test_tuned_passthrough_for_unknown_name():
    assert tuning.tuned("NOT_A_TUNABLE", 0.123) == 0.123   # unknown -> fallback untouched


def test_non_finite_override_falls_back_to_default():
    # a NaN/inf override (bad evaluator or hand-edited DB row) must NOT propagate to a getter
    # (int(round(nan)) would raise); clamp returns the tunable's default instead.
    for bad in (float("nan"), float("inf"), float("-inf")):
        tuning.set_cache({"EXPERIENCE_MIN_RELEVANCE": bad, "EXPERIENCE_TOP_K": bad})
        assert abs(ex._min_relevance() - 0.62) < 1e-9       # default, not NaN
        assert ex._top_k() == 3 and isinstance(ex._top_k(), int)  # default, no ValueError


# ---- (b) refresh loads from DB and is TTL-gated ----
def test_refresh_loads_from_db_and_is_ttl_gated(tmp_path):
    m = _mem(tmp_path)
    m.set_tuned_config("CORPUS_MIN_RELEVANCE", 0.7)
    tuning.refresh(m, force=True)
    assert tuning.current_overrides()["CORPUS_MIN_RELEVANCE"] == 0.7
    # a later DB change is NOT picked up within the TTL (no DB hit on the hot path)...
    m.set_tuned_config("CORPUS_MIN_RELEVANCE", 0.8)
    tuning.refresh(m, ttl=10_000)
    assert tuning.current_overrides()["CORPUS_MIN_RELEVANCE"] == 0.7
    # ...until forced.
    tuning.refresh(m, force=True)
    assert tuning.current_overrides()["CORPUS_MIN_RELEVANCE"] == 0.8


# ---- (c) store round-trip for tuned_config + tuning_events ----
def test_store_tuned_config_and_events_roundtrip(tmp_path):
    m = _mem(tmp_path)
    assert m.get_tuned_config() == {}
    m.set_tuned_config("CRAG_PARTIAL_MIN", 0.25, source="t")
    m.set_tuned_config("CRAG_PARTIAL_MIN", 0.22, source="t")   # upsert
    assert m.get_tuned_config() == {"CRAG_PARTIAL_MIN": 0.22}
    m.record_tuning_event(name="CRAG_PARTIAL_MIN", old_value=0.30, new_value=0.22,
                          metric_before=0.5, metric_after=0.6, accepted=True, applied=True, note="x")
    hist = m.get_tuning_history(limit=10)
    assert len(hist) == 1 and hist[0]["accepted"] == 1 and hist[0]["new_value"] == 0.22
    assert m.clear_tuned_config("CRAG_PARTIAL_MIN") == 1
    assert m.get_tuned_config() == {}


# ---- (d) the tuner ADOPTS an improving change, IGNORES a non-improving one ----
def _peak_evaluator(name, target):
    """Higher score the closer `name`'s value is to `target` (a single-peak metric)."""
    def _ev(overrides):
        t = tuning.TUNABLES[name]
        v = overrides.get(name, t.default)
        return -abs(v - target)
    return _ev


def test_tuner_adopts_improving_change(tmp_path):
    m = _mem(tmp_path)
    # peak well ABOVE the 0.62 default -> tuner should climb upward and persist
    res = st.tune(mem=m, evaluate_fn=_peak_evaluator("EXPERIENCE_MIN_RELEVANCE", 0.90),
                  names=["EXPERIENCE_MIN_RELEVANCE"], rounds=5, apply=True)
    assert res["applied"] is True
    assert res["changes"] and res["changes"][0]["name"] == "EXPERIENCE_MIN_RELEVANCE"
    assert res["final_metric"] >= res["start_metric"]
    assert m.get_tuned_config()["EXPERIENCE_MIN_RELEVANCE"] > 0.62
    assert ex._min_relevance() > 0.62                      # the change is LIVE


def test_tuner_ignores_non_improving_change(tmp_path):
    m = _mem(tmp_path)
    # metric is flat -> no candidate beats the baseline -> nothing adopted
    res = st.tune(mem=m, evaluate_fn=lambda o: 1.0,
                  names=["CRAG_STRONG_MIN"], rounds=3, apply=True)
    assert res["changes"] == []
    assert m.get_tuned_config() == {}
    # but trials were still RECORDED for audit
    assert len(m.get_tuning_history(limit=50)) >= 2


# ---- (e) the tuner respects bounds ----
def test_tuner_never_exceeds_bounds(tmp_path):
    m = _mem(tmp_path)
    t = tuning.TUNABLES["CRAG_PARTIAL_MIN"]
    # target far above the upper bound -> the adopted value is clamped at hi, never beyond
    st.tune(mem=m, evaluate_fn=_peak_evaluator("CRAG_PARTIAL_MIN", 99.0),
            names=["CRAG_PARTIAL_MIN"], rounds=20, apply=True)
    val = m.get_tuned_config().get("CRAG_PARTIAL_MIN", t.default)
    assert t.lo <= val <= t.hi


# ---- (f) propose-only changes nothing; apply persists ----
def test_propose_only_does_not_change_behaviour(tmp_path, monkeypatch):
    m = _mem(tmp_path)
    monkeypatch.setenv("SELF_TUNING", "false")
    res = st.tune(mem=m, evaluate_fn=_peak_evaluator("CORPUS_MIN_RELEVANCE", 0.90),
                  names=["CORPUS_MIN_RELEVANCE"], rounds=5)     # apply defaults to SELF_TUNING (off)
    assert res["applied"] is False
    assert res["changes"]                                  # it WOULD change...
    assert m.get_tuned_config() == {}                      # ...but persisted nothing
    assert len(m.get_tuning_history(limit=50)) >= 1        # trials recorded as proposals


def test_apply_gated_on_self_tuning_env(tmp_path, monkeypatch):
    m = _mem(tmp_path)
    monkeypatch.setenv("SELF_TUNING", "on")
    assert st.self_tuning_enabled() is True
    res = st.tune(mem=m, evaluate_fn=_peak_evaluator("CORPUS_MIN_RELEVANCE", 0.90),
                  names=["CORPUS_MIN_RELEVANCE"], rounds=5)     # apply=None -> uses SELF_TUNING=on
    assert res["applied"] is True and m.get_tuned_config().get("CORPUS_MIN_RELEVANCE", 0.5) > 0.5


# ---- (g) reversibility ----
def test_clear_overrides_reverts(tmp_path):
    m = _mem(tmp_path)
    tuning.set_override(m, "EXPERIENCE_MIN_RELEVANCE", 0.85)
    assert ex._min_relevance() == 0.85
    removed = tuning.clear_overrides(m)
    assert removed == 1
    assert abs(ex._min_relevance() - 0.62) < 1e-9          # back to stock default


# ---- (h) a pinned candidate survives a refresh ----
def test_pinned_candidate_not_clobbered_by_refresh(tmp_path):
    m = _mem(tmp_path)
    m.set_tuned_config("EXPERIENCE_MIN_RELEVANCE", 0.70)    # persisted value
    with st.pinned({"EXPERIENCE_MIN_RELEVANCE": 0.88}):
        tuning.refresh(m)                                  # would normally load 0.70 — but it's pinned
        assert ex._min_relevance() == 0.88
    tuning.refresh(m, force=True)                          # pin released -> persisted value loads
    assert ex._min_relevance() == 0.70
