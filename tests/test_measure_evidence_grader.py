"""Tests for the CRAG evidence-grader measurement harness (measure_evidence_grader.py).

These check the metric math + that the report is deterministic and offline, mirroring
tests/test_measure_classifiers.py."""
import pytest

from backend.answering import evidence_grader as eg
from backend.evaluation import measure_evidence_grader as meg


@pytest.fixture(autouse=True)
def _default_thresholds(monkeypatch):
    # Pin the grader thresholds so the eval is independent of whatever .env carries.
    monkeypatch.setenv("CRAG_STRONG_MIN", "0.55")
    monkeypatch.setenv("CRAG_PARTIAL_MIN", "0.30")
    monkeypatch.setenv("CRAG_STRONG_COUNT", "2")


def test_wilson_interval_basic():
    lo, hi = meg._wilson(8, 10)
    assert 0.0 <= lo < 0.8 < hi <= 1.0
    assert meg._wilson(0, 0) == (0.0, 0.0)


def test_confusion_matrix_totals_match_dataset():
    cm, misses = meg._confusion(meg.GRADES)
    assert sum(sum(row) for row in cm) == len(meg.GRADES)
    # Exactly the two curated boundary disagreements are misclassified.
    assert len(misses) == 2
    kinds = {(lbl, pr) for _d, lbl, pr in misses}
    assert (eg.STRONG, eg.PARTIAL) in kinds      # sub-bar chunks under-graded (safe)
    assert (eg.PARTIAL, eg.STRONG) in kinds      # barely-over chunks over-graded


def test_overall_accuracy_is_honest_not_trivial():
    cm, _ = meg._confusion(meg.GRADES)
    _per, acc, _macro, _wt = meg._per_class(cm)
    assert 0.70 <= acc < 1.0                     # boundary cases keep it below a trivial 100%


def test_skip_stats_counts_and_precision():
    s = meg.skip_stats(meg.GRADES)
    assert s["skipped"] == 4                      # 3 clear STRONG + 1 over-graded boundary
    assert s["correct_skip"] == 3
    assert abs(s["skip_precision"] - 0.75) < 1e-9


def test_build_report_is_deterministic_and_offline():
    r1 = meg.build_report()
    r2 = meg.build_report()
    assert r1 == r2
    assert "CRAG Evidence-Grader Measurement Report" in r1
    assert "Confusion matrix" in r1 and "External-skip rate" in r1


def test_per_class_metrics_are_well_formed():
    cm, _ = meg._confusion(meg.GRADES)
    per, acc, macro, wt = meg._per_class(cm)
    for c in meg.CLASSES:
        assert 0.0 <= per[c]["precision"] <= 1.0
        assert 0.0 <= per[c]["recall"] <= 1.0
    assert 0.0 <= macro["f1"] <= 1.0 and 0.0 <= wt["f1"] <= 1.0
