"""The classifier-measurement tool: metric math is correct and the report is deterministic +
fully offline (regex layer, no LLM/network)."""
from backend.evaluation import measure_classifiers as mc


def test_binary_metrics_math():
    m = mc._binary(tp=2, fp=1, tn=3, fn=4)
    assert m["support"] == 10
    assert abs(m["accuracy"] - 0.5) < 1e-9
    assert abs(m["precision"] - 2 / 3) < 1e-9
    assert abs(m["recall_sensitivity_tpr"] - 1 / 3) < 1e-9
    assert abs(m["specificity_tnr"] - 0.75) < 1e-9
    assert abs(m["f1"] - (2 * (2 / 3) * (1 / 3)) / ((2 / 3) + (1 / 3))) < 1e-9
    assert -1.0 <= m["mcc"] <= 1.0 and 0.0 <= m["cohen_kappa"] <= 1.0


def test_perfect_classifier_metrics_are_one():
    m = mc._binary(tp=5, fp=0, tn=5, fn=0)
    for k in ("accuracy", "precision", "recall_sensitivity_tpr", "specificity_tnr", "f1"):
        assert abs(m[k] - 1.0) < 1e-9
    assert abs(m["mcc"] - 1.0) < 1e-9


def test_run_binary_counts_and_misses():
    data = [("yes", 1), ("yes", 1), ("no", 0), ("yes", 0)]
    m, misses = mc._run_binary(lambda t: t == "yes", data)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 1, 0)
    assert len(misses) == 1 and misses[0][0] == "yes"


def test_run_multiclass_confusion_and_accuracy():
    data = [("i1", "x"), ("i2", "y"), ("i3", "x")]
    preds = {"i1": "x", "i2": "y", "i3": "y"}        # i3 mislabeled (true x, pred y)
    cm, per, acc, macro, wt, misses = mc._run_multiclass(lambda t: preds[t], data, ["x", "y"])
    assert cm == [[1, 1], [0, 1]]                    # rows=actual, cols=pred
    assert abs(acc - 2 / 3) < 1e-9
    assert per["x"]["support"] == 2 and len(misses) == 1


def test_build_report_is_deterministic_and_offline():
    r1 = mc.build_report()
    r2 = mc.build_report()
    assert r1 == r2                                   # deterministic (no randomness, no network)
    assert "Confusion matrix" in r1
    for section in ("Code-intent router", "Task-type classifier",
                    "Query-sanity gate", "Answer-reuse safety"):
        assert section in r1
