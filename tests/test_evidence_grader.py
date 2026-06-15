"""Unit tests for the CRAG evidence grader (backend/answering/evidence_grader.py)."""
from backend.answering import evidence_grader as eg


def _item(score, title="Paper A", text="some text", section=""):
    return {"score": score, "title": title, "text": text, "section": section}


# ----------------------------------------------------------------------
# grade_evidence
# ----------------------------------------------------------------------
def test_grade_strong_when_enough_high_relevance_chunks():
    # two chunks >= 0.55 -> STRONG (default CRAG_STRONG_COUNT=2)
    items = [_item(0.71), _item(0.61), _item(0.40)]
    assert eg.grade_evidence(items) == eg.STRONG


def test_grade_partial_when_relevant_but_thin():
    # one strong is not enough for the count; some chunks clear the partial floor -> PARTIAL
    items = [_item(0.62), _item(0.35), _item(0.10)]
    assert eg.grade_evidence(items) == eg.PARTIAL


def test_grade_none_when_nothing_relevant():
    items = [_item(0.21), _item(0.05)]
    assert eg.grade_evidence(items) == eg.NONE


def test_grade_none_when_empty():
    assert eg.grade_evidence([]) == eg.NONE


def test_grade_reads_rerank_score_field():
    items = [{"rerank_score": 0.8}, {"rerank_score": 0.7}]
    assert eg.grade_evidence(items) == eg.STRONG


def test_grade_respects_env_threshold_overrides(monkeypatch):
    monkeypatch.setenv("CRAG_STRONG_MIN", "0.85")
    # 0.71/0.61 were STRONG by default; with the higher bar they drop to PARTIAL
    items = [_item(0.71), _item(0.61)]
    assert eg.grade_evidence(items) == eg.PARTIAL


def test_grade_respects_strong_count_override(monkeypatch):
    monkeypatch.setenv("CRAG_STRONG_COUNT", "1")
    items = [_item(0.60), _item(0.20)]  # one strong chunk now suffices
    assert eg.grade_evidence(items) == eg.STRONG


# ----------------------------------------------------------------------
# relevant_items / paper_is_thin
# ----------------------------------------------------------------------
def test_relevant_items_filters_and_sorts_best_first():
    items = [_item(0.35), _item(0.80), _item(0.10), _item(0.50)]
    rel = eg.relevant_items(items)
    scores = [eg._item_score(r) for r in rel]
    assert scores == [0.80, 0.50, 0.35]  # 0.10 dropped, sorted desc


def test_paper_is_thin_true_when_few_relevant_chunks():
    assert eg.paper_is_thin([_item(0.90)]) is True  # 1 < default 2


def test_paper_is_thin_false_when_enough_relevant_chunks():
    assert eg.paper_is_thin([_item(0.90), _item(0.70), _item(0.40)]) is False


# ----------------------------------------------------------------------
# extract_algorithm_spec
# ----------------------------------------------------------------------
def test_extract_algorithm_spec_builds_spec_and_citation():
    items = [
        _item(0.80, title="MVDR Beamforming", text="Compute the steering vector then ...",
              section="Method"),
        _item(0.60, title="MVDR Beamforming", text="The covariance matrix is estimated ..."),
        _item(0.10, title="Irrelevant", text="ignore me"),
    ]
    spec, citation = eg.extract_algorithm_spec(items)
    assert "MVDR Beamforming" in citation
    assert "steering vector" in spec
    assert "covariance matrix" in spec
    assert "ignore me" not in spec  # below the relevance floor
    assert "Method" in spec  # section header included


def test_extract_algorithm_spec_empty_when_no_relevant_evidence():
    spec, citation = eg.extract_algorithm_spec([_item(0.10), _item(0.05)])
    assert spec == ""
    assert citation == ""


def test_extract_algorithm_spec_respects_max_chars():
    big = "x" * 5000
    items = [_item(0.80, text=big), _item(0.70, text=big)]
    spec, _ = eg.extract_algorithm_spec(items, max_chars=2000)
    # first block alone exceeds the budget, so only one block is kept (plus header)
    assert spec.count("[from ") == 1
