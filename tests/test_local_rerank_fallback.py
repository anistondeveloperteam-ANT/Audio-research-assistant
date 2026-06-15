"""Local retrieval degrades gracefully when the cross-encoder reranker is off or can't load
(e.g. a low-memory host that OOMs torch): rerank() falls back to a cheap lexical scorer that keeps
rerank_score on the same ~0..1 scale the pipeline + CRAG grader expect. Fully offline."""
import backend.retrieval.hybrid_retrieve as hr


def _cands():
    return [
        {"title": "MVDR beamforming", "section": "", "concepts": "",
         "text": "steering vector and noise covariance matrix"},
        {"title": "Unrelated", "section": "", "concepts": "", "text": "cooking recipes for pasta"},
    ]


def test_lexical_rerank_scores_in_unit_range_and_ranked():
    out = hr._lexical_rerank("mvdr beamforming steering vector", _cands(), top_k=2)
    assert all(0.0 <= c["rerank_score"] <= 1.0 for c in out)
    assert out[0]["title"] == "MVDR beamforming"          # the relevant doc ranks first
    assert out[0]["rerank_score"] > out[1]["rerank_score"]


def test_lexical_rerank_handles_empty_query():
    out = hr._lexical_rerank("", _cands(), top_k=2)
    assert all(c["rerank_score"] == 0.0 for c in out)


def test_rerank_uses_lexical_when_flag_disabled(monkeypatch):
    monkeypatch.setattr(hr, "LOCAL_RERANK_CROSS_ENCODER", False)

    def must_not_load():
        raise AssertionError("get_reranker must not be called when the flag is off")

    monkeypatch.setattr(hr, "get_reranker", must_not_load)
    out = hr.rerank("mvdr beamforming steering vector", _cands(), top_k=2)
    assert out and out[0]["title"] == "MVDR beamforming"
    assert all(0.0 <= c["rerank_score"] <= 1.0 for c in out)


def test_rerank_falls_back_to_lexical_when_cross_encoder_raises(monkeypatch):
    monkeypatch.setattr(hr, "LOCAL_RERANK_CROSS_ENCODER", True)

    def oom():
        raise OSError("The paging file is too small for this operation to complete")

    monkeypatch.setattr(hr, "get_reranker", oom)
    out = hr.rerank("mvdr beamforming steering vector", _cands(), top_k=2)
    assert out and out[0]["title"] == "MVDR beamforming"   # degraded, still functional
    assert all(0.0 <= c["rerank_score"] <= 1.0 for c in out)


def test_rerank_empty_candidates_returns_empty():
    assert hr.rerank("anything", [], top_k=5) == []
