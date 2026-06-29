"""Guards the hard-eval gold matcher. Stored paper titles collapse punctuation (e.g. 'Miipher-2' ->
'Miipher 2'), so matching MUST be punctuation-normalized on both sides — a naive substring match
silently false-misses real hits (the exact bug this eval hit during development)."""
from backend.evaluation.cluster_eval import _normalize, gold_rank


def test_normalize_collapses_punctuation():
    assert _normalize("Miipher-2") == "miipher 2"
    assert _normalize("Resolution-Aware!!") == "resolution aware"
    assert _normalize("Radiation-induced") == "radiation induced"
    assert _normalize(None) == ""


def test_gold_rank_matches_across_punctuation_difference():
    # gold key is hyphenated; stored titles are not -> must still match (the false-miss bug)
    results = [
        {"title": "ReverbMiipher  Generative Speech Restoration"},
        {"title": "Miipher 2  A Universal Speech Restoration Model"},
    ]
    assert gold_rank(results, "Miipher-2") == 2          # rank 2, not a miss
    assert gold_rank(results, "Resolution-Aware") is None


def test_gold_rank_returns_first_match_rank():
    results = [{"title": "GreenPeas Unlocking Adaptive Quantum Error Correction"},
               {"title": "Quantum error correction with the toric code"}]
    assert gold_rank(results, "GreenPeas") == 1
    assert gold_rank(results, "toric code") == 2


def test_gold_rank_empty_key_is_none():
    assert gold_rank([{"title": "anything"}], "") is None
