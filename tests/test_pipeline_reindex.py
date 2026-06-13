"""pipeline.py --reindex target resolution (pure; no Oracle, no files)."""
import pipeline


def test_resolve_target_exact_match():
    names = ["DeepFilterNet.pdf", "Other.pdf"]
    assert pipeline._resolve_target("Other.pdf", names) == ("Other.pdf", None)


def test_resolve_target_unique_substring_case_insensitive():
    names = ["DeepFilterNet.pdf", "Other.pdf"]
    match, err = pipeline._resolve_target("filternet", names)
    assert match == "DeepFilterNet.pdf" and err is None


def test_resolve_target_no_match():
    match, err = pipeline._resolve_target("nope", ["A.pdf", "B.pdf"])
    assert match is None and "no PDF" in err


def test_resolve_target_ambiguous():
    names = ["DeepFilterNet.pdf", "deep_other.pdf"]
    match, err = pipeline._resolve_target("deep", names)
    assert match is None and "matches 2" in err
