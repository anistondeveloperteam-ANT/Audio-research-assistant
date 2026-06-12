"""Citation numbers must always match the actual returned sources."""
from backend.answering.citations import find_citations, repair_citations, validate_citations


def test_valid_citations_unchanged():
    text = "First claim [1]. Second claim [2][3]."
    repaired, removed = repair_citations(text, n_sources=3)
    assert repaired == text
    assert removed == []
    valid, invalid = validate_citations(text, 3)
    assert valid == {1, 2, 3} and invalid == set()


def test_invalid_citation_is_removed():
    # the classic: cite [15] when only 8 sources exist
    text = "Grounded claim [2]. Hallucinated source [15]."
    repaired, removed = repair_citations(text, n_sources=8)
    assert "[15]" not in repaired
    assert "[2]" in repaired
    assert removed == [15]
    _, invalid = validate_citations(text, 8)
    assert invalid == {15}


def test_duplicate_citations_are_kept():
    text = "A [1]. B [1]. C [1]."
    repaired, removed = repair_citations(text, n_sources=2)
    assert repaired == text          # duplicates are legitimate
    assert removed == []
    valid, invalid = validate_citations(text, 2)
    assert valid == {1} and invalid == set()


def test_answer_with_no_citations():
    text = "A complete answer that cites nothing at all."
    repaired, removed = repair_citations(text, n_sources=5)
    assert repaired == text
    assert removed == []
    assert find_citations(text) == set()


def test_grouped_citation_keeps_valid_members():
    text = "Mixed evidence [1, 9, 3] supports this."
    repaired, removed = repair_citations(text, n_sources=5)
    assert "9" not in repaired
    assert "[1, 3]" in repaired
    assert removed == [9]


def test_zero_sources_strips_all_citations():
    text = "No sources but cited [1] anyway."
    repaired, removed = repair_citations(text, n_sources=0)
    assert "[1]" not in repaired
    assert removed == [1]
