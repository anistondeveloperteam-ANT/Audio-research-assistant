"""Long table chunks must be split into header-preserving row groups so each piece fits the embedder's
~512-token window (otherwise the tail of a big table is silently dropped from its vector)."""
from backend.ingestion.document_chunker import _split_markdown_table


def test_short_table_is_one_piece():
    md = "|A|B|\n|---|---|\n|1|2|"
    assert _split_markdown_table(md, max_chars=1000) == [md]


def test_long_table_splits_and_repeats_header():
    header = "|Model|PESQ|STOI|\n|---|---|---|"
    rows = "\n".join(f"|model_{i}|2.{i:02d}|0.9{i % 10}|" for i in range(40))
    md = header + "\n" + rows
    parts = _split_markdown_table(md, max_chars=200)
    assert len(parts) >= 2                                  # actually split
    for p in parts:
        assert p.startswith("|Model|PESQ|STOI|")            # header row repeated in each piece
        assert p.splitlines()[1] == "|---|---|---|"         # separator row repeated too
        assert len(p) <= 200 + 60                           # each piece within the window
    # no body row is lost
    assert sum(p.count("|model_") for p in parts) == 40


def test_giant_single_row_is_hard_capped():
    header = "|A|B|\n|---|---|"
    giant = "|" + "x" * 5000 + "|y|"               # one row far bigger than the window
    parts = _split_markdown_table(header + "\n" + giant, max_chars=400)
    assert parts and all(len(p) <= 400 + 60 for p in parts)   # nothing exceeds the window


def test_empty_table_returns_empty():
    assert _split_markdown_table("") == []
    assert _split_markdown_table("   ") == []
