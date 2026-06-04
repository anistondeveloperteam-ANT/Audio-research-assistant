"""Unit tests for frontend/chat_ui_utils.py (pure helpers, no Streamlit)."""
import hashlib

from chat_ui_utils import (
    compute_file_hash,
    parse_ollama_tags_response,
    patch_env_text,
    safe_pdf_target,
    list_existing_pdf_hashes,
)


def test_compute_file_hash_matches_sha256():
    assert compute_file_hash(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_parse_ollama_tags_response():
    payload = {"models": [{"name": "qwen2.5:7b"}, {"name": ""}, {"no_name": 1}]}
    assert parse_ollama_tags_response(payload) == ["qwen2.5:7b"]
    assert parse_ollama_tags_response({}) == []
    assert parse_ollama_tags_response(None) == []


def test_patch_env_text_replaces_existing_key():
    out = patch_env_text("A=1\nB=2\n", "A", "9")
    assert "A=9" in out
    assert "A=1" not in out
    assert "B=2" in out


def test_patch_env_text_appends_missing_key_and_keeps_comments():
    out = patch_env_text("# comment\nA=1\n", "C", "3")
    assert "# comment" in out
    assert "C=3" in out


def test_safe_pdf_target_avoids_collision(tmp_path):
    first = safe_pdf_target(tmp_path, "paper.pdf")
    assert first == tmp_path / "paper.pdf"
    first.write_bytes(b"x")
    second = safe_pdf_target(tmp_path, "paper.pdf")
    assert second == tmp_path / "paper_1.pdf"


def test_list_existing_pdf_hashes(tmp_path):
    (tmp_path / "x.pdf").write_bytes(b"data")
    result = list_existing_pdf_hashes(tmp_path)
    assert result == {hashlib.sha256(b"data").hexdigest(): "x.pdf"}


def test_list_existing_pdf_hashes_missing_dir(tmp_path):
    assert list_existing_pdf_hashes(tmp_path / "nope") == {}
