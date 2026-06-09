from backend.retrieval import turbovec_index as tv


def _signature(**overrides):
    base = {
        "chunk_count": 3,
        "min_chunk_id": 10,
        "max_chunk_id": 30,
        "id_sum": 60,
        "embedding_vec_count": 3,
        "bit_width": 4,
        "embedding_provider": "google",
        "embedding_model": "gemini-embedding-2",
        "embedding_dim_env": "768",
    }
    base.update(overrides)
    return base


def _manifest(signature=None, **overrides):
    sig = signature or _signature()
    base = {
        "schema_version": 1,
        "source": "oracle_chunks_embedding",
        "source_signature": sig,
        "vector_count": 3,
        "skipped_count": 0,
        "embedding_dim": 768,
        "bit_width": 4,
    }
    base.update(overrides)
    return base


def test_turbovec_enabled_from_backend(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "turbovec")
    monkeypatch.setenv("TURBOVEC_ENABLED", "false")
    assert tv.turbovec_enabled() is True


def test_turbovec_enabled_from_flag(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "oracle")
    monkeypatch.setenv("TURBOVEC_ENABLED", "true")
    assert tv.turbovec_enabled() is True


def test_manifest_matches_valid_signature():
    sig = _signature()
    assert tv.manifest_matches(_manifest(sig), sig) is True


def test_manifest_rejects_stale_chunk_signature():
    old = _signature(chunk_count=3, max_chunk_id=30, id_sum=60)
    new = _signature(chunk_count=4, max_chunk_id=40, id_sum=100)
    assert tv.manifest_matches(_manifest(old), new) is False


def test_manifest_allows_recorded_skips():
    sig = _signature(chunk_count=5)
    manifest = _manifest(sig, vector_count=4, skipped_count=1)
    assert tv.manifest_matches(manifest, sig) is True


def test_parse_embedding_rejects_bad_values():
    assert tv.parse_embedding("[1, 2, 3]", expected_dim=3) == [1.0, 2.0, 3.0]
    assert tv.parse_embedding("[1, 2, 3]", expected_dim=4) is None
    assert tv.parse_embedding("[1, \"x\"]") is None
    assert tv.parse_embedding("[1, NaN]") is None
    assert tv.parse_embedding("{}") is None


def test_rows_to_results_preserves_search_order_and_schema():
    rows = {
        20: {"id": 20, "title": "B", "section": "Methods", "text": "beta"},
        10: {"id": 10, "title": "A", "section": "Intro", "text": "alpha"},
    }
    out = tv.rows_to_results([20, 99, 10], [0.9, 0.8, 0.7], rows, top_k=5)
    assert [r["id"] for r in out] == [20, 10]
    assert out[0]["vector_score"] == 0.9
    assert round(out[0]["distance"], 6) == 0.1
    assert out[0]["source"] == "turbovec_vector"
