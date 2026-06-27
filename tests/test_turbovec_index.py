from pathlib import Path

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


# ---- incremental build: add only new chunks instead of rebuilding the whole corpus ----
def _local_sig(**o):
    defaults = {"embedding_provider": "local", "embedding_model": "bge-large-en-v1.5",
                "embedding_dim_env": "1024"}
    defaults.update(o)
    return _signature(**defaults)


def _local_manifest(sig=None, **o):
    base = dict(embedding_dim=1024)
    base.update(o)
    return _manifest(sig or _local_sig(), **base)


def test_incremental_cutoff_on_pure_addition():
    old = _local_manifest(_local_sig(chunk_count=3, min_chunk_id=10, max_chunk_id=30, id_sum=60))
    new = _local_sig(chunk_count=5, min_chunk_id=10, max_chunk_id=50, id_sum=150)  # added ids 40,50
    # in-range over id<=30 still == old (3 chunks, sum 60) -> pure addition, add chunks with id > 30
    assert tv._incremental_add_cutoff(old, new, in_range_count=3, in_range_sum=60) == 30


def test_incremental_none_on_deletion_in_old_range():
    old = _local_manifest(_local_sig(chunk_count=3, max_chunk_id=30, id_sum=60))
    new = _local_sig(chunk_count=4, max_chunk_id=40, id_sum=80)
    # a chunk in the old range was deleted -> in-range count/sum no longer match -> full rebuild
    assert tv._incremental_add_cutoff(old, new, in_range_count=2, in_range_sum=30) is None


def test_incremental_none_on_model_change():
    old = _local_manifest(_local_sig(chunk_count=3, max_chunk_id=30, id_sum=60))
    new = _local_sig(chunk_count=5, max_chunk_id=50, id_sum=150, embedding_model="bge-m3")
    assert tv._incremental_add_cutoff(old, new, in_range_count=3, in_range_sum=60) is None


def test_incremental_none_when_not_grown():
    old = _local_manifest(_local_sig(chunk_count=5, max_chunk_id=50, id_sum=150))
    new = _local_sig(chunk_count=5, max_chunk_id=50, id_sum=150)
    assert tv._incremental_add_cutoff(old, new, in_range_count=5, in_range_sum=150) is None


def test_incremental_none_without_manifest():
    assert tv._incremental_add_cutoff(None, _local_sig(), 0, 0) is None


def test_build_index_incremental_adds_only_new_chunks(monkeypatch, tmp_path):
    """End-to-end (mocked Oracle + index): a grown corpus adds ONLY the new chunks' vectors and updates
    the manifest, instead of re-reading/re-quantizing the whole corpus."""
    path = tmp_path / "chunks.tvim"
    path.write_bytes(b"EXISTING-INDEX")
    monkeypatch.setattr(tv, "index_path", lambda: path)
    monkeypatch.setattr(tv, "bit_width", lambda: 4)

    old_sig = _local_sig(chunk_count=3, min_chunk_id=10, max_chunk_id=30, id_sum=60)
    tv.write_manifest(_local_manifest(old_sig, vector_count=3, skipped_count=0), path)
    new_sig = _local_sig(chunk_count=5, min_chunk_id=10, max_chunk_id=50, id_sum=150)
    monkeypatch.setattr(tv, "oracle_signature", lambda conn=None: new_sig)

    class _Cur:
        _fetch = None
        _rows = None

        def execute(self, sql, binds=None):
            low = " ".join(sql.lower().split())
            if "count(*)" in low and "id <=" in low:
                self._fetch = (3, 60)                      # in-range matches old -> pure addition
            elif "id >" in low:
                self._rows = [(40, "[1, 0, 0, 0]"), (50, "[0, 1, 0, 0]")]   # 4-dim to match embedding_dim

        def fetchone(self):
            return self._fetch

        def fetchmany(self, n):
            r, self._rows = (self._rows or []), []
            return r

        def close(self):
            pass

    monkeypatch.setattr(tv, "connect", lambda: type("C", (), {"cursor": lambda s: _Cur(), "close": lambda s: None})())
    # embedding_dim in the manifest must match the test vectors' dim (4)
    tv.write_manifest(_local_manifest(old_sig, vector_count=3, skipped_count=0, embedding_dim=4), path)

    added = {"ids": []}

    class _Idx:
        @staticmethod
        def load(p):
            return _Idx()

        def add_with_ids(self, arr, ids):
            added["ids"].extend(int(x) for x in ids.tolist())

        def prepare(self):
            pass

        def write(self, p):
            Path(p).write_bytes(b"UPDATED-INDEX")

    monkeypatch.setattr(tv, "_load_id_map_class", lambda: _Idx)

    stats = tv.build_index(force=False, prepare=False)
    assert stats.incremental is True and stats.rebuilt is False
    assert added["ids"] == [40, 50]                        # ONLY new chunks added (not 10, 20, 30)
    assert stats.vector_count == 5                         # 3 existing + 2 new
    updated = tv.load_manifest(path)
    assert updated["source_signature"]["chunk_count"] == 5  # manifest advanced to the new signature
