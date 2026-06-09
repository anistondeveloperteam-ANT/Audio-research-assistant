"""
Optional turbovec accelerator for local dense-vector retrieval.

Oracle remains the source of truth for papers, chunks, metadata, and citations.
This module builds a compressed turbovec IdMapIndex from the Oracle chunk
embeddings, using chunks.id as the stable external id. At query time it returns
the same result shape as vector_retriever.vector_search(), then the existing
BM25/RRF/rerank/MMR pipeline continues unchanged.

Usage:
    python -m backend.retrieval.turbovec_index status
    python -m backend.retrieval.turbovec_index build
    python -m backend.retrieval.turbovec_index clear
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = ROOT / "data" / "vector_cache" / "chunks.tvim"
MANIFEST_SUFFIX = ".manifest.json"

_INDEX_CACHE: Any = None
_MANIFEST_CACHE: Optional[Dict[str, Any]] = None
_CACHE_PATH: Optional[Path] = None


class TurbovecUnavailable(RuntimeError):
    """Raised when the optional turbovec backend cannot serve a query."""


@dataclass
class BuildStats:
    index_path: Path
    manifest_path: Path
    vector_count: int
    skipped_count: int
    embedding_dim: int
    bit_width: int
    rebuilt: bool
    seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_path": str(self.index_path),
            "manifest_path": str(self.manifest_path),
            "vector_count": self.vector_count,
            "skipped_count": self.skipped_count,
            "embedding_dim": self.embedding_dim,
            "bit_width": self.bit_width,
            "rebuilt": self.rebuilt,
            "seconds": round(self.seconds, 3),
        }


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def index_path() -> Path:
    raw = (os.getenv("TURBOVEC_INDEX_PATH") or "").strip()
    if not raw:
        return DEFAULT_INDEX_PATH
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def manifest_path(path: Optional[Path] = None) -> Path:
    path = path or index_path()
    return path.with_suffix(path.suffix + MANIFEST_SUFFIX)


def vector_backend() -> str:
    return (os.getenv("VECTOR_BACKEND") or "oracle").strip().lower()


def turbovec_enabled() -> bool:
    backend = vector_backend()
    return backend in {"turbovec", "turboquant"} or env_flag("TURBOVEC_ENABLED", False)


def strict_enabled() -> bool:
    return env_flag("TURBOVEC_STRICT", False)


def autobuild_enabled() -> bool:
    return env_flag("TURBOVEC_AUTOBUILD", True)


def build_in_pipeline_enabled() -> bool:
    return turbovec_enabled() or env_flag("TURBOVEC_BUILD_IN_PIPELINE", False)


def bit_width() -> int:
    try:
        value = int(os.getenv("TURBOVEC_BIT_WIDTH", "4"))
    except ValueError:
        return 4
    return value if value in {2, 4} else 4


def overfetch_multiplier() -> int:
    try:
        value = int(os.getenv("TURBOVEC_OVERFETCH", "3"))
    except ValueError:
        return 3
    return max(1, min(20, value))


def build_batch_size() -> int:
    try:
        value = int(os.getenv("TURBOVEC_BUILD_BATCH", "4096"))
    except ValueError:
        return 4096
    return max(1, value)


def read_lob(value: Any) -> str:
    if value is None:
        return ""
    try:
        if hasattr(value, "read"):
            return value.read()
    except Exception:
        return str(value)
    return str(value)


def connect():
    import oracledb

    return oracledb.connect(
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PASSWORD"),
        dsn=os.getenv("ORACLE_DSN"),
    )


def _turbovec_version() -> str:
    try:
        from importlib.metadata import version

        return version("turbovec")
    except Exception:
        return "unknown"


def _load_id_map_class():
    try:
        from turbovec import IdMapIndex

        return IdMapIndex
    except Exception as exc:
        raise TurbovecUnavailable(
            "turbovec is not installed. Run `pip install -r requirements.txt` "
            "after updating the project dependencies."
        ) from exc


def _embedding_identity() -> Dict[str, Any]:
    return {
        "embedding_provider": (os.getenv("EMBEDDING_PROVIDER") or "local").strip().lower(),
        "embedding_model": os.getenv("EMBEDDING_MODEL", ""),
        "embedding_dim_env": os.getenv("EMBEDDING_DIM", ""),
    }


def oracle_signature(conn=None) -> Dict[str, Any]:
    owned = conn is None
    conn = conn or connect()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT
                COUNT(*),
                COALESCE(MIN(id), 0),
                COALESCE(MAX(id), 0),
                COALESCE(SUM(id), 0)
            FROM chunks
            WHERE embedding IS NOT NULL
            """
        )
        count, min_id, max_id, id_sum = cur.fetchone()
        try:
            cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding_vec IS NOT NULL")
            vector_column_count = int(cur.fetchone()[0])
        except Exception:
            vector_column_count = None
    finally:
        cur.close()
        if owned:
            conn.close()

    out = {
        "chunk_count": int(count or 0),
        "min_chunk_id": int(min_id or 0),
        "max_chunk_id": int(max_id or 0),
        "id_sum": int(id_sum or 0),
        "embedding_vec_count": vector_column_count,
        "bit_width": bit_width(),
    }
    out.update(_embedding_identity())
    return out


def _signature_core(signature: Mapping[str, Any]) -> Dict[str, Any]:
    keys = [
        "chunk_count",
        "min_chunk_id",
        "max_chunk_id",
        "id_sum",
        "bit_width",
        "embedding_provider",
        "embedding_model",
        "embedding_dim_env",
    ]
    return {k: signature.get(k) for k in keys}


def manifest_matches(manifest: Mapping[str, Any], signature: Mapping[str, Any]) -> bool:
    if not manifest or manifest.get("schema_version") != 1:
        return False
    if manifest.get("source") != "oracle_chunks_embedding":
        return False
    if _signature_core(manifest.get("source_signature") or {}) != _signature_core(signature):
        return False
    try:
        vector_count = int(manifest.get("vector_count", 0))
        skipped_count = int(manifest.get("skipped_count", 0))
        chunk_count = int(signature.get("chunk_count", 0))
    except (TypeError, ValueError):
        return False
    if vector_count <= 0:
        return False
    return vector_count + skipped_count == chunk_count


def load_manifest(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    mpath = manifest_path(path)
    if not mpath.exists():
        return None
    try:
        return json.loads(mpath.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_manifest(manifest: Mapping[str, Any], path: Optional[Path] = None) -> None:
    mpath = manifest_path(path)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _iter_embedding_rows(conn, batch_size: int) -> Iterable[List[Tuple[int, str]]]:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT id, embedding
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY id
            """
        )
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            yield [(int(chunk_id), read_lob(raw)) for chunk_id, raw in rows]
    finally:
        cur.close()


def parse_embedding(raw: str, expected_dim: Optional[int] = None) -> Optional[List[float]]:
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None
    try:
        vec = [float(x) for x in data]
    except (TypeError, ValueError):
        return None
    if expected_dim is not None and len(vec) != expected_dim:
        return None
    if any((not math.isfinite(x)) or abs(x) >= 1e16 for x in vec):
        return None
    return vec


def _flush_batch(index: Any, ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
    if not ids:
        return
    import numpy as np

    arr = np.ascontiguousarray(vectors, dtype=np.float32)
    id_arr = np.asarray(ids, dtype=np.uint64)
    index.add_with_ids(arr, id_arr)


def build_index(*, force: bool = False, prepare: bool = True) -> BuildStats:
    started = time.time()
    path = index_path()
    mpath = manifest_path(path)

    IdMapIndex = _load_id_map_class()
    conn = connect()
    try:
        signature = oracle_signature(conn)
        existing_manifest = load_manifest(path)
        if (
            not force
            and path.exists()
            and existing_manifest
            and manifest_matches(existing_manifest, signature)
        ):
            return BuildStats(
                index_path=path,
                manifest_path=mpath,
                vector_count=int(existing_manifest.get("vector_count", 0)),
                skipped_count=int(existing_manifest.get("skipped_count", 0)),
                embedding_dim=int(existing_manifest.get("embedding_dim", 0)),
                bit_width=int(existing_manifest.get("bit_width", bit_width())),
                rebuilt=False,
                seconds=time.time() - started,
            )

        if int(signature.get("chunk_count") or 0) <= 0:
            raise TurbovecUnavailable("No chunk embeddings were found in Oracle.")

        idx = IdMapIndex(bit_width=bit_width())
        expected_dim: Optional[int] = None
        added = 0
        skipped = 0
        pending_ids: List[int] = []
        pending_vectors: List[List[float]] = []
        batch_size = build_batch_size()

        for rows in _iter_embedding_rows(conn, batch_size=batch_size):
            for chunk_id, raw in rows:
                vec = parse_embedding(raw, expected_dim)
                if vec is None:
                    skipped += 1
                    continue
                if expected_dim is None:
                    expected_dim = len(vec)
                pending_ids.append(chunk_id)
                pending_vectors.append(vec)
            if len(pending_ids) >= batch_size:
                _flush_batch(idx, pending_ids, pending_vectors)
                added += len(pending_ids)
                pending_ids = []
                pending_vectors = []

        if pending_ids:
            _flush_batch(idx, pending_ids, pending_vectors)
            added += len(pending_ids)

        if added <= 0 or expected_dim is None:
            raise TurbovecUnavailable("No valid embeddings were available for turbovec.")

        if prepare:
            idx.prepare()

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        idx.write(str(tmp_path))
        tmp_path.replace(path)

        manifest = {
            "schema_version": 1,
            "source": "oracle_chunks_embedding",
            "built_at": datetime.now(timezone.utc).isoformat(),
            "source_signature": signature,
            "vector_count": added,
            "skipped_count": skipped,
            "embedding_dim": expected_dim,
            "bit_width": bit_width(),
            "turbovec_version": _turbovec_version(),
        }
        write_manifest(manifest, path)
        clear_cache()

        return BuildStats(
            index_path=path,
            manifest_path=mpath,
            vector_count=added,
            skipped_count=skipped,
            embedding_dim=expected_dim,
            bit_width=bit_width(),
            rebuilt=True,
            seconds=time.time() - started,
        )
    finally:
        conn.close()


def clear_cache() -> None:
    global _INDEX_CACHE, _MANIFEST_CACHE, _CACHE_PATH
    _INDEX_CACHE = None
    _MANIFEST_CACHE = None
    _CACHE_PATH = None


def delete_index_files() -> None:
    clear_cache()
    for path in (index_path(), manifest_path()):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def ensure_index() -> Tuple[Any, Dict[str, Any]]:
    global _INDEX_CACHE, _MANIFEST_CACHE, _CACHE_PATH

    path = index_path()
    signature = oracle_signature()

    if (
        _INDEX_CACHE is not None
        and _MANIFEST_CACHE is not None
        and _CACHE_PATH == path
        and manifest_matches(_MANIFEST_CACHE, signature)
    ):
        return _INDEX_CACHE, _MANIFEST_CACHE

    manifest = load_manifest(path)
    if not path.exists() or not manifest or not manifest_matches(manifest, signature):
        if not autobuild_enabled():
            raise TurbovecUnavailable("turbovec index is missing or stale.")
        build_index(force=True)
        manifest = load_manifest(path)
        signature = oracle_signature()
        if not path.exists() or not manifest or not manifest_matches(manifest, signature):
            raise TurbovecUnavailable("turbovec index could not be built.")

    IdMapIndex = _load_id_map_class()
    idx = IdMapIndex.load(str(path))
    idx.prepare()
    _INDEX_CACHE = idx
    _MANIFEST_CACHE = dict(manifest)
    _CACHE_PATH = path
    return idx, dict(manifest)


def _chunked(values: Sequence[int], size: int = 900) -> Iterable[Sequence[int]]:
    for i in range(0, len(values), size):
        yield values[i:i + size]


def fetch_chunks_by_ids(ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    if not ids:
        return {}

    conn = connect()
    cur = conn.cursor()
    out: Dict[int, Dict[str, Any]] = {}
    try:
        for group in _chunked(list(dict.fromkeys(int(x) for x in ids))):
            binds = {f"id{i}": int(chunk_id) for i, chunk_id in enumerate(group)}
            placeholders = ", ".join(f":id{i}" for i in range(len(group)))
            cur.execute(
                f"""
                SELECT
                    c.id,
                    p.title,
                    c.section_name,
                    c.chunk_text,
                    c.chunk_type,
                    c.page_start,
                    c.page_end,
                    c.audio_concepts
                FROM chunks c
                JOIN papers p ON p.id = c.paper_id
                WHERE c.id IN ({placeholders})
                """,
                binds,
            )
            for row in cur.fetchall():
                (
                    chunk_id,
                    title,
                    section,
                    text,
                    chunk_type,
                    page_start,
                    page_end,
                    concepts,
                ) = row
                out[int(chunk_id)] = {
                    "id": int(chunk_id),
                    "title": str(read_lob(title) or ""),
                    "section": str(read_lob(section) or ""),
                    "text": str(read_lob(text) or ""),
                    "chunk_type": str(read_lob(chunk_type) or ""),
                    "page_start": int(page_start) if page_start is not None else None,
                    "page_end": int(page_end) if page_end is not None else None,
                    "concepts": str(read_lob(concepts) or ""),
                }
    finally:
        cur.close()
        conn.close()
    return out


def rows_to_results(
    ordered_ids: Sequence[int],
    scores: Sequence[float],
    rows_by_id: Mapping[int, Mapping[str, Any]],
    *,
    top_k: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for chunk_id, score in zip(ordered_ids, scores):
        row = rows_by_id.get(int(chunk_id))
        if not row:
            continue
        score = float(score)
        item = dict(row)
        item["vector_score"] = score
        item["distance"] = 1.0 - score
        item["source"] = "turbovec_vector"
        results.append(item)
        if len(results) >= top_k:
            break
    return results


def search_by_vector(query_vector: Sequence[float], top_k: int = 10) -> List[Dict[str, Any]]:
    if top_k <= 0:
        return []
    if not turbovec_enabled():
        raise TurbovecUnavailable("turbovec backend is disabled.")

    idx, manifest = ensure_index()
    expected_dim = int(manifest.get("embedding_dim") or 0)
    if expected_dim and len(query_vector) != expected_dim:
        raise TurbovecUnavailable(
            f"query vector dim {len(query_vector)} does not match turbovec dim {expected_dim}"
        )

    import numpy as np

    fetch_k = max(top_k, top_k * overfetch_multiplier())
    q = np.ascontiguousarray([list(query_vector)], dtype=np.float32)
    scores_arr, ids_arr = idx.search(q, fetch_k)
    if ids_arr.size == 0:
        return []

    ordered_ids = [int(x) for x in ids_arr[0].tolist()]
    scores = [float(x) for x in scores_arr[0].tolist()]
    rows = fetch_chunks_by_ids(ordered_ids)
    return rows_to_results(ordered_ids, scores, rows, top_k=top_k)


def status() -> Dict[str, Any]:
    path = index_path()
    manifest = load_manifest(path)
    info: Dict[str, Any] = {
        "enabled": turbovec_enabled(),
        "backend": vector_backend(),
        "index_path": str(path),
        "manifest_path": str(manifest_path(path)),
        "index_exists": path.exists(),
        "manifest_exists": manifest is not None,
        "valid": False,
        "reason": "",
        "manifest": manifest,
    }
    try:
        signature = oracle_signature()
        info["oracle_signature"] = signature
        info["valid"] = bool(path.exists() and manifest and manifest_matches(manifest, signature))
        if not info["valid"]:
            info["reason"] = "missing or stale index"
    except Exception as exc:
        info["reason"] = f"could not inspect Oracle: {exc}"
    return info


def _print_status() -> int:
    info = status()
    print("turbovec status")
    print("-" * 40)
    print(f"Enabled        : {info['enabled']} ({info['backend']})")
    print(f"Index path     : {info['index_path']}")
    print(f"Index exists   : {info['index_exists']}")
    print(f"Manifest exists: {info['manifest_exists']}")
    print(f"Valid          : {info['valid']}")
    if info.get("reason"):
        print(f"Reason         : {info['reason']}")
    manifest = info.get("manifest") or {}
    if manifest:
        print(f"Vectors        : {manifest.get('vector_count', '?')}")
        print(f"Skipped        : {manifest.get('skipped_count', '?')}")
        print(f"Dim            : {manifest.get('embedding_dim', '?')}")
        print(f"Bit width      : {manifest.get('bit_width', '?')}")
        print(f"Built at       : {manifest.get('built_at', '?')}")
    return 0 if info.get("valid") else 1


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build or inspect the optional turbovec vector cache.")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("status", help="Show cache status.")
    build = sub.add_parser("build", help="Build or refresh the cache from Oracle embeddings.")
    build.add_argument("--force", action="store_true", help="Rebuild even if the manifest is already valid.")
    build.add_argument("--no-prepare", action="store_true", help="Skip search-cache warmup after building.")
    sub.add_parser("clear", help="Delete the persisted cache files.")
    args = parser.parse_args(argv)

    command = args.command or "status"
    if command == "status":
        return _print_status()
    if command == "clear":
        delete_index_files()
        print("turbovec cache cleared.")
        return 0
    if command == "build":
        stats = build_index(force=bool(args.force), prepare=not bool(args.no_prepare))
        action = "rebuilt" if stats.rebuilt else "already valid"
        print(f"turbovec cache {action}:")
        for key, value in stats.to_dict().items():
            print(f"  {key}: {value}")
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
