"""Hard cluster-discrimination retrieval eval.

Each query describes ONE paper's contribution WITHOUT naming it, and must beat that paper's
topically-identical siblings (e.g. the 7 quantum-error-correction papers, the many speech-restoration
papers). Gold is PAPER-level (robust to chunk-boundary changes), matched by a distinctive title token.

IMPORTANT: matching is punctuation-normalized on BOTH sides — stored titles collapse punctuation
('Miipher-2' is stored as 'Miipher 2'), so a naive substring match silently false-misses. The
`_normalize` + `gold_rank` pair below (and test_cluster_eval.py) guard against that.

Run:  python -m backend.evaluation.cluster_eval
"""
from __future__ import annotations

import collections
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUESTIONS_FILE = ROOT / "data" / "eval_hard_questions.json"


def _normalize(s: Any) -> str:
    """Lowercase and collapse every run of non-alphanumeric characters to a single space, so
    'Miipher-2', 'Miipher 2' and 'Miipher_2' all compare equal (stored titles strip punctuation)."""
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def gold_rank(results: List[Dict[str, Any]], gold_key: str) -> Optional[int]:
    """1-indexed rank of the first result whose title contains the normalized gold key, else None."""
    g = _normalize(gold_key)
    if not g:
        return None
    for i, r in enumerate(results, 1):
        if g in _normalize(r.get("title", "")):
            return i
    return None


def load_questions() -> List[Dict[str, Any]]:
    return json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))


def run_eval(top_k: int = 10) -> Dict[str, Any]:
    """Run every question through the live retrieval pipeline and score paper-level recall@k + MRR."""
    from backend.retrieval.hybrid_retrieve import hybrid_retrieve

    questions = load_questions()
    if questions:
        hybrid_retrieve(questions[0]["question"], top_k=top_k)        # warm the reranker once

    recall = {1: 0, 3: 0, 5: 0, 10: 0}
    mrr = lat = 0.0
    by_cluster: Dict[str, List[int]] = collections.defaultdict(lambda: [0, 0])
    misses = []
    for q in questions:
        t = time.time()
        results = hybrid_retrieve(q["question"], top_k=top_k)
        lat += time.time() - t
        rank = gold_rank(results, q["gold_key"])
        for k in recall:
            if rank and rank <= k:
                recall[k] += 1
        if rank:
            mrr += 1.0 / rank
        cluster = q.get("cluster", "?")
        by_cluster[cluster][1] += 1
        if rank and rank <= 5:
            by_cluster[cluster][0] += 1
        if (not rank) or rank > 1:
            misses.append({"gold_key": q["gold_key"], "rank": rank, "question": q["question"]})

    n = len(questions) or 1
    return {
        "n": len(questions),
        "recall": {k: recall[k] / n for k in recall},
        "mrr": mrr / n,
        "latency_per_query": lat / n,
        "by_cluster_recall_at_5": {c: f"{h}/{t}" for c, (h, t) in sorted(by_cluster.items())},
        "misses": misses,
    }


def main() -> None:
    r = run_eval()
    print(f"Hard cluster-discrimination eval ({r['n']} queries):")
    for k in (1, 3, 5, 10):
        print(f"  paper_recall@{k}: {r['recall'][k]:.3f}")
    print(f"  MRR: {r['mrr']:.3f}   latency/query: {r['latency_per_query']:.2f}s")
    print(f"  per-cluster recall@5: {r['by_cluster_recall_at_5']}")
    print(f"  misses: {r['misses']}")


if __name__ == "__main__":
    main()
