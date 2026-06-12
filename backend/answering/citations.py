"""
Citation validation + repair.

The model is told to cite sources with [n] matching the numbered evidence. Nothing stops it
from emitting [15] when only 8 sources were returned. These helpers make citation numbers
ALWAYS match the actual source list: out-of-range [n] are stripped (and grouped citations
like [1, 9, 3] keep only the valid members). Pure functions — no I/O, easy to test.
"""
from __future__ import annotations

import re
from typing import List, Set, Tuple

# A citation token: [n] or a grouped list [n, m, ...].
_GROUP_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_NUM_RE = re.compile(r"\d+")


def find_citations(text: str) -> Set[int]:
    """All distinct citation numbers referenced anywhere in `text`."""
    nums: Set[int] = set()
    for grp in _GROUP_RE.findall(text or ""):
        nums.update(int(n) for n in _NUM_RE.findall(grp))
    return nums


def validate_citations(text: str, n_sources: int) -> Tuple[Set[int], Set[int]]:
    """Return (valid, invalid) citation numbers — valid are in [1, n_sources]."""
    nums = find_citations(text)
    valid = {n for n in nums if 1 <= n <= n_sources}
    invalid = {n for n in nums if n < 1 or n > n_sources}
    return valid, invalid


def repair_citations(text: str, n_sources: int) -> Tuple[str, List[int]]:
    """Strip citation numbers outside [1, n_sources]. Returns (repaired_text, removed_sorted).
    Grouped citations keep their valid members ([1, 9, 3] -> [1, 3]); a citation with no
    valid members is removed entirely. Duplicates and valid citations are left untouched."""
    if not text:
        return text, []
    removed: List[int] = []

    def _fix(match: "re.Match") -> str:
        kept: List[str] = []
        for n_str in _NUM_RE.findall(match.group(1)):
            n = int(n_str)
            if 1 <= n <= n_sources:
                kept.append(str(n))
            else:
                removed.append(n)
        if not kept:
            return ""                       # whole citation was invalid -> drop it
        return "[" + ", ".join(kept) + "]"

    repaired = _GROUP_RE.sub(_fix, text)
    # tidy whitespace left where a citation was removed (" word  ." -> " word.")
    repaired = re.sub(r"[ \t]+([.,;:!?])", r"\1", repaired)
    repaired = re.sub(r"[ \t]{2,}", " ", repaired)
    return repaired, sorted(set(removed))
