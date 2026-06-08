"""
Two-tier, constant-size agent memory.

Keeps the context the agent sees roughly fixed in size no matter how many cycles
it runs, so a long loop never bloats or grows expensive:

  Tier 1 — Brief : frozen, written once (the goal / constraints / decision tree).
  Tier 2 — Log   : a rolling record of attempts + decisions that auto-compacts
                   (drops the oldest entries) to stay under a character cap.

Design credit: the two-tier memory pattern comes from
`auto-deep-researcher-24x7` (Apache-2.0) — https://github.com/Xiangyue-Zhang/auto-deep-researcher-24x7
This is an original, from-scratch implementation of that idea (no source copied),
adapted to this project's code-writing agent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

BRIEF_MAX_CHARS = int(os.getenv("AGENT_BRIEF_MAX_CHARS", "3000"))
LOG_MAX_CHARS = int(os.getenv("AGENT_LOG_MAX_CHARS", "2000"))
LOG_KEEP_LAST = int(os.getenv("AGENT_LOG_KEEP_LAST", "12"))


@dataclass
class TwoTierMemory:
    """Frozen brief + auto-compacting log. `context()` is always ~constant size."""

    brief: str
    log_entries: List[str] = field(default_factory=list)
    brief_max: int = BRIEF_MAX_CHARS
    log_max: int = LOG_MAX_CHARS
    keep_last: int = LOG_KEEP_LAST

    def __post_init__(self) -> None:
        self.brief = _clip(self.brief.strip(), self.brief_max)

    def append(self, entry: str) -> None:
        """Record one attempt/decision, then compact so the log stays bounded."""
        entry = (entry or "").strip()
        if entry:
            self.log_entries.append(entry)
            self._compact()

    def _compact(self) -> None:
        # Hard cap on count, then on total characters — always drop the oldest first.
        if len(self.log_entries) > self.keep_last:
            self.log_entries = self.log_entries[-self.keep_last:]
        while len(self.log_entries) > 1 and self._log_chars() > self.log_max:
            self.log_entries.pop(0)

    def _log_chars(self) -> int:
        return sum(len(e) + 1 for e in self.log_entries)

    def recent_log(self) -> str:
        return "\n".join(f"- {e}" for e in self.log_entries)

    def context(self) -> str:
        """The constant-size view fed to the model each cycle: brief + recent log."""
        if not self.log_entries:
            return self.brief
        return f"{self.brief}\n\n## Progress so far (most recent last)\n{self.recent_log()}"


def _clip(text: str, limit: int) -> str:
    if text and len(text) > limit:
        return text[:limit].rstrip() + " …[clipped]"
    return text or ""
