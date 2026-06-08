"""
Pre-execution lifecycle hook for the agent's code runner.

Idea adapted from kimi-code's "lifecycle hooks" — local command-execution gates for
audit/automation. Before any AI-written program runs in the Docker sandbox, this hook:

  1. AUDITS it  — appends a JSON line (timestamp, task, code hash, length, decision)
                  to a log, so every program the agent runs is reviewable.
  2. GATES it   — a policy may ALLOW or BLOCK the run:
                    - AGENT_BLOCK_PATTERNS : comma-separated regexes; a match blocks.
                    - AGENT_PRERUN_HOOK    : a shell command; the code is piped to its
                                             stdin; a non-zero exit blocks the run.
                  Default policy is ALLOW (the throwaway, network-less container is the
                  real safety boundary; this hook adds transparency + defense in depth).

Original implementation — no code copied from kimi-code (which is TypeScript).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# data/logs/ is gitignored, so audit records never get committed.
AUDIT_LOG = os.getenv("AGENT_AUDIT_LOG", str(ROOT / "data" / "logs" / "agent_audit.jsonl"))
BLOCK_PATTERNS = os.getenv("AGENT_BLOCK_PATTERNS", "")
PRERUN_HOOK = os.getenv("AGENT_PRERUN_HOOK", "")


@dataclass
class HookDecision:
    allowed: bool
    reason: str = ""


def _code_sha(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8", "replace")).hexdigest()[:12]


def _audit(task: str, code: str, decision: HookDecision) -> None:
    """Append one JSON record. Best-effort — never breaks a run."""
    path = (AUDIT_LOG or "").strip()
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task": (task or "")[:300],
            "code_sha": _code_sha(code),
            "code_len": len(code),
            "allowed": decision.allowed,
            "reason": decision.reason,
        }
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _policy(code: str) -> HookDecision:
    # 1. Regex blocklist (off unless configured).
    for raw in (pat.strip() for pat in BLOCK_PATTERNS.split(",")):
        if not raw:
            continue
        try:
            if re.search(raw, code):
                return HookDecision(False, f"matched blocked pattern /{raw}/")
        except re.error:
            continue  # a bad regex never blocks
    # 2. External pre-run hook command (off unless configured).
    cmd = PRERUN_HOOK.strip()
    if cmd and shutil.which(cmd.split()[0]):
        try:
            r = subprocess.run(cmd, shell=True, input=code, capture_output=True,
                               text=True, timeout=15)
            if r.returncode != 0:
                msg = (r.stdout or r.stderr or "").strip()[:200]
                return HookDecision(False, f"pre-run hook rejected it: {msg or 'non-zero exit'}")
        except Exception as exc:
            return HookDecision(False, f"pre-run hook error: {exc}")
    return HookDecision(True, "ok")


def pre_run(code: str, *, task: str = "") -> HookDecision:
    """Audit the program, apply the policy, and return the allow/block decision."""
    decision = _policy(code or "")
    _audit(task, code or "", decision)
    return decision
