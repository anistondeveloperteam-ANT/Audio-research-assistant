"""
Autonomous coding agent (write -> run -> test -> fix) built on the Claude Agent SDK.

⚠️  HOST EXECUTION — UNLIKE the user-code runner (code_runner.py, Docker sandbox),
this agent edits files and runs shell commands on the HOST. It is an *owner* dev
tool, not for untrusted user input. The HTTP layer gates it hard: disabled by
default (ENABLE_AUTO_AGENT), loopback-only, and login-required.

Prerequisites to actually run:
  - pip install claude-agent-sdk              (done)
  - npm install -g @anthropic-ai/claude-code  (the CLI the SDK drives)
  - Authenticate the CLI: run `claude` once and log in with a Pro/Max subscription,
    OR set ANTHROPIC_API_KEY. Either works — no key is required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]

# The exact tools + turn cap requested.
ALLOWED_TOOLS = ["Read", "Edit", "Write", "Bash", "Glob", "Grep"]
MAX_TURNS = 20

SYSTEM_PROMPT = (
    "You are an autonomous coding agent working inside this repository. Implement the "
    "user's task by iterating: write code, run it, run the tests, and fix failures until "
    "everything passes. Prefer small, focused diffs and match the existing code style. "
    "After changes, run `python -m pytest -q` and `pyflakes backend webapp`; if they fail, "
    "fix and re-run until clean. Never touch .env or secrets."
)


def sdk_available() -> bool:
    """True if claude-agent-sdk is importable."""
    import importlib.util
    return importlib.util.find_spec("claude_agent_sdk") is not None


def _summarize_blocks(content: Any) -> List[Dict[str, Any]]:
    """Turn an AssistantMessage's content blocks into compact step dicts."""
    from claude_agent_sdk import TextBlock, ToolUseBlock  # local import; SDK may be absent in tests
    steps: List[Dict[str, Any]] = []
    for block in (content or []):
        if isinstance(block, TextBlock):
            text = (block.text or "").strip()
            if text:
                steps.append({"kind": "text", "text": text[:2000]})
        elif isinstance(block, ToolUseBlock):
            steps.append({"kind": "tool", "name": block.name, "input": str(block.input)[:600]})
    return steps


async def stream_auto_agent(task: str, project_dir: Optional[str] = None,
                            *, max_turns: int = MAX_TURNS):
    """Async generator yielding the agent's progress as it works. Each item is one of:
       {"type": "step",   "kind": "text"|"tool", ...}
       {"type": "result", "num_turns": ..., "is_error": ..., "result": ...}
       {"type": "error",  "message": "..."}
    It never raises — failures are emitted as an "error" event so they can be streamed.
    Auth is the Claude CLI's (Pro/Max subscription via `claude setup-token`, or ANTHROPIC_API_KEY).
    """
    task = (task or "").strip()
    if not task:
        yield {"type": "error", "message": "task is required"}
        return
    try:
        from claude_agent_sdk import (
            query, ClaudeAgentOptions, AssistantMessage, ResultMessage, CLINotFoundError,
        )
    except Exception as exc:
        yield {"type": "error", "message": f"claude-agent-sdk is not installed: {exc}"}
        return

    options = ClaudeAgentOptions(
        cwd=project_dir or str(ROOT),
        max_turns=max_turns,
        allowed_tools=ALLOWED_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        permission_mode="bypassPermissions",   # headless: no interactive permission prompts
    )
    try:
        async for msg in query(prompt=task, options=options):
            if isinstance(msg, AssistantMessage):
                for step in _summarize_blocks(msg.content):
                    yield {"type": "step", **step}
            elif isinstance(msg, ResultMessage):
                yield {
                    "type": "result",
                    "is_error": getattr(msg, "is_error", None),
                    "num_turns": getattr(msg, "num_turns", None),
                    "total_cost_usd": getattr(msg, "total_cost_usd", None),
                    "result": getattr(msg, "result", None),
                }
    except CLINotFoundError as exc:
        yield {"type": "error", "message":
               "The Claude Code CLI is not installed (the SDK drives it). Run "
               "`npm install -g @anthropic-ai/claude-code`. Details: " + str(exc)}
    except Exception as exc:                                   # auth / process failures
        yield {"type": "error", "message":
               "The agent could not run. Authenticate the Claude CLI (`claude setup-token` with a "
               "Pro/Max subscription, or set ANTHROPIC_API_KEY). Details: " + str(exc)}


async def run_auto_agent(task: str, project_dir: Optional[str] = None,
                         *, max_turns: int = MAX_TURNS) -> Dict[str, Any]:
    """Collecting wrapper around stream_auto_agent: run the loop, print steps, return the
    transcript. Raises ValueError on empty input and RuntimeError on agent/auth failure."""
    if not (task or "").strip():
        raise ValueError("task is required")
    steps: List[Dict[str, Any]] = []
    result: Optional[Dict[str, Any]] = None
    async for ev in stream_auto_agent(task, project_dir, max_turns=max_turns):
        kind = ev.get("type")
        if kind == "step":
            steps.append(ev)
            label = ev.get("name") or ev.get("kind")
            print(f"  → {label}: {ev.get('text') or ev.get('input') or ''}"[:200], flush=True)
        elif kind == "result":
            result = ev
            print(f"  ✓ finished in {ev.get('num_turns')} turns", flush=True)
        elif kind == "error":
            raise RuntimeError(ev["message"])
    return {"task": (task or "").strip(), "max_turns": max_turns, "steps": steps, "result": result}


def _cli() -> None:
    """`python -m backend.agent.auto_agent "<task>"` — run the agent locally."""
    import asyncio
    import sys
    task = " ".join(sys.argv[1:]).strip()
    if not task:
        print('usage: python -m backend.agent.auto_agent "<task description>"')
        raise SystemExit(2)
    try:
        out = asyncio.run(run_auto_agent(task))
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}")
        raise SystemExit(1)
    print(f"\n=== done in {out['result'] and out['result'].get('num_turns')} turns ===")
    if out["result"]:
        print(out["result"].get("result") or "")


if __name__ == "__main__":
    _cli()
