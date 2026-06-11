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
    """Turn an AssistantMessage's content blocks into compact transcript steps."""
    from claude_agent_sdk import TextBlock, ToolUseBlock  # local import; SDK may be absent in tests
    steps: List[Dict[str, Any]] = []
    for block in (content or []):
        if isinstance(block, TextBlock):
            text = (block.text or "").strip()
            if text:
                steps.append({"type": "text", "text": text[:2000]})
        elif isinstance(block, ToolUseBlock):
            steps.append({"type": "tool", "name": block.name, "input": str(block.input)[:600]})
    return steps


async def run_auto_agent(task: str, project_dir: Optional[str] = None,
                         *, max_turns: int = MAX_TURNS) -> Dict[str, Any]:
    """Run the autonomous write -> run -> test -> fix loop; stream + return a transcript.

    Auth is handled by the Claude Code CLI — either a Pro/Max subscription (run `claude`
    once and log in) OR an ANTHROPIC_API_KEY. No key is required here. Raises ValueError on
    bad input and RuntimeError (with a clear message) when the CLI is missing/unauthenticated.
    """
    task = (task or "").strip()
    if not task:
        raise ValueError("task is required")
    try:
        from claude_agent_sdk import (
            query, ClaudeAgentOptions, AssistantMessage, ResultMessage, CLINotFoundError,
        )
    except Exception as exc:
        raise RuntimeError(f"claude-agent-sdk is not installed: {exc}")

    options = ClaudeAgentOptions(
        cwd=project_dir or str(ROOT),
        max_turns=max_turns,
        allowed_tools=ALLOWED_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        permission_mode="bypassPermissions",   # headless: no interactive permission prompts
    )

    steps: List[Dict[str, Any]] = []
    result: Optional[Dict[str, Any]] = None
    try:
        async for msg in query(prompt=task, options=options):
            if isinstance(msg, AssistantMessage):
                for step in _summarize_blocks(msg.content):
                    steps.append(step)
                    label = step.get("name") or step["type"]
                    print(f"  → {label}: {step.get('text') or step.get('input') or ''}"[:200], flush=True)
            elif isinstance(msg, ResultMessage):
                result = {
                    "is_error": getattr(msg, "is_error", None),
                    "num_turns": getattr(msg, "num_turns", None),
                    "total_cost_usd": getattr(msg, "total_cost_usd", None),
                    "result": getattr(msg, "result", None),
                }
                print(f"  ✓ finished in {result.get('num_turns')} turns", flush=True)
    except CLINotFoundError as exc:
        raise RuntimeError(
            "The Claude Code CLI is not installed (the SDK drives it). Run "
            "`npm install -g @anthropic-ai/claude-code`. Details: " + str(exc))
    except Exception as exc:                                  # auth / process failures
        raise RuntimeError(
            "The agent could not run. Make sure the Claude CLI is authenticated — run `claude` "
            "and log in with your Pro/Max subscription (or set ANTHROPIC_API_KEY). "
            "Details: " + str(exc))

    return {"task": task, "max_turns": max_turns, "steps": steps, "result": result}


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
