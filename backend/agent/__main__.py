"""
Run the research agent from the command line:

    python -m backend.agent "Find the fastest correct primality test up to 10^7"
    python -m backend.agent --iters 6 --no-search "Implement and benchmark quicksort vs mergesort"

It streams each THINK -> RUN -> REFLECT step, then prints the best working program,
its output, and a one-line answer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from backend.agent.loop import run_agent  # noqa: E402


def _rule(ch: str = "-") -> str:
    return ch * 72


def _print_event(e: dict) -> None:
    t = e.get("type")
    if t == "status":
        print(f"\n[search] {e['message']}")
    elif t == "warning":
        print(f"[warn] {e['message']}")
    elif t == "error":
        print(f"\n[error] {e['message']}")
    elif t == "context":
        if e.get("chars"):
            print(f"   found {e['chars']} chars of relevant background")
    elif t == "think":
        print(f"\n{_rule('=')}\n[THINK] {e['message']}")
    elif t == "code":
        print("\n--- generated program ---")
        print(e["code"])
        print("--- end program ---")
    elif t == "run":
        print(f"\n[RUN] {e['message']}")
    elif t == "run_result":
        print(f"   result: {e['summary']}")
        if e.get("stdout"):
            print("   stdout:")
            for line in e["stdout"].splitlines()[:20]:
                print(f"      {line}")
        if not e.get("ok") and e.get("stderr"):
            print("   stderr (tail):")
            for line in e["stderr"].splitlines()[-8:]:
                print(f"      {line}")
    elif t == "reflect":
        v = e.get("verdict", {})
        print(f"\n[REVIEW] success={v.get('success')} score={v.get('score')} done={v.get('done')}")
        if v.get("feedback"):
            print(f"   feedback: {v['feedback']}")


def main() -> int:
    p = argparse.ArgumentParser(description="Autonomous code-writing research agent.")
    p.add_argument("task", nargs="+", help="The problem to solve.")
    p.add_argument("--iters", type=int, default=None, help="Max THINK/RUN/REFLECT cycles.")
    p.add_argument("--no-search", action="store_true", help="Skip the web/paper research step.")
    args = p.parse_args()

    # Windows consoles default to cp1252; model output can contain any character.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    task = " ".join(args.task)
    kwargs = {"use_search": not args.no_search, "on_event": _print_event}
    if args.iters:
        kwargs["max_iters"] = args.iters

    res = run_agent(task, **kwargs)

    print(f"\n{_rule('=')}")
    print("BEST RESULT (verified)" if res.success else "BEST ATTEMPT (not fully verified)")
    print(_rule("="))
    if res.answer:
        print(f"\nAnswer: {res.answer}")
    if res.best_output:
        print("\nProgram output:")
        print(res.best_output)
    if res.best_code:
        print("\nBest program:\n")
        print(res.best_code)
    print(f"\n({len(res.attempts)} attempt(s))")
    return 0 if res.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
