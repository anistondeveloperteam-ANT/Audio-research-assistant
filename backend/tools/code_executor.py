"""
code_executor.py  --  Batch 11 (Phase 2)

Parent-side of the sandboxed code executor. Spawns a subprocess
running _sandbox_runner.py, pipes user source into stdin, reads the
JSON envelope from stdout, kills the child on timeout.

Defense-in-depth layers (in order):

  1.  AST-level scan of the source BEFORE spawning anything. Rejects
      code that imports disallowed modules, calls eval/exec/compile,
      uses attribute access to dangerous dunders, etc.
  2.  Subprocess isolation. The sandbox runs in a separate Python
      process. If it crashes, hangs, or somehow escapes its python-
      level restrictions, the parent UI keeps running.
  3.  Hard timeout. Default 30 seconds. The parent kills the child
      process if it doesn't return by then.
  4.  Import allowlist enforced inside the child (in _sandbox_runner).
      A second layer in case the AST scan missed something.
  5.  Builtins shim inside the child. open(), exec(), eval(), etc.
      are removed from the user's namespace.
  6.  Stdout capped at 10KB in the child.

Honest limits:
  - On Windows, the `resource` module isn't available, so we can't
    enforce memory or CPU caps the way Linux does. We rely on the
    timeout + import allowlist + AST screen.
  - This is NOT an airtight sandbox suitable for hostile attackers.
    It IS reasonable protection against LLM-generated code accidentally
    doing something bad like trying to read a file or open the network.

Public API:

    result = run_code(source, timeout_sec=30.0)

    result -> {
      "ok": bool,
      "error": str | None,          # short message
      "error_type": str | None,     # e.g. "ZeroDivisionError"
      "traceback": str | None,      # full python traceback (or None)
      "stdout": str,                # everything user code printed
      "plots": list[str],           # base64 PNGs from matplotlib
      "elapsed_sec": float,
      "killed_by_timeout": bool,    # True if we killed it
    }
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "sandbox_runner.py"


# Mirror of _sandbox_runner.ALLOWED_IMPORTS. Kept here as a separate
# copy so the parent doesn't need to import the child module (which
# has matplotlib import side effects).
ALLOWED_IMPORTS = frozenset({
    "math",
    "statistics",
    "random",
    "decimal",
    "fractions",
    "itertools",
    "functools",
    "operator",
    "collections",
    "re",
    "json",
    "string",
    "time",
    "datetime",
    "numpy",
    "scipy",
    "scipy.signal",
    "scipy.fft",
    "scipy.linalg",
    "scipy.stats",
    "scipy.special",
    "scipy.optimize",
    "scipy.interpolate",
    "scipy.io",
    "scipy.io.wavfile",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.figure",
    "matplotlib.colors",
    "matplotlib.cm",
    "pandas",
    "sympy",
    # Batch 12: pre-built DSP helpers
    "dsp_toolkit",
})


# Names we flat-out refuse to see anywhere in the AST. These are
# function calls or attribute accesses that have no legitimate use
# in a sandboxed analysis context.
FORBIDDEN_NAMES = frozenset({
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "exit",
    "quit",
    "breakpoint",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",      # too useful for attribute traversal attacks
    "setattr",
    "delattr",
    "memoryview",
})


# Attribute names that are dangerous regardless of which object they
# hang off of. (e.g. `something.__subclasses__()` can sometimes reach
# the file builtin even without `import os`.)
FORBIDDEN_ATTRS = frozenset({
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__class__",       # blocked to prevent class-walking escapes
    "__globals__",
    "__builtins__",
    "__loader__",
    "__import__",
    "__getattribute__",
    "__reduce__",
    "__reduce_ex__",
})


def _is_allowed_import(name: str) -> bool:
    if name in ALLOWED_IMPORTS:
        return True
    top = name.split(".")[0]
    if top in ALLOWED_IMPORTS:
        return True
    for allowed in ALLOWED_IMPORTS:
        if name.startswith(allowed + "."):
            return True
    return False


def ast_safety_screen(source: str) -> Optional[str]:
    """Return None if the AST looks safe; otherwise return a short
    string describing the first problem found. Does NOT execute the
    code -- just parses it."""
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} at line {exc.lineno}"

    for node in ast.walk(tree):
        # `import X` or `import X.Y`
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_allowed_import(alias.name):
                    return f"Import of {alias.name!r} is not allowed"
        # `from X import Y`
        elif isinstance(node, ast.ImportFrom):
            if node.module is None or not _is_allowed_import(node.module):
                return f"Import from {node.module!r} is not allowed"
        # Calls like `open(...)`, `eval(...)`, etc.
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_NAMES:
                return f"Call to {func.id!r} is not allowed"
        # Bare references to forbidden builtins -- catches `f = eval`
        elif isinstance(node, ast.Name):
            # Only flag if it's being LOADED (used), not stored to
            if isinstance(node.ctx, ast.Load) and node.id in FORBIDDEN_NAMES:
                return f"Reference to {node.id!r} is not allowed"
        # Attribute access to dangerous dunders
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                return f"Attribute {node.attr!r} is not allowed"

    return None


def run_code(
    source: str,
    timeout_sec: float = 30.0,
    skip_ast_screen: bool = False,
) -> Dict[str, Any]:
    """Run `source` in a sandboxed subprocess. See module docstring
    for the return shape.

    timeout_sec controls how long the parent waits before killing
    the child. Default 30s.
    """
    # Layer 1: AST screen
    if not skip_ast_screen:
        problem = ast_safety_screen(source)
        if problem is not None:
            return {
                "ok": False,
                "error": f"Blocked by safety check: {problem}",
                "error_type": "SandboxBlocked",
                "traceback": None,
                "stdout": "",
                "plots": [],
                "elapsed_sec": 0.0,
                "killed_by_timeout": False,
            }

    if not RUNNER.exists():
        return {
            "ok": False,
            "error": f"_sandbox_runner.py not found at {RUNNER}",
            "error_type": "ConfigError",
            "traceback": None,
            "stdout": "",
            "plots": [],
            "elapsed_sec": 0.0,
            "killed_by_timeout": False,
        }

    start = time.time()
    killed = False

    try:
        proc = subprocess.Popen(
            [sys.executable, str(RUNNER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": f"failed to spawn subprocess: {exc}",
            "error_type": "SpawnError",
            "traceback": None,
            "stdout": "",
            "plots": [],
            "elapsed_sec": 0.0,
            "killed_by_timeout": False,
        }

    try:
        stdout_text, _ = proc.communicate(input=source, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            stdout_text, _ = proc.communicate(timeout=2.0)
        except Exception:
            stdout_text = ""
        killed = True

    elapsed = time.time() - start

    if killed:
        return {
            "ok": False,
            "error": f"Code execution exceeded {timeout_sec:.1f}s timeout and was killed",
            "error_type": "Timeout",
            "traceback": None,
            "stdout": stdout_text or "",
            "plots": [],
            "elapsed_sec": round(elapsed, 3),
            "killed_by_timeout": True,
        }

    # Parse the JSON envelope from the tail of stdout
    marker = "__SANDBOX_RESULT__"
    idx = stdout_text.rfind(marker)
    if idx < 0:
        # Sandbox died before writing its envelope. Return what we got.
        return {
            "ok": False,
            "error": "sandbox produced no result envelope (child likely crashed)",
            "error_type": "SandboxCrash",
            "traceback": None,
            "stdout": stdout_text,
            "plots": [],
            "elapsed_sec": round(elapsed, 3),
            "killed_by_timeout": False,
        }

    json_part = stdout_text[idx + len(marker):].strip()
    try:
        envelope = json.loads(json_part)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"could not parse sandbox JSON: {exc}",
            "error_type": "EnvelopeError",
            "traceback": None,
            "stdout": stdout_text,
            "plots": [],
            "elapsed_sec": round(elapsed, 3),
            "killed_by_timeout": False,
        }

    envelope["killed_by_timeout"] = False
    return envelope


# Convenience: quick CLI for hand-testing without going through Streamlit
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        result = run_code(
            "import numpy as np\n"
            "print('hello from sandbox')\n"
            "print('mean:', np.mean([1, 2, 3, 4]))\n"
        )
        print(json.dumps(result, indent=2)[:600])
