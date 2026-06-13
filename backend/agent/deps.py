"""
Dynamic sandbox dependencies — the domain-independent fix.

Parse the imports of AI-generated code and map them to pip packages, so the code agent can run
code from ANY domain without hardcoding domain libraries. The mapping is a small GENERIC
import-name->package alias table plus same-name passthrough; the packages actually installed are
whatever the generated code imports. code_runner builds (and caches by hash) a sandbox image with
exactly those packages on top of the base scientific image.
"""
from __future__ import annotations

import ast
import hashlib
import sys
from typing import Iterable, List, Set

# Generic import-name -> pip-package aliases (only where the two differ). Not domain-specific.
_ALIASES = {
    "cv2": "opencv-python-headless",
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "bs4": "beautifulsoup4",
    "yaml": "PyYAML",
    "skimage": "scikit-image",
    "Crypto": "pycryptodome",
    "OpenSSL": "pyOpenSSL",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "serial": "pyserial",
    "usb": "pyusb",
    "fitz": "PyMuPDF",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "Bio": "biopython",
    "attr": "attrs",
}

# Already baked into backend/agent/sandbox.Dockerfile — never re-install these (import names
# + their package names).
_BASE = {
    "numpy", "scipy", "pandas", "matplotlib", "sklearn", "scikit-learn",
    "sympy", "soundfile",
}
_BASE_LOWER = {b.lower() for b in _BASE}


def parse_imports(code: str) -> Set[str]:
    """Top-level module names imported by `code` (best-effort; never raises)."""
    out: Set[str] = set()
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:        # skip relative imports
                out.add(node.module.split(".")[0])
    return out


def _is_stdlib(mod: str) -> bool:
    return mod in getattr(sys, "stdlib_module_names", frozenset())


def modules_to_packages(modules: Iterable[str]) -> List[str]:
    """Map import names -> pip packages: drop stdlib + base-image packages, apply the alias table,
    passthrough same-name, dedupe, sort. Returns the EXTRA packages to install in the sandbox."""
    pkgs: Set[str] = set()
    for mod in modules:
        if not mod or _is_stdlib(mod) or mod in _BASE:
            continue
        pkg = _ALIASES.get(mod, mod)
        if pkg.lower() in _BASE_LOWER:
            continue
        pkgs.add(pkg)
    return sorted(pkgs)


def requirements_for(code: str) -> List[str]:
    """Extra pip packages the generated code needs (parse imports + map)."""
    return modules_to_packages(parse_imports(code))


def requirements_hash(packages: Iterable[str]) -> str:
    """Stable short hash of a package set — names the cached sandbox image (agent-sandbox:<hash>)."""
    joined = "\n".join(sorted(set(packages)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
