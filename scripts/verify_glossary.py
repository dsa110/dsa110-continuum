#!/opt/miniforge/envs/casa6/bin/python
"""Verify file::symbol citations in CONTEXT.md against the codebase.

Parses backtick-quoted citations of the form ``path/to/file.py::Symbol`` (and
dotted forms like ``path/to/file.py::Class.attribute``) and checks that the
referenced symbol still resolves. For non-Python files (``.md``, ``.sh``,
``.lua``, etc.) only the file's existence is verified.

Usage::

    /opt/miniforge/envs/casa6/bin/python scripts/verify_glossary.py [GLOSSARY ...]

Default glossary: ``CONTEXT.md`` at the repo root. Pass additional paths to
verify other files (e.g. ``docs/agents/domain.md``).

Exit codes::

    0 — all citations resolved
    1 — at least one citation is stale or unresolved
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# ``path/to/file.ext::Dotted.Symbol`` inside backticks.
CITATION_RE = re.compile(
    r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)::([A-Za-z_][A-Za-z0-9_.]*)`"
)


def repo_root() -> Path:
    """Return the repo root (assumes script lives at <root>/scripts/)."""
    return Path(__file__).resolve().parent.parent


def parse_citations(text: str) -> list[tuple[str, str]]:
    """Return list of unique (file_path, symbol) tuples from text."""
    seen: set[tuple[str, str]] = set()
    for m in CITATION_RE.finditer(text):
        seen.add((m.group(1), m.group(2)))
    return sorted(seen)


def resolve_python_symbol(file_path: Path, symbol: str) -> tuple[bool, str]:
    """Check that *symbol* resolves inside the Python *file_path*.

    Walks module-level definitions and into classes for dotted symbols. Returns
    ``(ok, reason)``.
    """
    try:
        tree = ast.parse(file_path.read_text())
    except (SyntaxError, OSError) as exc:
        return False, f"parse error: {exc}"

    parts = symbol.split(".")
    head, rest = parts[0], parts[1:]

    head_node = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == head:
            head_node = node
            break
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == head:
                    head_node = node
                    break
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == head:
            head_node = node
            break
        if head_node is not None:
            break

    if head_node is None:
        return False, f"top-level symbol {head!r} not found"

    if not rest:
        return True, "ok"

    if not isinstance(head_node, ast.ClassDef):
        return False, f"{head!r} is not a class — cannot resolve attribute {'.'.join(rest)!r}"

    cls = head_node
    for attr in rest:
        found = False
        for body_node in cls.body:
            if isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and body_node.name == attr:
                found = True
                break
            if isinstance(body_node, ast.ClassDef) and body_node.name == attr:
                cls = body_node
                found = True
                break
            if isinstance(body_node, ast.Assign):
                for tgt in body_node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == attr:
                        found = True
                        break
            if isinstance(body_node, ast.AnnAssign) and isinstance(body_node.target, ast.Name) and body_node.target.id == attr:
                found = True
                break
            if found:
                break
        if not found:
            return False, f"attribute {attr!r} not found on class {cls.name!r}"

    return True, "ok"


def verify_citation(root: Path, file_path_str: str, symbol: str) -> tuple[bool, str]:
    """Verify a single (file, symbol) citation. Returns (ok, reason)."""
    file_path = root / file_path_str
    if not file_path.exists():
        return False, f"file not found: {file_path_str}"
    if file_path.suffix == ".py":
        return resolve_python_symbol(file_path, symbol)
    # Non-Python: file existence only, with a soft grep for the symbol literal.
    text = file_path.read_text(errors="replace")
    if symbol in text:
        return True, "ok (literal match)"
    return True, "ok (file exists; symbol not literally present — non-Python files are not strictly verified)"


def main(argv: list[str]) -> int:
    root = repo_root()
    targets = [Path(p) for p in argv[1:]] or [root / "CONTEXT.md"]

    overall_ok = True
    for target in targets:
        if not target.exists():
            print(f"✗ {target}: file not found")
            overall_ok = False
            continue
        text = target.read_text()
        citations = parse_citations(text)
        if not citations:
            print(f"⚠ {target}: no citations found")
            continue

        print(f"\n{target} — {len(citations)} citation(s)")
        n_ok = n_fail = 0
        for file_path_str, symbol in citations:
            ok, reason = verify_citation(root, file_path_str, symbol)
            mark = "✓" if ok else "✗"
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                overall_ok = False
            print(f"  {mark} {file_path_str}::{symbol}  —  {reason}")
        print(f"  → {n_ok} resolved, {n_fail} stale")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
