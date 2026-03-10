#!/opt/miniforge/envs/casa6/bin/python
"""Import-migration verifier for dsa110_contimg → dsa110_continuum.

Scans dsa110_continuum/ for stale references to the old dsa110_contimg package
and optionally checks whether the affected modules can still be imported.

Usage:
    # Report stale imports (read-only)
    python scripts/check_import_migration.py

    # Also attempt imports of every affected module
    python scripts/check_import_migration.py --check-imports

    # Scan a different directory
    python scripts/check_import_migration.py --root /some/other/pkg

Exit codes:
    0 – no stale imports found
    1 – stale imports remain (or import failures when --check-imports used)
"""
import argparse
import subprocess
import sys
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
SCAN_DIR = REPO_ROOT / "dsa110_continuum"
STALE_PATTERN = "dsa110_contimg"  # old package name
IMPORT_PREFIXES = ("from dsa110_contimg", "import dsa110_contimg")
CASA6_PYTHON = "/opt/miniforge/envs/casa6/bin/python"


# ── Scanning ──────────────────────────────────────────────────────────────────

def scan_stale_imports(root: Path) -> dict[Path, list[tuple[int, str]]]:
    """Return {file: [(lineno, line), ...]} for files with stale imports."""
    hits: dict[Path, list[tuple[int, str]]] = {}
    for py_file in sorted(root.rglob("*.py")):
        file_hits: list[tuple[int, str]] = []
        try:
            lines = py_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, 1):
            stripped = line.strip()
            if any(stripped.startswith(p) for p in IMPORT_PREFIXES):
                file_hits.append((lineno, line.rstrip()))
        if file_hits:
            hits[py_file] = file_hits
    return hits


def file_to_module(py_file: Path, root: Path) -> str | None:
    """Convert a file path inside root to a dotted module name."""
    try:
        rel = py_file.relative_to(root.parent)
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


# ── Import check ──────────────────────────────────────────────────────────────

def try_import_module(module_name: str) -> tuple[bool, str]:
    """Try to import a module in a subprocess (isolates import-time crashes)."""
    cmd = [
        CASA6_PYTHON,
        "-c",
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); import {module_name}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if result.returncode == 0:
        return True, ""
    stderr = result.stderr.strip().splitlines()
    error_line = stderr[-1] if stderr else "(no output)"
    return False, error_line


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(hits: dict[Path, list[tuple[int, str]]], root: Path, check_imports: bool) -> int:
    """Print the migration status report. Returns exit code."""
    total_lines = sum(len(v) for v in hits.values())
    n_files = len(hits)

    print(f"\n{'=' * 72}")
    print("  DSA-110 Import Migration Check")
    print(f"  Scanning: {root}")
    print(f"  Pattern:  '{STALE_PATTERN}'")
    print(f"{'=' * 72}")

    if total_lines == 0:
        print(f"\n  [OK] No stale '{STALE_PATTERN}' imports found.")
        print(f"{'=' * 72}\n")
        return 0

    print(f"\n  Found {total_lines} stale import line(s) in {n_files} file(s):\n")

    import_results: dict[str, tuple[bool, str]] = {}

    for py_file in sorted(hits):
        rel_path = py_file.relative_to(REPO_ROOT)
        print(f"  {rel_path}")
        for lineno, line in hits[py_file]:
            print(f"    {lineno:>5}: {line}")

        if check_imports:
            module = file_to_module(py_file, root)
            if module and module not in import_results:
                ok, err = try_import_module(module)
                import_results[module] = (ok, err)

        print()

    if check_imports:
        print(f"  {'─' * 68}")
        print(f"  Import check results (using {CASA6_PYTHON}):\n")
        n_ok = sum(1 for ok, _ in import_results.values() if ok)
        n_fail = len(import_results) - n_ok
        for module, (ok, err) in sorted(import_results.items()):
            icon = "OK  " if ok else "FAIL"
            line = f"  [{icon}] {module}"
            if not ok:
                line += f"\n         {err}"
            print(line)
        print()
        print(f"  Import results: {n_ok} pass, {n_fail} fail out of {len(import_results)} modules")
        print(f"{'=' * 72}\n")
        return 1 if (n_fail > 0 or total_lines > 0) else 0

    print(f"  {'─' * 68}")
    print(f"  Summary: {total_lines} stale import line(s) in {n_files} file(s) remain.")
    print("  Run with --check-imports to also verify module importability.")
    print(f"{'=' * 72}\n")
    return 1


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Report remaining dsa110_contimg imports in dsa110_continuum/",
        epilog="Exit 0 = clean, Exit 1 = stale imports remain.",
    )
    parser.add_argument(
        "--root",
        default=str(SCAN_DIR),
        help="Directory to scan (default: dsa110_continuum/)",
    )
    parser.add_argument(
        "--check-imports",
        action="store_true",
        help="Also attempt to import each affected module and report pass/fail",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: directory not found: {root}", file=sys.stderr)
        sys.exit(2)

    hits = scan_stale_imports(root)
    rc = print_report(hits, root, args.check_imports)
    sys.exit(rc)


if __name__ == "__main__":
    main()
