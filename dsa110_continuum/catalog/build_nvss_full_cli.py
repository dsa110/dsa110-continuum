#!/usr/bin/env python
"""
Compatibility wrapper for NVSS full catalog build.

Use the unified entry point instead:
  python -m dsa110_contimg.core.catalog.build_full_catalogs_cli nvss ...
"""

from __future__ import annotations

from dsa110_contimg.core.catalog.build_full_catalogs_cli import main as unified_main


def main(argv: list[str] | None = None) -> int:
    argv = [] if argv is None else list(argv)
    if "--check" in argv:
        argv = [arg for arg in argv if arg != "--check"]
        argv.insert(0, "--status")
    return unified_main(["nvss", *argv])


if __name__ == "__main__":
    import sys

    sys.exit(main())
