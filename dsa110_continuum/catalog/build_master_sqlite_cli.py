#!/usr/bin/env python
"""
Compatibility wrapper for master catalog build (SQLite sources).

Use the unified entry point instead:
  python -m dsa110_contimg.core.catalog.build_master_cli sqlite ...
"""

from __future__ import annotations

from dsa110_contimg.core.catalog.build_master_cli import main as unified_main


def main(argv: list[str] | None = None) -> int:
    argv = [] if argv is None else list(argv)
    return unified_main(["sqlite", *argv])


if __name__ == "__main__":
    import sys

    sys.exit(main())
