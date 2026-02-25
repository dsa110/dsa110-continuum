#!/usr/bin/env python
"""
Compatibility wrapper for RAX strip catalog build.

Use the unified entry point instead:
  python -m dsa110_contimg.core.catalog.build_strip_cli rax ...
"""

from dsa110_contimg.core.catalog.build_strip_cli import main as unified_main


def main(argv: list[str] | None = None) -> int:
    argv = [] if argv is None else list(argv)
    return unified_main(["rax", *argv])


if __name__ == "__main__":
    import sys

    sys.exit(main())
