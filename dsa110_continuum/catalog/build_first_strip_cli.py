#!/usr/bin/env python
"""
Compatibility wrapper for FIRST strip catalog build.

Use the unified entry point instead:
  python -m dsa110_contimg.core.catalog.build_strip_cli first ...
"""

from dsa110_contimg.core.catalog.build_strip_cli import main as unified_main


def main(argv: list[str] | None = None) -> int:
    argv = [] if argv is None else list(argv)
    return unified_main(["first", *argv])


if __name__ == "__main__":
    import sys

    sys.exit(main())
