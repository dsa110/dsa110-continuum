#!/usr/bin/env python3
"""
Build _site/ for GitHub Pages without scanning /data/incoming (CI, or offline preview).

Copies the static UI from dsacamera_monitor/site/ and writes a valid empty
manifest so the page loads. Real inventory data only appears after you run
dsacamera-incoming-scan on dsacamera and either deploy that output elsewhere
or commit an updated public/manifest (see README).
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    """Copy static site to a build directory for manual Pages upload."""
    root = Path(__file__).resolve().parent.parent
    site_src = root / "dsacamera_monitor" / "site"
    if not site_src.is_dir():
        print(f"error: missing {site_src}", file=sys.stderr)
        return 1

    out = root / "_site"
    if out.is_dir():
        shutil.rmtree(out)
    shutil.copytree(site_src, out)

    # Lazy import so `python scripts/build_...` works before editable install
    sys.path.insert(0, str(root))
    from dsacamera_monitor.manifest import ScanAccum, build_manifest  # noqa: E402

    accum = ScanAccum()
    manifest = build_manifest(
        source_root="(CI placeholder — not scanned; run on dsacamera to publish real data)",
        accum=accum,
        no_stat=True,
        generated_at=datetime.now(timezone.utc),
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out}/ ({len(manifest.get('by_day', []))} days in manifest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
