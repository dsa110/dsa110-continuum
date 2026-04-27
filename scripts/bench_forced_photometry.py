#!/opt/miniforge/envs/casa6/bin/python
"""Lightweight benchmark for parallel two-stage forced photometry (Batch D).

Generates a synthetic mosaic with N injected sources, runs the two-stage
photometry path serially and with multiple worker counts, and writes a JSON
summary plus a per-run CSV of timings to::

    /data/dsa110-continuum/outputs/photometry-bench/

The benchmark is intentionally small (default 1000 sources on a 1024×1024
image) so it runs in ~1 minute and is reproducible. It is a directional
speedup check, not a science-quality measurement.

Usage:
    /opt/miniforge/envs/casa6/bin/python scripts/bench_forced_photometry.py
    /opt/miniforge/envs/casa6/bin/python scripts/bench_forced_photometry.py --n-sources 2000 --workers 1 2 4 8
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# Project on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsa110_continuum.photometry.two_stage import run_two_stage_parallel  # noqa: E402

OUT_DIR = Path("/data/dsa110-continuum/outputs/photometry-bench")


def make_synthetic_mosaic(
    n_sources: int = 1000,
    image_size: int = 1024,
    seed: int = 17,
) -> tuple[Path, list[tuple[float, float]]]:
    """Create a 1024×1024 synthetic FITS with `n_sources` injected point sources."""
    rng = np.random.default_rng(seed)
    ny = nx = image_size
    data = rng.normal(0, 0.001, (ny, nx)).astype(np.float64)

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.crval = [180.0, 30.0]
    w.wcs.crpix = [nx // 2, ny // 2]
    w.wcs.cdelt = [-3.0 / 3600, 3.0 / 3600]  # 3 arcsec/pix to mimic DSA-110 mosaics

    coords: list[tuple[float, float]] = []
    for _ in range(n_sources):
        xpix = float(rng.uniform(nx * 0.05, nx * 0.95))
        ypix = float(rng.uniform(ny * 0.05, ny * 0.95))
        sky = w.pixel_to_world(xpix, ypix)
        coords.append((float(sky.ra.deg), float(sky.dec.deg)))
        ix, iy = int(round(xpix)), int(round(ypix))
        # Mix bright + faint so coarse-pass survivor count is realistic
        flux_jy = 0.05 if rng.random() < 0.3 else 0.005
        data[iy, ix] += flux_jy

    hdr = w.to_header()
    hdr["BMAJ"] = 36.0 / 3600
    hdr["BMIN"] = 25.0 / 3600
    hdr["BPA"] = 0.0

    tmpdir = Path(tempfile.mkdtemp(prefix="bench_phot_"))
    fpath = tmpdir / "synthetic.fits"
    fits.PrimaryHDU(data=data, header=hdr).writeto(fpath, overwrite=True)
    return fpath, coords


def time_run(fits_path: str, coords: list, workers: int) -> dict:
    """Time one run; return summary dict."""
    t0 = time.perf_counter()
    results, augments = run_two_stage_parallel(
        fits_path, coords, workers=workers, snr_coarse_min=3.0,
    )
    elapsed = time.perf_counter() - t0
    n_finite = sum(1 for r in results if np.isfinite(r.peak_jyb))
    n_passed_coarse = sum(1 for a in augments if a.passed_coarse)
    return {
        "workers": workers,
        "n_sources": len(coords),
        "n_passed_coarse": n_passed_coarse,
        "n_finite_results": n_finite,
        "elapsed_sec": round(elapsed, 3),
        "throughput_sources_per_sec": round(len(coords) / max(elapsed, 1e-9), 1),
    }


def main() -> int:
    """CLI entry point — generate, time, and write benchmark JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sources", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument(
        "--workers", type=int, nargs="+", default=[1, 2, 4],
        help="Worker counts to benchmark (default: 1 2 4)",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    stamp = started_at.strftime("%Y-%m-%dT%H_%M_%SZ")

    print(
        f"[bench] generating {args.n_sources} sources on "
        f"{args.image_size}×{args.image_size} synthetic mosaic..."
    )
    fits_path, coords = make_synthetic_mosaic(
        n_sources=args.n_sources, image_size=args.image_size, seed=args.seed,
    )

    runs: list[dict] = []
    for w in args.workers:
        print(f"[bench] running workers={w} ...", end=" ", flush=True)
        info = time_run(str(fits_path), coords, workers=w)
        runs.append(info)
        print(
            f"{info['elapsed_sec']:.2f}s  "
            f"({info['throughput_sources_per_sec']:.0f} src/s)"
        )

    serial_t = next((r["elapsed_sec"] for r in runs if r["workers"] == 1), None)
    for r in runs:
        if serial_t is not None:
            r["speedup_vs_workers_1"] = (
                round(serial_t / max(r["elapsed_sec"], 1e-9), 2)
            )

    summary = {
        "started_at_utc": started_at.isoformat(),
        "n_sources": args.n_sources,
        "image_size": args.image_size,
        "seed": args.seed,
        "synthetic_fits": str(fits_path),
        "runs": runs,
    }

    json_path = OUT_DIR / f"bench_{stamp}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[bench] wrote {json_path}")

    print("\n=== SUMMARY ===")
    for r in runs:
        sp = r.get("speedup_vs_workers_1", "—")
        print(
            f"  workers={r['workers']:>2}  "
            f"elapsed={r['elapsed_sec']:>6.2f}s  "
            f"throughput={r['throughput_sources_per_sec']:>7.0f} src/s  "
            f"speedup={sp}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
