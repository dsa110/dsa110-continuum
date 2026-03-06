#!/usr/bin/env python
"""
Quick diagnostic PDF for mosaics and tiles.

Usage:
    python scripts/inspect_fits.py                        # default: all known epochs
    python scripts/inspect_fits.py path/to/file.fits ...  # specific files
    python scripts/inspect_fits.py --out /tmp/out.pdf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, Normalize
from astropy.io import fits
from astropy.wcs import WCS

STAGE = Path("/stage/dsa110-contimg/images")
PRODUCTS = Path("/data/dsa110-continuum/products/mosaics")

DEFAULT_FILES = [
    # label, path
    ("Jan25 T02h mosaic [GOOD]",   PRODUCTS / "2026-01-25/2026-01-25T0200_mosaic.fits"),
    ("Jan25 T22h mosaic",          PRODUCTS / "2026-01-25/2026-01-25T2200_mosaic.fits"),
    ("Jan25 T22h tile-0 raw",      STAGE / "mosaic_2026-01-25/2026-01-25T22:00:18-image.fits"),
    ("Jan25 T22h tile-0 pbcor",    STAGE / "mosaic_2026-01-25/2026-01-25T22:00:18-image-pb.fits"),
    ("Feb15 T00h mosaic",          STAGE / "mosaic_2026-02-15/2026-02-15T0000_mosaic.fits"),
    ("Feb15 T00h tile-0 raw",      STAGE / "mosaic_2026-02-15/2026-02-15T00:38:57-image.fits"),
    ("Feb15 T00h tile-0 pbcor",    STAGE / "mosaic_2026-02-15/2026-02-15T00:38:57-image-pb.fits"),
    ("Feb23 T00h mosaic [Dec=33?]", STAGE / "mosaic_2026-02-23/2026-02-23T0000_mosaic.fits"),
    ("Feb23 T00h tile-0 raw",      STAGE / "mosaic_2026-02-23/2026-02-23T00:05:44-image.fits"),
    ("Feb23 T00h tile-0 pbcor",    STAGE / "mosaic_2026-02-23/2026-02-23T00:05:44-image-pb.fits"),
    ("Feb26 T00h mosaic",          STAGE / "mosaic_2026-02-26/2026-02-26T0000_mosaic.fits"),
    ("Feb26 T00h tile-0 raw",      STAGE / "mosaic_2026-02-26/2026-02-26T00:04:49-image.fits"),
    ("Feb26 T00h tile-0 pbcor",    STAGE / "mosaic_2026-02-26/2026-02-26T00:04:49-image-pb.fits"),
]


def load(path: Path):
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data.squeeze()
    try:
        wcs = WCS(hdr, naxis=2)
    except Exception:
        wcs = None
    return data, hdr, wcs


def vmin_vmax(data, pct_lo=1, pct_hi=99.5):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0, 1
    lo = np.percentile(finite, pct_lo)
    hi = np.percentile(finite, pct_hi)
    return lo, hi


def render_page(pdf, label, path):
    try:
        data, hdr, wcs = load(path)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, f"Could not load:\n{path}\n{e}",
                ha="center", va="center", transform=ax.transAxes, color="red")
        ax.set_title(label)
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    finite = data[np.isfinite(data)]
    peak   = float(np.nanmax(data)) if finite.size else float("nan")
    rms    = float(np.nanstd(finite)) if finite.size else float("nan")
    n_nan  = int(np.sum(~np.isfinite(data)))
    ra0    = hdr.get("CRVAL1", float("nan"))
    dec0   = hdr.get("CRVAL2", float("nan"))

    fig = plt.figure(figsize=(12, 8))

    subplot_kw = {"projection": wcs} if wcs is not None else {}
    ax = fig.add_subplot(1, 1, 1, **subplot_kw)

    lo, hi = vmin_vmax(data)
    if hi <= lo:
        hi = lo + 1e-6
    # clip to [lo, hi] and use linear scale; avoid log with negatives
    display = np.clip(data, lo, hi)
    im = ax.imshow(display, origin="lower", cmap="inferno",
                   norm=Normalize(vmin=lo, vmax=hi),
                   interpolation="nearest", aspect="auto")
    plt.colorbar(im, ax=ax, label="Jy/beam", fraction=0.03, pad=0.04)

    if wcs is not None:
        ax.set_xlabel("RA (J2000)")
        ax.set_ylabel("Dec (J2000)")
        try:
            ax.coords[0].set_format_unit("deg")
            ax.coords[1].set_format_unit("deg")
        except Exception:
            pass
    else:
        ax.set_xlabel("pixel x")
        ax.set_ylabel("pixel y")

    ax.set_title(
        f"{label}\n"
        f"peak={peak:.4f} Jy/bm   rms={rms*1e3:.1f} mJy   "
        f"NaN={n_nan}/{data.size}   "
        f"RA={ra0:.2f}°   Dec={dec0:.2f}°",
        fontsize=9,
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"  rendered: {label}  peak={peak:.4f}  rms={rms*1e3:.1f}mJy  RA={ra0:.1f} Dec={dec0:.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="FITS files to render (optional; uses defaults if omitted)")
    parser.add_argument("--out", default="/tmp/dsa110_diagnostic.pdf", help="Output PDF path")
    args = parser.parse_args()

    if args.files:
        entries = [(Path(f).stem, Path(f)) for f in args.files]
    else:
        entries = [(label, Path(path)) for label, path in DEFAULT_FILES]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing {len(entries)} pages to {out}")
    with PdfPages(out) as pdf:
        for label, path in entries:
            render_page(pdf, label, path)

    print(f"\nDone → {out}")


if __name__ == "__main__":
    main()
