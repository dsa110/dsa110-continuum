#!/usr/bin/env python
from __future__ import annotations

# Version guard - prevent use of Python 2.7 or 3.6
import sys

if sys.version_info < (3, 11) or sys.version_info[:2] in [(2, 7), (3, 6)]:
    sys.stderr.write(f"ERROR: Python 3.11+ required. Found: {sys.version}\n")
    sys.stderr.write("Activate casa6: conda activate casa6 (see backend/docs/reference/CASA_REFERENCE.md)\n")
    sys.exit(1)
"""
CLI for forced photometry on FITS images.

Examples:
  # Single coordinate
  python -m dsa110_contimg.core.photometry.cli peak \
    --fits /path/to/image.pbcor.fits \
    --ra 128.725 --dec 55.573 \
    --box 5 --annulus 12 20

  # Multiple coordinates
  python -m dsa110_contimg.core.photometry.cli peak-many \
    --fits /path/to/image.pbcor.fits \
    --coords "128.725,55.573; 129.002,55.610"
"""

import argparse
import json
import os
import time
from pathlib import Path

import astropy.coordinates as acoords  # type: ignore[reportMissingTypeStubs]
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits  # type: ignore[reportMissingTypeStubs]
from astropy.wcs import WCS  # type: ignore[reportMissingTypeStubs]
from matplotlib.colors import Normalize

from dsa110_contimg.core.catalog.query import query_sources
from dsa110_contimg.infrastructure.database import (
    ensure_pipeline_db,
    photometry_insert,
)
from dsa110_contimg.core.photometry.ese_detection import detect_ese_candidates

from .adaptive_binning import AdaptiveBinningConfig
from .adaptive_photometry import measure_with_adaptive_binning
from .aegean_fitting import measure_with_aegean
from .forced import measure_forced_peak, measure_many


def _get_pipeline_db_path(cli_arg: str | None = None) -> Path:
    """Get pipeline database path from centralized settings or CLI argument.

    Priority order:
    1. CLI argument (if provided and not default)
    2. Centralized settings (DatabaseSettings.path)
    3. Default path state/db/pipeline.sqlite3
    """
    # Default values
    default_path = "state/db/pipeline.sqlite3"  # CLI default (unified DB)
    unified_default = os.environ.get(
        "PIPELINE_DB", "/data/dsa110-contimg/state/db/pipeline.sqlite3"
    )

    # If CLI arg provided and not the default, use it
    if cli_arg and cli_arg != default_path:
        return Path(cli_arg)

    # Try centralized settings
    try:
        from dsa110_contimg.common.unified_config import get_settings

        settings = get_settings()
        if hasattr(settings, "database") and hasattr(settings.database, "path"):
            return settings.database.path
    except (ImportError, Exception):
        pass

    # Fail fast on legacy env vars to force migration
    legacy_envs = [env for env in ("PIPELINE_PRODUCTS_DB", "CONTIMG_PRODUCTS_DB") if os.getenv(env)]
    if legacy_envs:
        raise RuntimeError(
            f"Legacy database env vars are no longer supported ({', '.join(legacy_envs)}). "
            "Set PIPELINE_DB instead."
        )

    # Fall back to unified env var or default
    return Path(os.getenv("PIPELINE_DB", unified_default))


def _parse_coords_arg(coords_arg: str) -> list[tuple[float, float]]:
    parts = [c.strip() for c in coords_arg.split(";") if c.strip()]
    coords: list[tuple[float, float]] = []
    for p in parts:
        ra_s, dec_s = [s.strip() for s in p.split(",")]
        coords.append((float(ra_s), float(dec_s)))
    return coords


def cmd_peak(args: argparse.Namespace) -> int:
    if args.use_aegean:
        # Use Aegean forced fitting
        res_aegean = measure_with_aegean(
            args.fits,
            args.ra,
            args.dec,
            use_prioritized=args.aegean_prioritized,
            negative=args.aegean_negative,
        )
        # Convert to compatible format
        result_dict = {
            "ra_deg": res_aegean.ra_deg,
            "dec_deg": res_aegean.dec_deg,
            "peak_jyb": res_aegean.peak_flux_jy,
            "peak_err_jyb": res_aegean.err_peak_flux_jy,
            "local_rms_jy": res_aegean.local_rms_jy,
            "integrated_flux_jy": res_aegean.integrated_flux_jy,
            "err_integrated_flux_jy": res_aegean.err_integrated_flux_jy,
            "success": res_aegean.success,
            "error_message": res_aegean.error_message,
            "method": "aegean",
        }
        print(json.dumps(result_dict, indent=2))
        return 0 if res_aegean.success else 1
    else:
        # Use simple peak measurement
        res = measure_forced_peak(
            args.fits,
            args.ra,
            args.dec,
            box_size_pix=args.box,
            annulus_pix=(args.annulus[0], args.annulus[1]),
        )
        result_dict = res.__dict__
        result_dict["method"] = "peak"
        print(json.dumps(result_dict, indent=2))
        return 0


def cmd_peak_many(args: argparse.Namespace) -> int:
    coords = _parse_coords_arg(args.coords)
    results = measure_many(args.fits, coords, box_size_pix=args.box)
    print(json.dumps([r.__dict__ for r in results], indent=2))
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """Run batch photometry from a CSV source list across multiple images."""
    import csv
    import glob

    # Parse source list
    sources = []
    with open(args.source_list) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", row.get("source_id", "")).strip()
            ra = float(row.get("ra", row.get("ra_deg", 0)))
            dec = float(row.get("dec", row.get("dec_deg", 0)))
            sources.append({"name": name, "ra": ra, "dec": dec})

    if not sources:
        print(json.dumps({"error": "No sources found in CSV file"}, indent=2))
        return 1

    # Get image list from either --image-dir or --image-list
    image_paths: list[str] = []
    if args.image_dir:
        # Recursively search for FITS files
        image_paths = sorted(
            glob.glob(os.path.join(args.image_dir, "**", "*.fits"), recursive=True)
        )
    elif args.image_list:
        # Read paths from file
        with open(args.image_list) as f:
            image_paths = [line.strip() for line in f if line.strip()]
    else:
        print(json.dumps({"error": "Either --image-dir or --image-list is required"}, indent=2))
        return 1

    if not image_paths:
        print(json.dumps({"error": "No images found"}, indent=2))
        return 1

    # Setup database connection - always store results
    pdb_path = _get_pipeline_db_path(getattr(args, "products_db", None))
    conn = ensure_pipeline_db()

    results = []
    now = time.time()
    total_measured = 0
    total_skipped = 0

    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue

        for src in sources:
            try:
                m = measure_forced_peak(
                    img_path,
                    float(src["ra"]),
                    float(src["dec"]),
                    box_size_pix=args.box,
                    annulus_pix=(args.annulus[0], args.annulus[1]),
                )

                if not np.isfinite(m.peak_jyb):
                    total_skipped += 1
                    continue

                result_entry = {
                    "source_name": src["name"],
                    "image_path": img_path,
                    "ra_deg": m.ra_deg,
                    "dec_deg": m.dec_deg,
                    "peak_jyb": m.peak_jyb,
                    "peak_err_jyb": m.peak_err_jyb,
                }
                results.append(result_entry)
                total_measured += 1

                # Store in database if requested
                if conn is not None:
                    perr = (
                        None
                        if (m.peak_err_jyb is None or not np.isfinite(m.peak_err_jyb))
                        else float(m.peak_err_jyb)
                    )
                    src_id = (
                        str(src["name"])
                        if src.get("name")
                        else f"src_{m.ra_deg:.4f}_{m.dec_deg:.4f}"
                    )
                    photometry_insert(
                        conn,
                        image_path=img_path,
                        source_id=src_id,
                        ra_deg=m.ra_deg,
                        dec_deg=m.dec_deg,
                        flux_jy=m.peak_jyb,
                        nvss_flux_mjy=None,
                        peak_jyb=m.peak_jyb,
                        peak_err_jyb=perr,
                        measured_at=now,
                    )
            except Exception as e:
                total_skipped += 1
                if args.verbose:
                    print(
                        f"Warning: Failed to measure {src['name']} in {img_path}: {e}",
                        file=sys.stderr,
                    )

    if conn is not None:
        conn.commit()
        conn.close()

    # Write output CSV if requested
    if args.output:
        import csv as csv_module

        with open(args.output, "w", newline="") as f:
            if results:
                writer = csv_module.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        print(f"Wrote {len(results)} measurements to {args.output}")

    # Print summary
    summary: dict[str, object] = {
        "source_list": args.source_list,
        "image_dir": args.image_dir,
        "image_list": args.image_list,
        "n_sources": len(sources),
        "n_images": len(image_paths),
        "total_measured": total_measured,
        "total_skipped": total_skipped,
        "products_db": pdb_path,
    }
    if not args.output:
        summary["results"] = results

    print(json.dumps(summary, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export lightcurve data for a source from the database."""
    import csv as csv_module

    conn = ensure_pipeline_db()

    # Query photometry for the source
    rows = conn.execute(
        """
        SELECT source_id, image_path, ra_deg, dec_deg, peak_jyb, peak_err_jyb, measured_at
        FROM photometry
        WHERE source_id = ?
        ORDER BY measured_at
        """,
        (args.source_id,),
    ).fetchall()
    conn.close()

    if not rows:
        print(json.dumps({"error": f"No photometry found for source: {args.source_id}"}, indent=2))
        return 1

    # Convert to list of dicts
    data = [
        {
            "source_id": r[0],
            "image_path": r[1],
            "ra_deg": r[2],
            "dec_deg": r[3],
            "flux_jy": r[4],
            "flux_err_jy": r[5],
            "measured_at": r[6],
        }
        for r in rows
    ]

    # Output based on format
    if args.format == "json":
        output = json.dumps(data, indent=2)
    elif args.format == "csv":
        import io

        buf = io.StringIO()
        writer = csv_module.DictWriter(buf, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        output = buf.getvalue()
    else:
        print(json.dumps({"error": f"Unknown format: {args.format}"}, indent=2))
        return 1

    # Write to file or stdout
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Exported {len(data)} measurements to {args.output}")
    else:
        print(output)

    return 0


def _image_center_and_radius_deg(fits_path: str) -> tuple[float, float, float]:
    hdr = fits.getheader(fits_path)
    w = WCS(hdr).celestial
    nx = hdr.get("NAXIS1", 0)
    ny = hdr.get("NAXIS2", 0)
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    c = w.pixel_to_world(cx, cy)
    # Corners
    corners = [
        w.pixel_to_world(0.0, 0.0),
        w.pixel_to_world(nx - 1.0, 0.0),
        w.pixel_to_world(0.0, ny - 1.0),
        w.pixel_to_world(nx - 1.0, ny - 1.0),
    ]
    center = acoords.SkyCoord(c.ra.deg, c.dec.deg, unit="deg", frame="icrs")
    rad = 0.0
    for s in corners:
        sep = center.separation(s).deg
        if sep > rad:
            rad = sep
    # Add small margin
    rad = float(rad * 1.02)
    return float(center.ra.deg), float(center.dec.deg), rad


def cmd_nvss(args: argparse.Namespace) -> int:
    ra0, dec0, auto_rad = _image_center_and_radius_deg(args.fits)
    radius_deg = float(args.radius_deg) if args.radius_deg is not None else auto_rad

    # Get catalog type from args (default: master)
    catalog_type = getattr(args, "catalog", "master")

    # Use SQLite-first query function (falls back to CSV if needed)
    df = query_sources(
        catalog_type=catalog_type,
        ra_deg=ra0,
        dec_deg=dec0,
        radius_deg=radius_deg,
        min_flux_mjy=float(args.min_mjy),
        catalog_path=args.catalog_path,
    )
    # Rename columns to match expected format
    df = df.rename(columns={"ra_deg": "ra", "dec_deg": "dec", "flux_mjy": "flux_20_cm"})
    ra_sel = df["ra"].to_numpy()
    dec_sel = df["dec"].to_numpy()
    flux_sel = df["flux_20_cm"].to_numpy()

    results = []
    now = time.time()
    conn = ensure_pipeline_db()
    try:
        inserted = 0
        skipped = 0
        for ra, dec, nvss in zip(ra_sel, dec_sel, flux_sel):
            m = measure_forced_peak(
                args.fits,
                float(ra),
                float(dec),
                box_size_pix=args.box,
                annulus_pix=(args.annulus[0], args.annulus[1]),
            )
            if not np.isfinite(m.peak_jyb):
                skipped += 1
                continue
            perr = (
                None
                if (m.peak_err_jyb is None or not np.isfinite(m.peak_err_jyb))
                else float(m.peak_err_jyb)
            )
            src_id = f"nvss_{m.ra_deg:.4f}_{m.dec_deg:.4f}"
            photometry_insert(
                conn,
                image_path=args.fits,
                source_id=src_id,
                ra_deg=m.ra_deg,
                dec_deg=m.dec_deg,
                flux_jy=m.peak_jyb,
                nvss_flux_mjy=float(nvss),
                peak_jyb=m.peak_jyb,
                peak_err_jyb=perr,
                measured_at=now,
            )
            results.append(m.__dict__)
            inserted += 1
        conn.commit()
    finally:
        conn.close()
    print(
        json.dumps(
            {
                "image": args.fits,
                "center_ra_deg": ra0,
                "center_dec_deg": dec0,
                "radius_deg": radius_deg,
                "min_mjy": float(args.min_mjy),
                "count": len(results),
                "inserted": inserted,
                "skipped": skipped,
                "results": results,
            },
            indent=2,
        )
    )
    return 0


def cmd_adaptive(args: argparse.Namespace) -> int:
    """Run adaptive binning photometry."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create adaptive binning config
    config = AdaptiveBinningConfig(
        target_snr=args.target_snr,
        max_width=args.max_width,
    )

    # Run adaptive binning photometry
    result = measure_with_adaptive_binning(
        ms_path=args.ms,
        ra_deg=args.ra,
        dec_deg=args.dec,
        output_dir=output_dir,
        config=config,
        max_spws=args.max_spws,
        imsize=args.imsize,
        quality_tier=args.quality_tier,
        backend=args.backend,
        parallel=args.parallel,
        max_workers=args.max_workers,
        serialize_ms_access=args.serialize_ms_access,
    )

    # Format output
    output_dict = {
        "ra_deg": result.ra_deg,
        "dec_deg": result.dec_deg,
        "n_spws": result.n_spws,
        "success": result.success,
        "error_message": result.error_message,
        "detections": [
            {
                "spw_ids": det.channels,
                "flux_jy": det.flux_jy,
                "rms_jy": det.rms_jy,
                "snr": det.snr,
                "center_freq_mhz": det.center_freq_mhz,
                "bin_width": det.bin_width,
            }
            for det in result.detections
        ],
        "spw_info": [
            {
                "spw_id": info.spw_id,
                "center_freq_mhz": info.center_freq_mhz,
                "bandwidth_mhz": info.bandwidth_mhz,
                "num_channels": info.num_channels,
            }
            for info in result.spw_info
        ],
    }

    print(json.dumps(output_dict, indent=2))
    return 0 if result.success else 1


def cmd_ese_detect(args: argparse.Namespace) -> int:
    """Detect ESE candidates from variability statistics."""
    from dsa110_contimg.core.photometry.thresholds import get_threshold_preset

    products_db = _get_pipeline_db_path(getattr(args, "products_db", None))

    if not products_db.exists():
        print(json.dumps({"error": f"Products database not found: {products_db}"}, indent=2))
        return 1

    # Handle preset or min_sigma
    preset = getattr(args, "preset", None)
    min_sigma_param = getattr(args, "min_sigma", None)

    if preset:
        thresholds = get_threshold_preset(preset)
        min_sigma = thresholds.get("min_sigma", 5.0)
    elif min_sigma_param is not None:
        min_sigma = min_sigma_param
    else:
        # Default fallback
        min_sigma = 5.0

    try:
        candidates = detect_ese_candidates(
            products_db=products_db,
            min_sigma=min_sigma,
            source_id=getattr(args, "source_id", None),
            recompute=getattr(args, "recompute", False),
            use_composite_scoring=getattr(args, "use_composite_scoring", False),
        )

        result = {
            "products_db": str(products_db),
            "preset": preset,
            "min_sigma": min_sigma,
            "source_id": getattr(args, "source_id", None),
            "recompute": getattr(args, "recompute", False),
            "candidates_found": len(candidates),
            "candidates": candidates,
        }

        print(json.dumps(result, indent=2))
        return 0

    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2))
        return 1


def cmd_plot(args: argparse.Namespace) -> int:
    # Load image
    hdr = fits.getheader(args.fits)
    data = np.asarray(fits.getdata(args.fits)).squeeze()
    w = WCS(hdr).celestial

    # Build mask for valid pixels
    m = np.isfinite(data)
    vals = data[m]
    lo, hi = (
        (np.nanpercentile(vals, 2.0), np.nanpercentile(vals, 98.0)) if vals.size else (0.0, 1.0)
    )
    img = np.clip(data, lo, hi)

    # Compute FoV outline directly in pixel space to avoid spherical :arrow_right: WCS distortions
    nx = hdr.get("NAXIS1", 0)
    ny = hdr.get("NAXIS2", 0)
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    # Use largest inscribed circle in the image bounds as an outline; shrink slightly
    r_pix = 0.98 * float(min(cx, cy, (nx - 1) - cx, (ny - 1) - cy))
    th = np.linspace(0, 2 * np.pi, 360)
    xcirc = cx + r_pix * np.cos(th)
    ycirc = cy + r_pix * np.sin(th)

    # Load photometry rows for this image
    conn = ensure_pipeline_db()
    rows = conn.execute(
        "SELECT ra_deg, dec_deg, peak_jyb, nvss_flux_mjy FROM photometry WHERE image_path = ?",
        (args.fits,),
    ).fetchall()
    conn.close()
    if not rows:
        print("No photometry rows for image; run nvss first")
        return 1
    ra = np.array([r[0] for r in rows], dtype=float)
    dec = np.array([r[1] for r in rows], dtype=float)
    peak = np.array([r[2] for r in rows], dtype=float)
    nvss_jy = np.array([np.nan if r[3] is None else (float(r[3]) / 1e3) for r in rows], dtype=float)
    coords = acoords.SkyCoord(ra, dec, unit="deg", frame="icrs")
    x, y = w.world_to_pixel(coords)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": w})
    # Left: forced photometry peak
    ax = axes[0]
    ax.imshow(img, origin="lower", cmap="gray")
    ax.plot(
        xcirc,
        ycirc,
        color="black",
        linewidth=1.0,
        alpha=0.8,
        transform=ax.get_transform("pixel"),
    )
    norm_p = Normalize(
        vmin=args.vmin if args.vmin is not None else np.nanmin(peak),
        vmax=args.vmax if args.vmax is not None else np.nanmax(peak),
    )
    sc0 = ax.scatter(
        x,
        y,
        c=peak,
        s=24,
        cmap=args.cmap,
        norm=norm_p,
        edgecolor="white",
        linewidths=0.3,
    )
    cb0 = fig.colorbar(sc0, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cb0.set_label("Peak [Jy/beam]")
    ax.set_title("Forced Photometry Peak")
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")

    # Right: NVSS catalog flux (Jy)
    ax = axes[1]
    ax.imshow(img, origin="lower", cmap="gray")
    ax.plot(
        xcirc,
        ycirc,
        color="black",
        linewidth=1.0,
        alpha=0.8,
        transform=ax.get_transform("pixel"),
    )
    norm_n = Normalize(vmin=np.nanmin(nvss_jy), vmax=np.nanmax(nvss_jy))
    sc1 = ax.scatter(
        x,
        y,
        c=nvss_jy,
        s=24,
        cmap=args.cmap,
        norm=norm_n,
        edgecolor="white",
        linewidths=0.3,
    )
    cb1 = fig.colorbar(sc1, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cb1.set_label("NVSS Flux [Jy]")
    ax.set_title("NVSS Catalog Flux")
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    out = args.out or (os.path.splitext(args.fits)[0] + "_photometry_compare.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Forced photometry utilities")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("peak", help="Measure peak at a single RA,Dec")
    sp.add_argument("--fits", required=True, help="Input FITS image (PB-corrected)")
    sp.add_argument("--ra", type=float, required=True, help="Right Ascension (deg)")
    sp.add_argument("--dec", type=float, required=True, help="Declination (deg)")
    sp.add_argument("--box", type=int, default=5, help="Box size in pixels")
    sp.add_argument(
        "--annulus",
        type=int,
        nargs=2,
        default=(12, 20),
        help="Annulus radii in pixels [rin rout]",
    )
    sp.add_argument(
        "--use-aegean",
        action="store_true",
        help="Use Aegean forced fitting instead of simple peak measurement",
    )
    sp.add_argument(
        "--aegean-prioritized",
        action="store_true",
        default=True,
        help="Use --priorized flag for Aegean (handles blended sources, default: True)",
    )
    sp.add_argument(
        "--no-aegean-prioritized",
        dest="aegean_prioritized",
        action="store_false",
        help="Disable --priorized flag for Aegean",
    )
    sp.add_argument(
        "--aegean-negative",
        action="store_true",
        help="Allow negative detections in Aegean",
    )
    sp.set_defaults(func=cmd_peak)

    sp = sub.add_parser("peak-many", help="Measure peaks for a list of RA,Dec pairs")
    sp.add_argument("--fits", required=True, help="Input FITS image (PB-corrected)")
    sp.add_argument(
        "--coords",
        required=True,
        help='Semicolon-separated RA,Dec pairs: "ra1,dec1; ra2,dec2"',
    )
    sp.add_argument("--box", type=int, default=5, help="Box size in pixels")
    sp.set_defaults(func=cmd_peak_many)

    sp = sub.add_parser("nvss", help="Forced photometry for catalog sources within the image FoV")
    sp.add_argument("--fits", required=True, help="Input FITS image (PB-corrected)")
    sp.add_argument(
        "--products-db",
        default="state/db/pipeline.sqlite3",
        help="Pipeline database (default: unified DB)",
    )
    sp.add_argument(
        "--catalog",
        type=str,
        choices=["nvss", "first", "rax", "vlass", "master"],
        default="master",
        help="Catalog to query (default: master - unified NVSS+FIRST+VLASS+RACS)",
    )
    sp.add_argument("--min-mjy", type=float, default=10.0, help="Minimum flux threshold (mJy)")
    sp.add_argument("--radius-deg", type=float, default=None, help="Override FoV radius (deg)")
    sp.add_argument(
        "--catalog-path",
        type=str,
        default=None,
        help="Explicit path to catalog SQLite database (overrides auto-resolution)",
    )
    sp.add_argument("--box", type=int, default=5, help="Box size in pixels")
    sp.add_argument(
        "--annulus",
        type=int,
        nargs=2,
        default=(12, 20),
        help="Annulus radii in pixels [rin rout]",
    )
    sp.set_defaults(func=cmd_nvss)

    sp = sub.add_parser("plot", help="Visualize photometry results overlaid on FITS image")
    sp.add_argument("--fits", required=True, help="Input FITS image (PB-corrected)")
    sp.add_argument(
        "--products-db",
        default="state/db/pipeline.sqlite3",
        help="Pipeline database (default: unified DB)",
    )
    sp.add_argument("--out", default=None, help="Output PNG path")
    sp.add_argument("--cmap", default="viridis")
    sp.add_argument("--vmin", type=float, default=None)
    sp.add_argument("--vmax", type=float, default=None)
    sp.set_defaults(func=cmd_plot)

    sp = sub.add_parser("adaptive", help="Adaptive channel binning photometry from Measurement Set")
    sp.add_argument("--ms", required=True, help="Input Measurement Set path")
    sp.add_argument("--ra", type=float, required=True, help="Right Ascension (deg)")
    sp.add_argument("--dec", type=float, required=True, help="Declination (deg)")
    sp.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for SPW images and results",
    )
    sp.add_argument(
        "--target-snr",
        type=float,
        default=5.0,
        help="Target SNR threshold for detections (default: 5.0)",
    )
    sp.add_argument(
        "--max-width",
        type=int,
        default=16,
        help="Maximum bin width in channels (default: 16 for DSA-110)",
    )
    sp.add_argument("--imsize", type=int, default=1024, help="Image size in pixels (default: 1024)")
    sp.add_argument(
        "--quality-tier",
        type=str,
        default="standard",
        choices=["development", "standard", "high_precision"],
        help="Imaging quality tier (default: standard)",
    )
    sp.add_argument(
        "--backend",
        type=str,
        default="wsclean",
        choices=["wsclean", "tclean"],
        help="Imaging backend (default: wsclean)",
    )
    sp.add_argument(
        "--max-spws",
        type=int,
        default=None,
        help="Maximum number of SPWs to process (default: all). Useful for testing.",
    )
    sp.add_argument(
        "--parallel",
        action="store_true",
        help="Image SPWs in parallel for faster processing",
    )
    sp.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: CPU count)",
    )
    sp.add_argument(
        "--serialize-ms-access",
        action="store_true",
        help="Serialize MS access using file locking to prevent CASA table "
        "lock conflicts when multiple processes access the same MS. "
        "Recommended when processing multiple sources in parallel.",
    )
    sp.set_defaults(func=cmd_adaptive)

    sp = sub.add_parser("ese-detect", help="Detect ESE candidates from variability statistics")
    sp.add_argument(
        "--products-db",
        type=str,
        default="state/db/pipeline.sqlite3",
        help="Path to pipeline database (default: state/db/pipeline.sqlite3)",
    )
    sp.add_argument(
        "--min-sigma",
        type=float,
        default=None,
        help="Minimum sigma deviation threshold (ignored if --preset is provided)",
    )
    sp.add_argument(
        "--preset",
        type=str,
        choices=["conservative", "moderate", "sensitive"],
        default=None,
        help="Threshold preset: 'conservative' (5.0), 'moderate' (3.5), or 'sensitive' (3.0)",
    )
    sp.add_argument(
        "--source-id",
        type=str,
        default=None,
        help="Optional specific source ID to check (if not provided, checks all sources)",
    )
    sp.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute variability statistics before detection",
    )
    sp.add_argument(
        "--use-composite-scoring",
        action="store_true",
        help="Enable multi-metric composite scoring for better confidence assessment",
    )
    sp.set_defaults(func=cmd_ese_detect)

    # Batch photometry command
    sp = sub.add_parser("batch", help="Batch photometry on multiple sources across multiple images")
    sp.add_argument(
        "--source-list",
        type=str,
        required=True,
        help="CSV file with columns: name, ra, dec",
    )
    sp.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing FITS images (searches recursively for *.fits)",
    )
    sp.add_argument(
        "--image-list",
        type=str,
        default=None,
        help="Text file with one image path per line (alternative to --image-dir)",
    )
    sp.add_argument(
        "--output",
        type=str,
        default="batch_photometry.csv",
        help="Output CSV file for results (default: batch_photometry.csv)",
    )
    sp.add_argument(
        "--products-db",
        type=str,
        default="state/db/pipeline.sqlite3",
        help="Path to pipeline database for storing results (default: state/db/pipeline.sqlite3)",
    )
    sp.add_argument(
        "--box",
        type=int,
        default=5,
        help="Box size for peak measurement in pixels (default: 5)",
    )
    sp.add_argument(
        "--annulus",
        type=int,
        nargs=2,
        default=[12, 20],
        help="Annulus for RMS estimation: inner outer (default: 12 20)",
    )
    sp.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output including warnings",
    )
    sp.set_defaults(func=cmd_batch)

    # Export photometry command
    sp = sub.add_parser("export", help="Export photometry measurements for a source")
    sp.add_argument(
        "--source-id",
        type=str,
        required=True,
        help="Source identifier to export measurements for",
    )
    sp.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (default: stdout). Extension determines format: .csv, .json, .vot",
    )
    sp.add_argument(
        "--format",
        type=str,
        choices=["csv", "json", "votable"],
        default="csv",
        help="Output format (default: csv). Overrides extension-based detection.",
    )
    sp.add_argument(
        "--products-db",
        type=str,
        default="state/db/pipeline.sqlite3",
        help="Path to pipeline database (default: state/db/pipeline.sqlite3)",
    )
    sp.set_defaults(func=cmd_export)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    # Input validation
    if hasattr(args, "fits") and args.fits:
        if not os.path.exists(args.fits):
            raise FileNotFoundError(f"FITS file not found: {args.fits}")
    if hasattr(args, "ms") and args.ms:
        if not os.path.exists(args.ms):
            raise FileNotFoundError(f"MS file not found: {args.ms}")
    if hasattr(args, "ra") and args.ra is not None:
        if not (-180 <= args.ra <= 360):
            raise ValueError(f"RA must be between -180 and 360 degrees, got {args.ra}")
    if hasattr(args, "dec") and args.dec is not None:
        if not (-90 <= args.dec <= 90):
            raise ValueError(f"Dec must be between -90 and 90 degrees, got {args.dec}")

    if not hasattr(args, "func"):
        p.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
