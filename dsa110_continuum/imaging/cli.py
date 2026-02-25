"""
CLI to image a Measurement Set using CASA tclean or WSClean.
WSClean is the default backend for faster imaging.

Selects CORRECTED_DATA when present; otherwise falls back to DATA.
Performs primary-beam correction and exports FITS products.

Supports hybrid workflow: CASA ft() for model seeding + WSClean for fast imaging.
"""

# Initialize CASA environment before importing CASA modules
from dsa110_contimg.common.utils.casa_init import ensure_casa_path
from dsa110_contimg.common.utils.cli_helpers import (
    configure_logging_from_args,
    ensure_scratch_dirs,
    setup_casa_environment,
)

from .cli_imaging import image_ms

ensure_casa_path()

import casacore.tables as casatables  # type: ignore[import]

table = casatables.table  # noqa: N816
import argparse
import logging
import os

import casacore.tables as casatables  # type: ignore[import]

table = casatables.table  # noqa: N816


logger = logging.getLogger(__name__)

# Use shared CLI utilities

# Set CASA log directory BEFORE any CASA imports - CASA writes logs to CWD
setup_casa_environment()

try:
    from casatools import msmetadata as _msmd  # type: ignore[import]
    from casatools import vpmanager as _vpmanager  # type: ignore[import]
except ImportError:  # pragma: no cover
    _vpmanager = None
    _msmd = None

LOG = logging.getLogger(__name__)

try:
    # Ensure temp artifacts go to scratch and not the repo root
    from dsa110_contimg.common.utils.run_isolation import prepare_temp_environment
except ImportError:  # pragma: no cover - defensive import
    prepare_temp_environment = None  # type: ignore


# NOTE: _configure_logging() has been removed. Use configure_logging_from_args() instead.
# This function was deprecated and unused. All logging now uses the shared utility.


# Utility functions moved to cli_utils.py

# Core imaging functions moved to cli_imaging.py


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(description="DSA-110 Imaging CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # image subcommand (main imaging functionality)
    img_parser = sub.add_parser(
        "image",
        help="Image an MS with tclean or WSClean (WSClean is default)",
        description=(
            "Create images from a Measurement Set using CASA tclean or WSClean. "
            "WSClean is the default backend for faster imaging. "
            "Automatically selects CORRECTED_DATA when present, otherwise uses DATA.\n\n"
            "Example:\n"
            "  python -m dsa110_contimg.core.imaging.cli image \\\n"
            "    --ms /data/ms/target.ms --imagename /data/images/target \\\n"
            "    --imsize 2048 --cell-arcsec 1.0 --quality-tier standard"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    img_parser.add_argument("--ms", required=True, help="Path to input MS")
    img_parser.add_argument("--imagename", required=True, help="Output image name prefix")
    img_parser.add_argument("--field", default="", help="Field selection")
    img_parser.add_argument("--spw", default="", help="SPW selection")
    img_parser.add_argument("--imsize", type=int, default=1024)
    img_parser.add_argument("--cell-arcsec", type=float, default=None)
    img_parser.add_argument("--weighting", default="briggs")
    img_parser.add_argument("--robust", type=float, default=0.5)
    # Friendly synonyms matching user vocabulary
    img_parser.add_argument(
        "--weighttype",
        dest="weighting_alias",
        default=None,
        help="Alias of --weighting",
    )
    img_parser.add_argument(
        "--weight",
        dest="robust_alias",
        type=float,
        default=None,
        help="Alias of --robust (Briggs robust)",
    )
    img_parser.add_argument("--specmode", default="mfs")
    img_parser.add_argument("--deconvolver", default="hogbom")
    img_parser.add_argument("--nterms", type=int, default=1)
    img_parser.add_argument("--niter", type=int, default=1000)
    img_parser.add_argument("--threshold", default="0.005Jy")
    img_parser.add_argument("--no-pbcor", action="store_true")
    img_parser.add_argument(
        "--quality-tier",
        choices=["development", "standard", "high_precision"],
        default="standard",
        help=(
            "Imaging quality tier with explicit trade-offs.\n"
            "  development: :warning:  NON-SCIENCE - coarser resolution, fewer iterations\n"
            "  standard: Recommended for all science observations (full quality)\n"
            "  high_precision: Enhanced settings for maximum quality (slower)\n"
            "Default: standard"
        ),
    )
    img_parser.add_argument(
        "--skip-fits",
        action="store_true",
        help="Do not export FITS products after tclean",
    )
    img_parser.add_argument(
        "--phasecenter",
        default=None,
        help=("CASA phasecenter string (e.g., 'J2000 08h34m54.9 +55d34m21.1')"),
    )
    img_parser.add_argument(
        "--gridder",
        default="idg",
        help="WSClean gridder (idg|wgridder|wstacking)",
    )
    img_parser.add_argument(
        "--wprojplanes",
        type=int,
        default=-1,
        help=("Number of w-projection planes when gridder=wproject (-1 for auto)"),
    )
    img_parser.add_argument(
        "--uvrange",
        default=">1klambda",
        help="uvrange selection, e.g. '>1klambda'",
    )
    img_parser.add_argument("--pblimit", type=float, default=0.2)
    img_parser.add_argument("--psfcutoff", type=float, default=None)
    img_parser.add_argument("--verbose", action="store_true")
    # Unified catalog skymodel seeding
    img_parser.add_argument(
        "--unicat-min-mjy",
        type=float,
        default=5.0,
        help=(
            "If set, seed MODEL_DATA from unified catalog (FIRST+RACS+NVSS) sources above this flux. "
            "In development quality tier, defaults to 10.0 mJy. "
            "In high_precision tier, defaults to 5.0 mJy."
        ),
    )
    img_parser.add_argument(
        "--export-model-image",
        action="store_true",
        help=(
            "Export MODEL_DATA as FITS image after unified catalog seeding. "
            "Useful for visualizing the sky model used during imaging. "
            "Output will be saved as {imagename}.unicat_model.fits"
        ),
    )
    # Masking parameters
    img_parser.add_argument(
        "--no-unicat-mask",
        action="store_true",
        help="Disable unified catalog masking (masking is enabled by default for 2-4x faster imaging)",
    )
    img_parser.add_argument(
        "--mask-radius-arcsec",
        type=float,
        default=60.0,
        help="Mask radius around catalog sources in arcseconds (default: 60.0, ~2-3× beam)",
    )
    # A-Projection related options
    img_parser.add_argument(
        "--vptable",
        default=None,
        help="Path to CASA VP table (vpmanager.saveastable)",
    )
    img_parser.add_argument(
        "--wbawp",
        action="store_true",
        help="Enable wideband A-Projection approximation",
    )
    img_parser.add_argument(
        "--cfcache",
        default=None,
        help="Convolution function cache directory",
    )
    # Backend selection
    img_parser.add_argument(
        "--backend",
        choices=["tclean", "wsclean"],
        default="wsclean",
        help="Imaging backend: tclean (CASA) or wsclean (default: wsclean)",
    )
    img_parser.add_argument(
        "--wsclean-path",
        default=None,
        help="Path to WSClean executable (or 'docker' for Docker container). "
        "If not set, searches PATH or uses Docker if available.",
    )
    # Calibrator seeding
    img_parser.add_argument(
        "--calib-ra-deg",
        type=float,
        default=None,
        help="Calibrator RA (degrees) for single-component model seeding",
    )
    img_parser.add_argument(
        "--calib-dec-deg",
        type=float,
        default=None,
        help="Calibrator Dec (degrees) for single-component model seeding",
    )
    img_parser.add_argument(
        "--calib-flux-jy",
        type=float,
        default=None,
        help="Calibrator flux (Jy) for single-component model seeding",
    )

    # export subcommand
    exp_parser = sub.add_parser("export", help="Export CASA images to FITS and PNG")
    exp_parser.add_argument("--source", required=True, help="Directory containing CASA images")
    exp_parser.add_argument("--prefix", required=True, help="Prefix of image set")
    exp_parser.add_argument("--make-fits", action="store_true", help="Export FITS from CASA images")
    exp_parser.add_argument("--make-png", action="store_true", help="Convert FITS to PNGs")

    # create-mask subcommand
    mask_parser = sub.add_parser(
        "create-mask",
        help="Create CRTF or FITS mask from catalog sources",
        description=(
            "Create a mask for CLEAN imaging from any supported catalog.\n"
            "Supports: nvss, first, vlass, unicat (default), atnf, rax\n\n"
            "Example:\n"
            "  python -m dsa110_contimg.core.imaging.cli create-mask \\\n"
            "    --image observation.fits --catalog unicat --min-mjy 5.0"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mask_parser.add_argument("--image", required=True, help="Input FITS image path")
    mask_parser.add_argument(
        "--catalog",
        choices=["nvss", "first", "vlass", "unicat", "atnf", "rax"],
        default="unicat",
        help="Catalog to use for mask (default: unicat)",
    )
    mask_parser.add_argument("--min-mjy", type=float, default=None, help="Minimum flux in mJy")
    mask_parser.add_argument(
        "--radius-arcsec", type=float, default=60.0, help="Mask circle radius (arcsec)"
    )
    mask_parser.add_argument(
        "--format",
        choices=["crtf", "fits"],
        default="crtf",
        help="Output format: crtf (CASA region) or fits (WSClean)",
    )
    mask_parser.add_argument("--out", help="Output path (auto-generated if not specified)")

    # create-overlay subcommand
    overlay_parser = sub.add_parser(
        "create-overlay",
        help="Create diagnostic overlay image with catalog sources",
        description=(
            "Overlay catalog sources on a FITS image for visual inspection.\n"
            "Supports: nvss, first, vlass, unicat (default), atnf, rax\n\n"
            "Example:\n"
            "  python -m dsa110_contimg.core.imaging.cli create-overlay \\\n"
            "    --image observation.fits --out overlay.png --catalog nvss"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    overlay_parser.add_argument("--image", required=True, help="Input FITS image path")
    overlay_parser.add_argument("--out", required=True, help="Output PNG path")
    overlay_parser.add_argument(
        "--catalog",
        choices=["nvss", "first", "vlass", "unicat", "atnf", "rax"],
        default="unicat",
        help="Catalog to overlay (default: unicat)",
    )
    overlay_parser.add_argument("--pb", help="Primary beam FITS to mask detections (optional)")
    overlay_parser.add_argument(
        "--pblimit", type=float, default=0.2, help="PB cutoff when --pb is provided"
    )
    overlay_parser.add_argument(
        "--min-mjy", type=float, default=10.0, help="Minimum flux (mJy) to plot"
    )

    args = parser.parse_args(argv)

    # Input validation
    if hasattr(args, "ms") and args.ms:
        if not os.path.exists(args.ms):
            raise FileNotFoundError(f"MS file not found: {args.ms}")
    if hasattr(args, "imagename") and args.imagename:
        output_dir = os.path.dirname(args.imagename) if os.path.dirname(args.imagename) else "."
        if not os.path.exists(output_dir):
            raise ValueError(f"Output directory does not exist: {output_dir}")

    # Configure logging using shared utility
    configure_logging_from_args(args)

    # Ensure scratch directory structure exists
    try:
        ensure_scratch_dirs()
    except OSError:
        pass  # Best-effort; continue if setup fails

    if args.cmd == "image":
        # Apply aliases if provided
        weighting = args.weighting_alias if args.weighting_alias else args.weighting
        robust = args.robust_alias if args.robust_alias is not None else args.robust

        # Provenance Injection for Manual CLI Runs
        try:
            from dsa110_contimg.core.conversion.ms_utils import inject_provenance_metadata
            from dsa110_contimg.infrastructure.tracking.provenance import ProvenanceTracker
            import uuid
            import time
            
            job_id = str(uuid.uuid4())
            tracker = ProvenanceTracker(job_id=job_id)
            tracker.set_config({
                "service": "manual_cli",
                "command": "image_ms",
                "ms_path": args.ms,
                "params": {
                    "imagename": args.imagename,
                    "field": args.field,
                    "quality": args.quality_tier
                },
                "timestamp": time.time()
            })
            
            if tracker.provenance.config_hash:
                inject_provenance_metadata(args.ms, job_id, tracker.provenance.config_hash)
                tracker.save()
                print(f"Provenance injected: Job ID {job_id}")
        except Exception as e:
            print(f"Warning: Provenance injection failed: {e}")

        image_ms(
            args.ms,
            imagename=args.imagename,
            field=args.field,
            spw=args.spw,
            imsize=args.imsize,
            cell_arcsec=args.cell_arcsec,
            weighting=weighting,
            robust=robust,
            specmode=args.specmode,
            deconvolver=args.deconvolver,
            nterms=args.nterms,
            niter=args.niter,
            threshold=args.threshold,
            pbcor=not args.no_pbcor,
            phasecenter=args.phasecenter,
            gridder=args.gridder,
            wprojplanes=args.wprojplanes,
            uvrange=args.uvrange,
            pblimit=args.pblimit,
            psfcutoff=args.psfcutoff,
            quality_tier=args.quality_tier,
            skip_fits=bool(args.skip_fits),
            vptable=args.vptable,
            wbawp=bool(args.wbawp),
            cfcache=args.cfcache,
            unicat_min_mjy=args.unicat_min_mjy,
            calib_ra_deg=args.calib_ra_deg,
            calib_dec_deg=args.calib_dec_deg,
            calib_flux_jy=args.calib_flux_jy,
            backend=args.backend,
            wsclean_path=args.wsclean_path,
            export_model_image=args.export_model_image,
            use_unicat_mask=not args.no_unicat_mask,
            mask_radius_arcsec=args.mask_radius_arcsec,
        )

    elif args.cmd == "export":
        from glob import glob

        from dsa110_contimg.core.imaging.export import (
            _find_casa_images,
            export_fits,
            save_png_from_fits,
        )

        casa_images = _find_casa_images(args.source, args.prefix)
        if not casa_images:
            logger.warning(
                f"No CASA image directories found for prefix {args.prefix} under {args.source}"
            )
            print(
                "No CASA image directories found for prefix",
                args.prefix,
                "under",
                args.source,
            )
            return

        fits_paths: list[str] = []
        if args.make_fits:
            fits_paths = export_fits(casa_images)
            if not fits_paths:
                logger.warning("No FITS files exported (check casatasks and inputs)")
                print("No FITS files exported (check casatasks and inputs)")
        if args.make_png:
            # If FITS were not just created, try to discover existing ones
            if not fits_paths:
                patt = os.path.join(args.source, args.prefix + "*.fits")
                fits_paths = sorted(glob(patt))
            if not fits_paths:
                logger.warning(f"No FITS files found to convert for {args.prefix}")
                print("No FITS files found to convert for", args.prefix)
            else:
                save_png_from_fits(fits_paths)

    elif args.cmd == "create-mask":
        from dsa110_contimg.core.imaging.catalog_tools import create_catalog_mask

        out_path = create_catalog_mask(
            image_path=args.image,
            catalog=args.catalog,
            min_flux_mjy=args.min_mjy,
            radius_arcsec=args.radius_arcsec,
            out_path=args.out,
            output_format=args.format,
        )
        print(f"Wrote mask: {out_path}")

    elif args.cmd == "create-overlay":
        from dsa110_contimg.core.imaging.catalog_tools import create_catalog_overlay

        create_catalog_overlay(
            image_path=args.image,
            out_path=args.out,
            catalog=args.catalog,
            min_flux_mjy=args.min_mjy,
            pb_path=args.pb,
            pblimit=args.pblimit,
        )
        print(f"Wrote overlay: {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
