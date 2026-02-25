"""
CLI for DSA-110 calibration.

Provides command-line interface for calibrating Measurement Sets.

Example usage:
    # Full calibration sequence for 0834+555
    python -m dsa110_contimg.core.calibration.cli calibrate \
        --ms "${CONTIMG_STAGING_DIR}/ms/2025-12-05T12:30:00.ms" \
        --calibrator 0834+555 \
        --field 12 \
        --refant 3

    # Phaseshift only (no calibration)
    python -m dsa110_contimg.core.calibration.cli phaseshift \
        --ms "${CONTIMG_STAGING_DIR}/ms/obs.ms" \
        --calibrator 0834+555 \
        --field 12 \
        --output "${CONTIMG_STAGING_DIR}/ms/obs_cal.ms"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Run full calibration sequence."""
    from dsa110_contimg.core.calibration.runner import run_calibrator

    setup_logging(args.verbose)

    if not Path(args.ms).exists():
        logger.error("MS not found: %s", args.ms)
        return 1

    try:
        caltables = run_calibrator(
            ms_path=args.ms,
            cal_field=args.field,
            refant=args.refant,
            calibrator_name=args.calibrator,
            do_flagging=not args.no_flagging,
            do_k=args.do_delay,
            do_phaseshift=not args.no_phaseshift,
            table_prefix=args.output_prefix,
        )

        logger.info(" Calibration complete. Created tables:")
        for ct in caltables:
            logger.info("  - %s", ct)

        return 0

    except Exception as e:
        logger.error("Calibration failed: %s", e)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_phaseshift(args: argparse.Namespace) -> int:
    """Phaseshift calibrator field to calibrator position."""
    from dsa110_contimg.core.calibration.runner import phaseshift_ms

    setup_logging(args.verbose)

    if not Path(args.ms).exists():
        logger.error("MS not found: %s", args.ms)
        return 1

    try:
        output_ms, phasecenter = phaseshift_ms(
            ms_path=args.ms,
            field=args.field,
            mode="calibrator",
            calibrator_name=args.calibrator,
            output_ms=args.output,
        )

        logger.info(" Phaseshift complete")
        logger.info("  Output MS: %s", output_ms)
        logger.info("  Phasecenter: %s", phasecenter)

        return 0

    except Exception as e:
        logger.error("Phaseshift failed: %s", e)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_flag_diagnostics(args: argparse.Namespace) -> int:
    """Generate flagging diagnostics for a calibration table."""
    from dsa110_contimg.core.visualization.calibration_plots import plot_flagging_diagnostics

    setup_logging(args.verbose)
    outputs = plot_flagging_diagnostics(
        args.caltable,
        output=args.output,
        time_bin_sec=args.time_bin,
    )
    for path in outputs:
        logger.info("Generated: %s", path)
    return 0


def cmd_plot_dterms(args: argparse.Namespace) -> int:
    """Plot D-term leakage diagnostics."""
    from dsa110_contimg.core.visualization.calibration_plots import plot_dterm_scatter

    setup_logging(args.verbose)
    outputs = plot_dterm_scatter(args.caltable, output=args.output)
    for path in outputs:
        logger.info("Generated: %s", path)
    return 0


def cmd_compare_gains(args: argparse.Namespace) -> int:
    """Compare two calibration tables."""
    from dsa110_contimg.core.calibration.qa_compare import compare_caltables
    from dsa110_contimg.core.visualization.calibration_plots import plot_gain_comparison

    setup_logging(args.verbose)
    outputs = plot_gain_comparison(args.caltable_a, args.caltable_b, output=args.output)
    for path in outputs:
        logger.info("Generated: %s", path)

    comparison = compare_caltables(args.caltable_a, args.caltable_b)
    logger.info("Delta flagged fraction: %.4f", comparison["deltas"]["fraction_flagged"])
    return 0


def cmd_plot_snr(args: argparse.Namespace) -> int:
    """Plot SNR diagnostics for a calibration table."""
    from dsa110_contimg.core.visualization.calibration_plots import plot_gain_snr

    setup_logging(args.verbose)
    outputs = plot_gain_snr(args.caltable, output=args.output)
    for path in outputs:
        logger.info("Generated: %s", path)
    return 0


def cmd_flux_bootstrap(args: argparse.Namespace) -> int:
    """Bootstrap flux scale from primary to secondary calibrator."""
    from dsa110_contimg.core.calibration.fluxscale import cli_flux_bootstrap

    return cli_flux_bootstrap(args)


def cmd_setjy(args: argparse.Namespace) -> int:
    """Set MODEL_DATA for primary flux calibrator."""
    from dsa110_contimg.core.calibration.fluxscale import cli_setjy

    return cli_setjy(args)


def cmd_list_flux_calibrators(args: argparse.Namespace) -> int:
    """List primary flux calibrators."""
    from dsa110_contimg.core.calibration.fluxscale import cli_list_primary_calibrators

    return cli_list_primary_calibrators(args)


def cmd_check_flux_calibrator(args: argparse.Namespace) -> int:
    """Check if a source is a primary flux calibrator."""
    from dsa110_contimg.core.calibration.fluxscale import (
        get_primary_calibrator_info,
        is_primary_flux_calibrator,
    )

    setup_logging(args.verbose)

    if is_primary_flux_calibrator(args.name):
        info = get_primary_calibrator_info(args.name)
        logger.info(" '%s' IS a primary flux calibrator", args.name)
        logger.info("  Canonical name: %s", info["canonical_name"])
        logger.info("  Flux (1.4 GHz): %.2f Jy", info["flux_1400mhz_jy"])
        logger.info("  Spectral index: %.3f", info["spectral_index"])
        return 0
    else:
        logger.info(" '%s' is NOT a primary flux calibrator", args.name)
        logger.info("  Use 'list-flux-calibrators' to see valid calibrators")
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        prog="dsa110_contimg.core.calibration.cli",
        description="DSA-110 Calibration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full calibration for 0834+555
  python -m dsa110_contimg.core.calibration.cli calibrate \\
      --ms "${CONTIMG_STAGING_DIR}/ms/obs.ms" \\
      --calibrator 0834+555 --field 12 --refant 3

  # Phaseshift only
  python -m dsa110_contimg.core.calibration.cli phaseshift \\
      --ms "${CONTIMG_STAGING_DIR}/ms/obs.ms" --calibrator 0834+555 --field 12
""",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    # calibrate subcommand
    cal_parser = sub.add_parser(
        "calibrate",
        help="Run full calibration sequence (phaseshift → model → bandpass → gains)",
        description=(
            "Performs complete calibration sequence for DSA-110 data:\n"
            "1. Phaseshift calibrator field to calibrator position\n"
            "2. Set model visibilities from catalog\n"
            "3. Solve bandpass\n"
            "4. Solve time-dependent gains"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cal_parser.add_argument("--ms", required=True, help="Path to Measurement Set")
    cal_parser.add_argument(
        "--calibrator",
        required=True,
        help="Calibrator name (e.g., '0834+555', '3C286')",
    )
    cal_parser.add_argument(
        "--field",
        required=True,
        help="Field selection (e.g., '12' or '11~13')",
    )
    cal_parser.add_argument(
        "--refant",
        default="104,105,106,107,108,109,110,111,112,113,114,115,116,103,117",
        help="Reference antenna chain (default: outrigger priority 104,105,...)",
    )
    cal_parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix for calibration table names (default: auto)",
    )
    cal_parser.add_argument(
        "--no-flagging",
        action="store_true",
        help="Skip pre-calibration flagging",
    )
    cal_parser.add_argument(
        "--do-delay",
        action="store_true",
        help="Solve for K (delay) calibration",
    )
    cal_parser.add_argument(
        "--no-phaseshift",
        action="store_true",
        help="Skip phaseshift (only if data already phased to calibrator)",
    )
    cal_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    cal_parser.set_defaults(func=cmd_calibrate)

    # phaseshift subcommand
    ps_parser = sub.add_parser(
        "phaseshift",
        help="Phaseshift calibrator field to calibrator position only",
        description=(
            "Extracts calibrator field and phaseshifts to calibrator's true position.\n"
            "This is step 1 of calibration - use 'calibrate' for full sequence."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ps_parser.add_argument("--ms", required=True, help="Path to input Measurement Set")
    ps_parser.add_argument(
        "--calibrator",
        required=True,
        help="Calibrator name (e.g., '0834+555')",
    )
    ps_parser.add_argument(
        "--field",
        required=True,
        help="Field selection (e.g., '12')",
    )
    ps_parser.add_argument(
        "--output",
        default=None,
        help="Output MS path (default: {input}_cal.ms)",
    )
    ps_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    ps_parser.set_defaults(func=cmd_phaseshift)

    # flagging diagnostics
    flag_parser = sub.add_parser(
        "diag-flags",
        help="Generate flagging diagnostics for a calibration table",
    )
    flag_parser.add_argument("--caltable", required=True, help="Path to calibration table")
    flag_parser.add_argument("--time-bin", type=float, default=120.0, help="Time bin (seconds)")
    flag_parser.add_argument("--output", default=None, help="Output prefix (PNG/JSON)")
    flag_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    flag_parser.set_defaults(func=cmd_flag_diagnostics)

    # D-term plotting
    dterm_parser = sub.add_parser(
        "plot-dterms",
        help="Plot D-term leakage scatter and histograms",
    )
    dterm_parser.add_argument("--caltable", required=True, help="Path to D-term calibration table")
    dterm_parser.add_argument("--output", default=None, help="Output prefix (PNG/JSON)")
    dterm_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    dterm_parser.set_defaults(func=cmd_plot_dterms)

    # Gain comparison
    compare_parser = sub.add_parser(
        "compare-gains",
        help="Compare two calibration tables (amplitude/phase per antenna)",
    )
    compare_parser.add_argument("--caltable-a", required=True, help="Reference calibration table")
    compare_parser.add_argument("--caltable-b", required=True, help="Comparison calibration table")
    compare_parser.add_argument("--output", default=None, help="Output prefix (PNG)")
    compare_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    compare_parser.set_defaults(func=cmd_compare_gains)

    # SNR diagnostics
    snr_parser = sub.add_parser(
        "plot-snr",
        help="Plot SNR histograms and time series from a calibration table",
    )
    snr_parser.add_argument("--caltable", required=True, help="Path to calibration table")
    snr_parser.add_argument("--output", default=None, help="Output prefix (PNG/JSON)")
    snr_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    snr_parser.set_defaults(func=cmd_plot_snr)

    # =========================================================================
    # Flux Calibration Commands
    # =========================================================================

    # flux-bootstrap subcommand
    flux_parser = sub.add_parser(
        "flux-bootstrap",
        help="Bootstrap absolute flux scale from primary to secondary calibrator",
        description=(
            "Transfers the VLA flux scale from a primary flux calibrator (3C286, 3C48, etc.)\n"
            "to a secondary calibrator using CASA fluxscale.\n\n"
            "Workflow:\n"
            "  1. setjy: Set MODEL_DATA on primary (Perley-Butler 2017)\n"
            "  2. gaincal: Solve gains on both calibrators\n"
            "  3. fluxscale: Transfer flux from primary → secondary\n\n"
            "The derived flux can be recorded to the pipeline database and used\n"
            "for future calibrations of the secondary calibrator."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    flux_parser.add_argument(
        "--ms-primary",
        required=True,
        help="Path to MS containing primary flux calibrator",
    )
    flux_parser.add_argument(
        "--ms-secondary",
        default=None,
        help="Path to MS containing secondary calibrator (if different from primary)",
    )
    flux_parser.add_argument(
        "--primary-name",
        required=True,
        help="Primary calibrator name (e.g., '3C286', '3C48')",
    )
    flux_parser.add_argument(
        "--secondary-name",
        required=True,
        help="Secondary calibrator name (e.g., '0834+555')",
    )
    flux_parser.add_argument(
        "--primary-field",
        default="",
        help="Field selection for primary calibrator (default: auto)",
    )
    flux_parser.add_argument(
        "--secondary-field",
        default="",
        help="Field selection for secondary calibrator (default: auto)",
    )
    flux_parser.add_argument(
        "--refant",
        default="104,105,106,107,108,109,110,111,112,113,114,115,116,103,117",
        help="Reference antenna chain",
    )
    flux_parser.add_argument(
        "--output-dir",
        default=os.path.join(os.getenv("CONTIMG_BASE_DIR", "."), "caltables"),
        help="Directory for output calibration tables",
    )
    flux_parser.add_argument(
        "--solint",
        default="inf",
        help="Solution interval (default: 'inf')",
    )
    flux_parser.add_argument(
        "--minsnr",
        type=float,
        default=3.0,
        help="Minimum SNR for solutions (default: 3.0)",
    )
    flux_parser.add_argument(
        "--combine-spw",
        action="store_true",
        help="Combine SPWs for gain solutions",
    )
    flux_parser.add_argument(
        "--single-ms",
        action="store_true",
        help="Both calibrators are in the same MS (--ms-primary)",
    )
    flux_parser.add_argument(
        "--no-apply",
        action="store_true",
        help="Don't apply flux-scaled gains (single-MS mode only)",
    )
    flux_parser.add_argument(
        "--record-db",
        action="store_true",
        help="Record result to pipeline database",
    )
    flux_parser.add_argument(
        "--update-catalog",
        action="store_true",
        help="Update calibrator catalog with derived flux",
    )
    flux_parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output result as JSON",
    )
    flux_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    flux_parser.set_defaults(func=cmd_flux_bootstrap)

    # setjy subcommand
    setjy_parser = sub.add_parser(
        "setjy",
        help="Set MODEL_DATA for primary flux calibrator using Perley-Butler 2017",
        description=(
            "Uses CASA setjy to populate MODEL_DATA with accurate flux values\n"
            "from the Perley-Butler 2017 flux density scale.\n\n"
            "Only works for primary VLA flux calibrators: 3C286, 3C48, 3C147, 3C138, 3C295, 3C196"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    setjy_parser.add_argument("--ms", required=True, help="Path to Measurement Set")
    setjy_parser.add_argument(
        "--calibrator",
        required=True,
        help="Primary calibrator name (e.g., '3C286')",
    )
    setjy_parser.add_argument(
        "--field",
        default="",
        help="Field selection (default: auto-detect from calibrator name)",
    )
    setjy_parser.add_argument(
        "--standard",
        default="Perley-Butler 2017",
        help="Flux density standard (default: 'Perley-Butler 2017')",
    )
    setjy_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    setjy_parser.set_defaults(func=cmd_setjy)

    # list-flux-calibrators subcommand
    list_flux_parser = sub.add_parser(
        "list-flux-calibrators",
        help="List all primary VLA flux calibrators",
        description="Shows all calibrators that can be used with setjy and flux-bootstrap.",
    )
    list_flux_parser.set_defaults(func=cmd_list_flux_calibrators)

    # check-flux-calibrator subcommand
    check_flux_parser = sub.add_parser(
        "check-flux-calibrator",
        help="Check if a source is a primary flux calibrator",
    )
    check_flux_parser.add_argument("name", help="Calibrator name to check")
    check_flux_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    check_flux_parser.set_defaults(func=cmd_check_flux_calibrator)

    return parser


def main(argv: list | None = None) -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 2

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
