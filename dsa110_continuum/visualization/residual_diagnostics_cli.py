#!/usr/bin/env python
r"""
CLI tool for visibility residual diagnostics.

Generates comprehensive residual analysis plots for calibrated Measurement Sets.

Examples
--------
Generate all residual diagnostic plots::

    python -m dsa110_contimg.visualization.residual_diagnostics_cli observation.ms --output-dir /data/dsa110-contimg/tmp/qa/

Generate interactive Vega-Lite specs for dashboard::

    python -m dsa110_contimg.visualization.residual_diagnostics_cli observation.ms --output-dir /data/dsa110-contimg/tmp/qa/ --interactive

Use specific data columns::

    python -m dsa110_contimg.visualization.residual_diagnostics_cli observation.ms --data-column DATA --model-column MODEL_DATA

After calibration, check residual quality::

    python -m dsa110_contimg.visualization.residual_diagnostics_cli \\
        /stage/dsa110-contimg/ms/2025-01-15_12h00m00s.ms \\
        --output-dir /dev/shm/qa/residuals/

Notes
-----
Output files generated:

- ``{ms_name}_residual_vs_baseline.png`` - Amplitude vs UV distance
- ``{ms_name}_residual_phase_vs_time.png`` - Phase vs time
- ``{ms_name}_residual_histogram.png`` - Real/Imag histograms
- ``{ms_name}_residual_complex.png`` - Complex plane scatter
- ``{ms_name}_residual_per_antenna.png`` - Per-antenna RMS bar chart
- ``{ms_name}_residual_statistics.json`` - Summary statistics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate visibility residual diagnostics for calibrated MS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "ms_path",
        type=str,
        help="Path to Measurement Set",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for plots (default: same as MS)",
    )
    parser.add_argument(
        "--data-column",
        type=str,
        default="CORRECTED_DATA",
        choices=["DATA", "CORRECTED_DATA"],
        help="Data column to use (default: CORRECTED_DATA)",
    )
    parser.add_argument(
        "--model-column",
        type=str,
        default="MODEL_DATA",
        help="Model column to use (default: MODEL_DATA)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate Vega-Lite JSON specs instead of PNG",
    )
    parser.add_argument(
        "--no-average-channels",
        action="store_true",
        help="Don't average over frequency channels",
    )
    parser.add_argument(
        "--channel-range",
        type=str,
        default=None,
        help="Channel range to use, e.g., '100:200'",
    )
    parser.add_argument(
        "--include-autocorr",
        action="store_true",
        help="Include auto-correlations in analysis",
    )
    parser.add_argument(
        "--json-stats",
        action="store_true",
        help="Also save statistics to JSON file",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress info messages",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug messages",
    )

    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    ms_path = Path(args.ms_path)
    if not ms_path.exists():
        logger.error(f"MS not found: {ms_path}")
        sys.exit(1)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ms_path.parent / "qa" / "residuals"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Parse channel range
    channel_range = None
    if args.channel_range:
        try:
            start, end = args.channel_range.split(":")
            channel_range = (int(start), int(end))
        except ValueError:
            logger.error(f"Invalid channel range: {args.channel_range}. Use format 'start:end'")
            sys.exit(1)

    try:
        from dsa110_contimg.core.visualization.config import FigureConfig, PlotStyle
        from dsa110_contimg.core.visualization.plot_context import PlotContext
        from dsa110_contimg.core.visualization.residual_diagnostics import (
            compute_residual_statistics,
            extract_residuals_from_ms,
            plot_residual_amplitude_vs_baseline,
            plot_residual_complex_scatter,
            plot_residual_histogram,
            plot_residual_per_antenna,
            plot_residual_phase_vs_time,
        )

        config = FigureConfig(style=PlotStyle.QUICKLOOK)
        context = PlotContext.API if args.interactive else PlotContext.PIPELINE

        # Extract residuals
        logger.info(f"Extracting residuals from {ms_path}")
        logger.info(f"  Data column: {args.data_column}")
        logger.info(f"  Model column: {args.model_column}")

        data = extract_residuals_from_ms(
            ms_path,
            data_column=args.data_column,
            model_column=args.model_column,
            average_channels=not args.no_average_channels,
            channel_range=channel_range,
            exclude_autocorr=not args.include_autocorr,
        )

        logger.info(f"Extracted {data.n_baselines} baselines × {data.n_times} times")
        logger.info(f"  Channels: {data.n_channels}, Polarizations: {data.n_pols}")

        # Compute statistics
        stats = compute_residual_statistics(data)
        logger.info("Residual statistics:")
        logger.info(f"  Flag fraction: {stats.flag_fraction:.1%}")
        logger.info(f"  RMS amplitude: {stats.rms_amplitude:.4e}")
        logger.info(f"  Phase scatter: {stats.std_phase_deg:.1f}°")
        logger.info(f"  Outlier fraction: {stats.outlier_fraction:.2%}")

        ms_name = ms_path.stem
        generated_plots = []

        # Generate plots
        ext = ".vega.json" if args.interactive else ".png"

        # 1. Amplitude vs baseline
        plot_path = output_dir / f"{ms_name}_residual_vs_baseline{ext}"
        plot_residual_amplitude_vs_baseline(
            data, output=plot_path, config=config, context=context, interactive=args.interactive
        )
        generated_plots.append(str(plot_path))
        logger.info(f"Generated: {plot_path}")

        # 2. Phase vs time
        plot_path = output_dir / f"{ms_name}_residual_phase_vs_time{ext}"
        plot_residual_phase_vs_time(
            data, output=plot_path, config=config, context=context, interactive=args.interactive
        )
        generated_plots.append(str(plot_path))
        logger.info(f"Generated: {plot_path}")

        # 3. Histogram
        plot_path = output_dir / f"{ms_name}_residual_histogram{ext}"
        plot_residual_histogram(
            data, output=plot_path, config=config, context=context, interactive=args.interactive
        )
        generated_plots.append(str(plot_path))
        logger.info(f"Generated: {plot_path}")

        # 4. Complex scatter
        plot_path = output_dir / f"{ms_name}_residual_complex{ext}"
        plot_residual_complex_scatter(
            data, output=plot_path, config=config, context=context, interactive=args.interactive
        )
        generated_plots.append(str(plot_path))
        logger.info(f"Generated: {plot_path}")

        # 5. Per-antenna RMS
        plot_path = output_dir / f"{ms_name}_residual_per_antenna{ext}"
        plot_residual_per_antenna(
            data,
            stats,
            output=plot_path,
            config=config,
            context=context,
            interactive=args.interactive,
        )
        generated_plots.append(str(plot_path))
        logger.info(f"Generated: {plot_path}")

        # Save statistics to JSON if requested
        if args.json_stats or args.interactive:
            stats_path = output_dir / f"{ms_name}_residual_statistics.json"
            with open(stats_path, "w") as f:
                json.dump(stats.to_dict(), f, indent=2)
            logger.info(f"Saved statistics: {stats_path}")

        # Print quality assessment
        from dsa110_contimg.core.visualization.residual_diagnostics import _assess_residual_quality

        assessment = _assess_residual_quality(stats)
        print("\n" + "=" * 60)
        print(
            f"RESIDUAL QUALITY ASSESSMENT: {assessment['overall']} (score: {assessment['score']}/100)"
        )
        print("=" * 60)

        if assessment["issues"]:
            print("\n  ISSUES:")
            for issue in assessment["issues"]:
                print(f"   • {issue}")

        if assessment["warnings"]:
            print("\n WARNINGS:")
            for warning in assessment["warnings"]:
                print(f"   • {warning}")

        print(f"\n {assessment['recommendation']}")
        print("=" * 60 + "\n")

        logger.info(f"Generated {len(generated_plots)} diagnostic plots")

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure you are in the casa6 conda environment")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error generating residual diagnostics: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
