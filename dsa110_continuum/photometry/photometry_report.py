"""
Photometry Diagnostics HTML Report Generator.

Generates comprehensive HTML reports for source photometry assessment,
including flux measurements, SNR, comparisons, and recommendations.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from dsa110_contimg.common.utils.template_styles import get_shared_css

logger = logging.getLogger(__name__)


@dataclass
class PhotometryReportData:
    """Container for all data needed to generate the photometry report."""

    # Identification
    fits_path: str
    source_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Source coordinates
    ra_deg: float = 0.0
    dec_deg: float = 0.0

    # Measurement parameters
    box_size: int = 0
    annulus_radii: list[int] = field(default_factory=lambda: [])
    method: str = "gaussian"

    # Photometry results
    peak_flux_jy: float = 0.0
    local_rms_jy: float = 0.0
    snr: float = 0.0
    integrated_flux_jy: float = 0.0

    # Expected values for comparison (optional)
    expected_flux_jy: float | None = None
    flux_ratio: float | None = None
    flux_deviation_percent: float | None = None

    # Quality assessment
    quality_grade: str = "unknown"
    recommendations: list[str] = field(default_factory=list)

    # Multiple source measurements (for multi-source reports)
    source_measurements: list[dict[str, Any]] = field(default_factory=list)

    # Figure base64 encoded images
    figure_base64: dict[str, str] = field(default_factory=dict)


def load_photometry_report_data(
    fits_path: str,
    ra_deg: float,
    dec_deg: float,
    measurement_result: dict[str, Any] | None = None,
    source_name: str = "",
    expected_flux_jy: float | None = None,
    box_size: int = 15,
    annulus_radii: list[int] | None = None,
) -> PhotometryReportData:
    """
    Load all data needed to generate the photometry diagnostics report.

    Parameters
    ----------
    fits_path : str
        Path to the FITS image
    ra_deg : float
        Right ascension in degrees
    dec_deg : float
        Declination in degrees
    measurement_result : Optional[Dict[str, Any]]
        Result from measure_forced_peak or similar
    source_name : str
        Name of the source being measured
    expected_flux_jy : Optional[float]
        Expected flux in Jy for comparison
    box_size : int
        Box size used for measurement
    annulus_radii : Optional[List[int]]
        Annulus radii used for background estimation

    Returns
    -------
    PhotometryReportData
        Data container for report generation
    """
    report = PhotometryReportData(
        fits_path=fits_path,
        source_name=source_name or f"RA={ra_deg:.4f}, Dec={dec_deg:.4f}",
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        box_size=box_size,
        annulus_radii=annulus_radii or [20, 30],
    )

    # Use provided measurement results
    if measurement_result:
        report.peak_flux_jy = measurement_result.get("peak_jyb", 0.0)
        report.local_rms_jy = measurement_result.get("local_rms_jy", 0.0) or measurement_result.get(
            "peak_err_jyb", 0.0
        )
        report.snr = measurement_result.get("snr", 0.0)
        report.integrated_flux_jy = measurement_result.get("integrated_flux_jy", 0.0)
        report.method = measurement_result.get("method", "gaussian")
        # Calculate SNR if not provided but we have peak and rms
        if report.snr == 0.0 and report.local_rms_jy > 0:
            report.snr = report.peak_flux_jy / report.local_rms_jy
    else:
        # Try to measure from the FITS file directly
        try:
            from dsa110_contimg.core.photometry.forced import measure_forced_peak

            result = measure_forced_peak(
                fits_path=fits_path,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                box_size_pix=box_size,
                annulus_pix=tuple(annulus_radii or [20, 30]),
            )
            report.peak_flux_jy = result.peak_jyb
            report.local_rms_jy = result.peak_err_jyb  # Use error as proxy for RMS
            if report.local_rms_jy > 0:
                report.snr = report.peak_flux_jy / report.local_rms_jy
        except Exception as err:
            logger.warning("Failed to measure photometry: %s", err)

    # Compare to expected flux if provided
    if expected_flux_jy is not None:
        report.expected_flux_jy = expected_flux_jy
        if expected_flux_jy > 0 and report.peak_flux_jy > 0:
            report.flux_ratio = report.peak_flux_jy / expected_flux_jy
            report.flux_deviation_percent = (report.flux_ratio - 1.0) * 100

    # Quality assessment based on SNR
    if report.snr >= 100:
        report.quality_grade = "Excellent"
    elif report.snr >= 20:
        report.quality_grade = "Good"
    elif report.snr >= 5:
        report.quality_grade = "Acceptable"
    elif report.snr > 0:
        report.quality_grade = "Marginal"
    else:
        report.quality_grade = "Unknown"

    # Generate recommendations
    if report.snr < 5:
        report.recommendations.append(
            f"SNR ({report.snr:.1f}) is below detection threshold. Source may not be detected."
        )
    if report.flux_deviation_percent is not None and abs(report.flux_deviation_percent) > 20:
        report.recommendations.append(
            f"Flux deviation ({report.flux_deviation_percent:.1f}%) exceeds 20%. Check calibration."
        )
    if report.local_rms_jy > 0.01:
        report.recommendations.append(
            f"Local RMS ({report.local_rms_jy * 1000:.2f} mJy) is high. Image may have artifacts."
        )

    return report


def generate_photometry_figures(
    report: PhotometryReportData,
    save_pngs_dir: str | None = None,
    image_prefix: str = "photometry",
) -> dict[str, str]:
    """
    Generate diagnostic figures and return as base64-encoded strings.

    Parameters
    ----------
    report : PhotometryReportData
        The report data container
    save_pngs_dir : Optional[str]
        If provided, save PNG files to this directory in addition to embedding
    image_prefix : str
        Prefix for saved PNG filenames (default: "photometry")

    Returns
    -------
    Dict[str, str]
        Dictionary mapping figure names to base64-encoded PNG data
    """
    figures = {}

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError:
        logger.warning("matplotlib not available, skipping figure generation")
        return figures

    # Create PNG output directory if specified
    if save_pngs_dir:
        png_dir = Path(save_pngs_dir)
        png_dir.mkdir(parents=True, exist_ok=True)

    fits_path = Path(report.fits_path)
    if not fits_path.exists():
        return figures

    try:
        from dsa110_contimg.common.utils.fits_utils import get_2d_data_and_wcs

        data, wcs, _ = get_2d_data_and_wcs(str(fits_path))

        # Convert RA/Dec to pixel
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        coord = SkyCoord(report.ra_deg * u.deg, report.dec_deg * u.deg, frame="icrs")
        px, py = wcs.world_to_pixel(coord)
        px, py = int(px), int(py)

        # Figure 1: Cutout around source
        cutout_size = max(report.box_size * 3, 50)
        x_min = max(0, px - cutout_size)
        x_max = min(data.shape[1], px + cutout_size)
        y_min = max(0, py - cutout_size)
        y_max = min(data.shape[0], py + cutout_size)

        cutout = data[y_min:y_max, x_min:x_max]

        fig, ax = plt.subplots(figsize=(8, 8))
        vmax = report.peak_flux_jy * 1.2 if report.peak_flux_jy > 0 else None
        vmin = -3 * report.local_rms_jy if report.local_rms_jy > 0 else None
        im = ax.imshow(
            cutout * 1000,
            origin="lower",
            cmap="RdBu_r",
            vmin=vmin * 1000 if vmin else None,
            vmax=vmax * 1000 if vmax else None,
        )
        plt.colorbar(im, ax=ax, label="Flux (mJy/beam)")

        # Mark source position
        rel_x = px - x_min
        rel_y = py - y_min
        ax.scatter([rel_x], [rel_y], marker="+", s=200, color="lime", linewidths=2)

        # Draw measurement box
        box_half = report.box_size // 2
        rect = plt.Rectangle(
            (rel_x - box_half, rel_y - box_half),
            report.box_size,
            report.box_size,
            fill=False,
            edgecolor="yellow",
            linestyle="--",
            linewidth=2,
        )
        ax.add_patch(rect)

        # Draw annulus if specified
        if report.annulus_radii and len(report.annulus_radii) >= 2:
            inner_ann = Circle(
                (rel_x, rel_y),
                report.annulus_radii[0],
                fill=False,
                edgecolor="cyan",
                linestyle=":",
                linewidth=1.5,
            )
            outer_ann = Circle(
                (rel_x, rel_y),
                report.annulus_radii[1],
                fill=False,
                edgecolor="cyan",
                linestyle=":",
                linewidth=1.5,
            )
            ax.add_patch(inner_ann)
            ax.add_patch(outer_ann)

        ax.set_xlabel("Pixel X (relative)")
        ax.set_ylabel("Pixel Y (relative)")
        ax.set_title(f"Source Cutout: {report.source_name}")
        fig.tight_layout()

        # Save to file if directory specified
        if save_pngs_dir:
            png_path = png_dir / f"{image_prefix}_cutout.png"
            fig.savefig(png_path, format="png", dpi=120, bbox_inches="tight")
            logger.info("Saved PNG: %s", png_path)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        figures["cutout"] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        # Figure 2: SNR gauge (if available)
        if report.snr > 0:
            fig, ax = plt.subplots(figsize=(6, 4))

            # Create horizontal bar
            categories = ["Detection\nThreshold (5σ)", "Good (20σ)", "Excellent (100σ)"]
            thresholds = [5, 20, 100]
            colors = ["#e74c3c", "#f39c12", "#27ae60"]

            ax.barh([0], [report.snr], height=0.4, color="#3498db", alpha=0.8)
            for i, (thresh, color) in enumerate(zip(thresholds, colors)):
                ax.axvline(
                    thresh, color=color, linestyle="--", linewidth=2, label=f"{categories[i]}"
                )

            ax.set_xlim(0, max(report.snr * 1.2, 120))
            ax.set_yticks([])
            ax.set_xlabel("Signal-to-Noise Ratio (SNR)")
            ax.set_title(f"Detection Quality: SNR = {report.snr:.1f}")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3, axis="x")
            fig.tight_layout()

            # Save to file if directory specified
            if save_pngs_dir:
                png_path = png_dir / f"{image_prefix}_snr_gauge.png"
                fig.savefig(png_path, format="png", dpi=120, bbox_inches="tight")
                logger.info("Saved PNG: %s", png_path)

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            figures["snr_gauge"] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)

        # Figure 3: Flux comparison if expected value provided
        if report.expected_flux_jy is not None and report.expected_flux_jy > 0:
            fig, ax = plt.subplots(figsize=(6, 5))

            labels = ["Expected", "Measured"]
            values = [report.expected_flux_jy * 1000, report.peak_flux_jy * 1000]
            bar_colors = [
                "#3498db",
                "#2ecc71" if abs(report.flux_deviation_percent or 0) < 20 else "#e74c3c",
            ]

            ax.bar(labels, values, color=bar_colors, edgecolor="black")

            # Add error bar on measured
            if report.local_rms_jy > 0:
                ax.errorbar(
                    "Measured",
                    report.peak_flux_jy * 1000,
                    yerr=report.local_rms_jy * 1000,
                    fmt="none",
                    color="black",
                    capsize=5,
                    capthick=2,
                )

            ax.set_ylabel("Peak Flux (mJy/beam)")
            ax.set_title(f"Flux Comparison ({report.flux_deviation_percent:+.1f}% deviation)")
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()

            # Save to file if directory specified
            if save_pngs_dir:
                png_path = png_dir / f"{image_prefix}_flux_comparison.png"
                fig.savefig(png_path, format="png", dpi=120, bbox_inches="tight")
                logger.info("Saved PNG: %s", png_path)

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            figures["flux_comparison"] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)

    except Exception as err:
        logger.warning("Failed to generate photometry figures: %s", err)

    return figures


def generate_html_report(
    report: PhotometryReportData,
    output_path: str | None = None,
    embed_figures: bool = True,
    save_pngs_dir: str | None = None,
    image_prefix: str = "photometry",
) -> str:
    """
    Generate HTML report from PhotometryReportData.

    Parameters
    ----------
    report : PhotometryReportData
        The report data container
    output_path : Optional[str]
        Path to save the HTML report. If None, returns HTML string only.
    embed_figures : bool
        Whether to generate and embed figures (default True)
    save_pngs_dir : Optional[str]
        If provided, save PNG files to this directory
    image_prefix : str
        Prefix for saved PNG filenames (default: "photometry")

    Returns
    -------
    str
        HTML content of the report
    """
    if embed_figures:
        report.figure_base64 = generate_photometry_figures(
            report, save_pngs_dir=save_pngs_dir, image_prefix=image_prefix
        )

    html = _generate_html_content(report)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html, encoding="utf-8")
        logger.info("Photometry diagnostics report saved to: %s", output_path)

    return html


def _generate_html_content(report: PhotometryReportData) -> str:
    """Generate the HTML content for the report."""
    grade_colors = {
        "Excellent": "var(--dsa-success)",
        "Good": "var(--dsa-success)",
        "Acceptable": "var(--dsa-warning)",
        "Marginal": "var(--dsa-warning)",
        "Unknown": "var(--dsa-text-muted)",
        "unknown": "var(--dsa-text-muted)",
    }
    grade_color = grade_colors.get(report.quality_grade, "var(--dsa-text-muted)")

    shared_css = get_shared_css()

    # Build recommendations list
    rec_items = "".join(f"<li>{r}</li>" for r in report.recommendations)
    recommendations_html = f"<ul>{rec_items}</ul>" if rec_items else "<p>Photometry looks good!</p>"

    # Build figure sections
    figures_html = ""
    if "cutout" in report.figure_base64:
        figures_html += f"""
        <div class="figure-container">
            <h3>Source Cutout</h3>
            <img src="data:image/png;base64,{report.figure_base64["cutout"]}" alt="Source Cutout">
        </div>
        """
    if "snr_gauge" in report.figure_base64:
        figures_html += f"""
        <div class="figure-container">
            <h3>Detection Quality</h3>
            <img src="data:image/png;base64,{report.figure_base64["snr_gauge"]}" alt="SNR Gauge">
        </div>
        """
    if "flux_comparison" in report.figure_base64:
        figures_html += f"""
        <div class="figure-container">
            <h3>Flux Comparison</h3>
            <img src="data:image/png;base64,{report.figure_base64["flux_comparison"]}" alt="Flux Comparison">
        </div>
        """

    # Format comparison row if expected flux provided
    comparison_html = ""
    if report.expected_flux_jy is not None:
        deviation_class = "good" if abs(report.flux_deviation_percent or 0) < 20 else "warning"
        comparison_html = f"""
        <div class="card">
            <h2>Flux Comparison</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Expected Flux</td><td>{report.expected_flux_jy * 1000:.2f} mJy</td></tr>
                <tr><td>Measured Flux</td><td>{report.peak_flux_jy * 1000:.2f} mJy</td></tr>
                <tr><td>Flux Ratio</td><td>{report.flux_ratio:.3f}</td></tr>
                <tr class="{deviation_class}"><td>Deviation</td><td>{report.flux_deviation_percent:+.1f}%</td></tr>
            </table>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Photometry Diagnostics Report</title>
    <style>
        {shared_css}

        body {{
            background: var(--dsa-bg-paper);
            max-width: 1200px;
            margin: 0 auto;
            padding: var(--dsa-spacing-lg);
        }}
        .header {{
            background: linear-gradient(135deg, var(--dsa-primary) 0%, var(--dsa-secondary) 100%);
            color: white;
            padding: var(--dsa-spacing-lg);
            border-radius: var(--dsa-radius-lg);
            margin-bottom: var(--dsa-spacing-lg);
            box-shadow: var(--dsa-shadow);
        }}
        .header h1 {{ margin: 0 0 10px 0; }}
        .header .meta {{ opacity: 0.9; font-size: 0.9em; }}
        .card {{
            background: var(--dsa-bg-default);
            border-radius: var(--dsa-radius-lg);
            padding: var(--dsa-spacing-lg);
            margin-bottom: var(--dsa-spacing-lg);
            border: 1px solid var(--dsa-border);
            box-shadow: var(--dsa-shadow);
        }}
        .card h2 {{
            margin-top: 0;
            color: var(--dsa-primary);
            border-bottom: 2px solid var(--dsa-secondary);
            padding-bottom: var(--dsa-spacing-sm);
        }}
        .quality-badge {{
            display: inline-block;
            padding: var(--dsa-spacing-sm) var(--dsa-spacing-md);
            border-radius: var(--dsa-radius);
            font-weight: bold;
            font-size: 1.2em;
            color: white;
            background: {grade_color};
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: var(--dsa-spacing-md);
            margin: var(--dsa-spacing-lg) 0;
        }}
        .stat-box {{
            background: var(--dsa-bg-paper);
            padding: var(--dsa-spacing-md);
            border-radius: var(--dsa-radius);
            text-align: center;
        }}
        .stat-box .value {{
            font-size: 1.3em;
            font-weight: bold;
            color: var(--dsa-primary);
        }}
        .stat-box .label {{
            font-size: 0.85em;
            color: var(--dsa-text-muted);
        }}
        .figure-container {{
            margin: var(--dsa-spacing-lg) 0;
            text-align: center;
        }}
        .figure-container img {{
            max-width: 100%;
            border-radius: var(--dsa-radius);
            border: 1px solid var(--dsa-border);
            box-shadow: var(--dsa-shadow);
        }}
        .recommendations {{
            background: var(--dsa-warning-bg);
            border-left: 4px solid var(--dsa-warning);
            padding: var(--dsa-spacing-md);
            border-radius: 0 var(--dsa-radius) var(--dsa-radius) 0;
        }}
        .recommendations.success {{
            background: var(--dsa-success-bg);
            border-left-color: var(--dsa-success);
        }}
        .recommendations ul {{ margin: var(--dsa-spacing-sm) 0; padding-left: 20px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: var(--dsa-spacing-md) 0;
        }}
        th, td {{
            padding: var(--dsa-spacing-sm) var(--dsa-spacing-md);
            text-align: left;
            border-bottom: 1px solid var(--dsa-border);
        }}
        th {{ background: var(--dsa-secondary); color: white; }}
        tr.good {{ background: var(--dsa-success-bg); }}
        tr.warning {{ background: var(--dsa-warning-bg); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Photometry Diagnostics Report</h1>
        <div class="meta">
            <strong>Source:</strong> {report.source_name}<br>
            <strong>Image:</strong> {report.fits_path}<br>
            <strong>Position:</strong> RA={report.ra_deg:.5f}°, Dec={report.dec_deg:.5f}°<br>
            <strong>Generated:</strong> {report.timestamp}
        </div>
    </div>

    <div class="card">
        <h2>Detection Quality</h2>
        <div class="quality-badge">{report.quality_grade}</div>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="value">{report.peak_flux_jy * 1000:.3f} mJy</div>
                <div class="label">Peak Flux</div>
            </div>
            <div class="stat-box">
                <div class="value">{report.local_rms_jy * 1e6:.1f} µJy</div>
                <div class="label">Local RMS</div>
            </div>
            <div class="stat-box">
                <div class="value">{report.snr:.1f}</div>
                <div class="label">SNR</div>
            </div>
            <div class="stat-box">
                <div class="value">{report.method}</div>
                <div class="label">Method</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Measurement Parameters</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Box Size</td><td>{report.box_size} pixels</td></tr>
            <tr><td>Annulus Radii</td><td>{report.annulus_radii}</td></tr>
            <tr><td>RA (deg)</td><td>{report.ra_deg:.6f}</td></tr>
            <tr><td>Dec (deg)</td><td>{report.dec_deg:.6f}</td></tr>
        </table>
    </div>

    {comparison_html}

    <div class="card">
        <h2>Recommendations</h2>
        <div class="recommendations{" success" if not report.recommendations else ""}">
            {recommendations_html}
        </div>
    </div>

    <div class="card">
        <h2>Diagnostic Figures</h2>
        {figures_html if figures_html else "<p>No figures generated.</p>"}
    </div>
</body>
</html>
"""
    return html


def generate_photometry_report(
    fits_path: str,
    ra_deg: float,
    dec_deg: float,
    output_dir: str,
    measurement_result: dict[str, Any] | None = None,
    source_name: str = "",
    expected_flux_jy: float | None = None,
    box_size: int = 15,
    annulus_radii: list[int] | None = None,
    save_pngs: bool = True,
) -> str:
    """
    High-level function to generate complete photometry diagnostics report.

    This is the main entry point for the pipeline integration.

    Parameters
    ----------
    fits_path : str
        Path to the FITS image
    ra_deg : float
        Right ascension in degrees
    dec_deg : float
        Declination in degrees
    output_dir : str
        Directory to save the report
    measurement_result : Optional[Dict[str, Any]]
        Result from measure_forced_peak (if already measured)
    source_name : str
        Name of the source
    expected_flux_jy : Optional[float]
        Expected flux for comparison
    box_size : int
        Box size for measurement
    annulus_radii : Optional[List[int]]
        Annulus radii for background
    save_pngs : bool
        Whether to save PNG files alongside the HTML report (default True)

    Returns
    -------
    str
        Path to the generated HTML report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading photometry diagnostics data")
    report_data = load_photometry_report_data(
        fits_path=fits_path,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        measurement_result=measurement_result,
        source_name=source_name,
        expected_flux_jy=expected_flux_jy,
        box_size=box_size,
        annulus_radii=annulus_radii,
    )

    # Generate report filename
    safe_name = (
        source_name.replace(" ", "_").replace("/", "_")
        if source_name
        else f"ra{ra_deg:.2f}_dec{dec_deg:.2f}"
    )
    report_filename = f"photometry_diagnostics_{safe_name}.html"
    report_path = output_dir / report_filename

    # Determine PNG save directory
    save_pngs_dir = str(output_dir) if save_pngs else None

    # Generate HTML report (and optionally save PNGs)
    logger.info("Generating photometry diagnostics report: %s", report_path)
    generate_html_report(
        report_data,
        str(report_path),
        embed_figures=True,
        save_pngs_dir=save_pngs_dir,
        image_prefix=f"photometry_{safe_name}",
    )

    return str(report_path)


def _generate_source_cutouts(
    fits_image: str | Path,
    measurements: list[dict[str, Any]],
    box_size_pix: int,
    annulus_inner_pix: int,
    annulus_outer_pix: int,
    n_cutouts: int = 5,
    snr_detection_threshold: float = 3.0,
) -> list[dict[str, Any]]:
    """Generate cutout images for the top N detected sources."""
    import base64
    import io

    import numpy as np

    cutouts = []

    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import astropy.units as u
        import matplotlib.pyplot as plt
        from astropy.coordinates import SkyCoord
        from astropy.io import fits
        from astropy.nddata import Cutout2D
        from astropy.visualization import ImageNormalize, ZScaleInterval
        from astropy.wcs import WCS
        from matplotlib.patches import Circle, Ellipse, Rectangle
    except ImportError as exc:
        logger.warning("Cannot generate cutouts: %s", exc)
        return []

    # Sort by SNR and get top N detected sources
    detected = [m for m in measurements if m.get("snr", 0.0) >= snr_detection_threshold]
    top_sources = sorted(detected, key=lambda x: x.get("snr", 0.0), reverse=True)[:n_cutouts]

    if not top_sources:
        logger.info("No detected sources for cutouts")
        return []

    # Load FITS data once
    fits_path = Path(fits_image)
    try:
        with fits.open(fits_path) as hdul:
            data = np.squeeze(hdul[0].data)
            wcs = WCS(hdul[0].header).celestial
            header = hdul[0].header
    except Exception as exc:
        logger.warning("Failed to load FITS for cutouts: %s", exc)
        return []

    # Get pixel scale and beam info
    cdelt = abs(header.get("CDELT1", header.get("CD1_1", 1 / 3600))) * 3600  # arcsec/pix
    bmaj = header.get("BMAJ", 0) * 3600  # arcsec
    bmin = header.get("BMIN", 0) * 3600
    bpa = header.get("BPA", 0)

    # Get photometry aperture/annulus sizes in arcsec
    box_size_arcsec = box_size_pix * cdelt
    annulus_inner_arcsec = annulus_inner_pix * cdelt
    annulus_outer_arcsec = annulus_outer_pix * cdelt

    # Calculate cutout size to show full annulus + margin (1.3x outer annulus)
    cutout_size_arcsec = annulus_outer_arcsec * 2.6
    cutout_size_arcmin = cutout_size_arcsec / 60.0

    for idx, src in enumerate(top_sources, 1):
        try:
            coord = SkyCoord(ra=src["ra_deg"], dec=src["dec_deg"], unit="deg")

            # Create cutout - sized to show full annulus
            size = cutout_size_arcmin * u.arcmin
            cutout = Cutout2D(data, position=coord, size=size, wcs=wcs)

            # Create figure
            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": cutout.wcs})

            # Normalize with zscale
            norm = ImageNormalize(cutout.data, interval=ZScaleInterval())

            # Display
            im = ax.imshow(cutout.data, origin="lower", cmap="viridis", norm=norm)

            # Add crosshair at source position
            ax.scatter(
                coord.ra.deg,
                coord.dec.deg,
                marker="+",
                c="red",
                s=200,
                linewidths=2.5,
                transform=ax.get_transform("fk5"),
                zorder=10,
            )

            # Add measurement aperture box (cyan)
            box_half = box_size_arcsec / 3600 / 2  # half-size in degrees
            aperture_rect = Rectangle(
                (coord.ra.deg - box_half, coord.dec.deg - box_half),
                box_half * 2,
                box_half * 2,
                edgecolor="cyan",
                facecolor="none",
                linewidth=2,
                linestyle="-",
                transform=ax.get_transform("fk5"),
                zorder=5,
                label=f"Aperture ({box_size_pix}px)",
            )
            ax.add_patch(aperture_rect)

            # Add inner annulus circle (yellow dashed)
            inner_annulus = Circle(
                (coord.ra.deg, coord.dec.deg),
                radius=annulus_inner_arcsec / 3600,
                edgecolor="yellow",
                facecolor="none",
                linewidth=1.5,
                linestyle="--",
                transform=ax.get_transform("fk5"),
                zorder=4,
                label=f"Inner ({annulus_inner_pix}px)",
            )
            ax.add_patch(inner_annulus)

            # Add outer annulus circle (orange dashed)
            outer_annulus = Circle(
                (coord.ra.deg, coord.dec.deg),
                radius=annulus_outer_arcsec / 3600,
                edgecolor="orange",
                facecolor="none",
                linewidth=1.5,
                linestyle="--",
                transform=ax.get_transform("fk5"),
                zorder=4,
                label=f"Outer ({annulus_outer_pix}px)",
            )
            ax.add_patch(outer_annulus)

            # Add beam ellipse in corner (white)
            beam_ellipse = Ellipse(
                (coord.ra.deg, coord.dec.deg),
                width=bmaj / 3600,
                height=bmin / 3600,
                angle=bpa,
                edgecolor="white",
                facecolor="none",
                linewidth=1.5,
                linestyle=":",
                transform=ax.get_transform("fk5"),
                zorder=3,
            )
            ax.add_patch(beam_ellipse)

            # Labels
            ax.coords[0].set_major_formatter("hh:mm:ss")
            ax.coords[1].set_major_formatter("dd:mm:ss")
            ax.set_xlabel("RA (J2000)")
            ax.set_ylabel("Dec (J2000)")

            # Title with source info
            ax.set_title(
                f"Source #{idx}: SNR={src['snr']:.1f}\n"
                f"Cat: {src['catalog_flux_mjy']:.1f} mJy | Meas: {src['measured_flux_mjy']:.3f} mJy",
                fontsize=9,
            )

            # Legend for apertures
            ax.legend(
                loc="upper right",
                fontsize=7,
                framealpha=0.9,
                facecolor="white",
                edgecolor="#ccc",
                labelcolor="#333",
            )

            # Colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label("Jy/beam", fontsize=8, color="#333")
            cbar.ax.tick_params(colors="#333")

            # Set tick label colors to dark for visibility on light background
            ax.coords[0].set_ticklabel(color="#333")
            ax.coords[1].set_ticklabel(color="#333")
            ax.set_xlabel("RA (J2000)", color="#333")
            ax.set_ylabel("Dec (J2000)", color="#333")
            ax.title.set_color("#333")

            plt.tight_layout()

            # Save to base64
            buf = io.BytesIO()
            fig.savefig(
                buf,
                format="png",
                dpi=120,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            buf.seek(0)
            b64_img = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)

            cutouts.append(
                {
                    "rank": idx,
                    "ra_deg": src["ra_deg"],
                    "dec_deg": src["dec_deg"],
                    "snr": src["snr"],
                    "catalog_flux_mjy": src["catalog_flux_mjy"],
                    "measured_flux_mjy": src["measured_flux_mjy"],
                    "image_b64": b64_img,
                }
            )
        except Exception as exc:
            logger.warning("Cutout %s failed: %s", idx, exc)
            continue

    return cutouts


def _generate_field_overview(
    fits_image: str | Path,
    measurements: list[dict[str, Any]],
    field_center: tuple[float, float],
    box_size_pix: int,
    annulus_inner_pix: int,
    annulus_outer_pix: int,
    snr_detection_threshold: float = 3.0,
    snr_strong_threshold: float = 5.0,
) -> str | None:
    """Generate a full-field overview image showing all sources with apertures."""
    import base64
    import io

    import numpy as np

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from astropy.io import fits
        from astropy.visualization import ImageNormalize, ZScaleInterval
        from astropy.wcs import WCS
        from matplotlib.patches import Circle
    except ImportError as exc:
        logger.warning("Cannot generate field overview: %s", exc)
        return None

    fits_path = Path(fits_image)
    try:
        with fits.open(fits_path) as hdul:
            data = np.squeeze(hdul[0].data)
            wcs = WCS(hdul[0].header).celestial
            header = hdul[0].header
    except Exception as exc:
        logger.warning("Failed to load FITS for overview: %s", exc)
        return None

    # Get pixel scale
    cdelt = abs(header.get("CDELT1", header.get("CD1_1", 1 / 3600))) * 3600  # arcsec/pix

    # Annulus size in degrees for plotting
    annulus_inner_deg = (annulus_inner_pix * cdelt) / 3600

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": wcs})

    # Normalize with zscale
    norm = ImageNormalize(data, interval=ZScaleInterval())

    # Display full field
    im = ax.imshow(data, origin="lower", cmap="viridis", norm=norm)

    # Plot all sources with color-coded markers based on SNR
    for measurement in measurements:
        ra = measurement["ra_deg"]
        dec = measurement["dec_deg"]
        snr = measurement["snr"]

        # Color based on detection status
        if snr >= snr_strong_threshold:
            color = "#2ecc71"  # Green - strong detection
            marker_size = 100
        elif snr >= snr_detection_threshold:
            color = "#f39c12"  # Orange - detected
            marker_size = 80
        else:
            color = "#e74c3c"  # Red - non-detection
            marker_size = 60

        # Plot source marker
        ax.scatter(
            ra,
            dec,
            marker="o",
            s=marker_size,
            facecolor="none",
            edgecolor=color,
            linewidth=2,
            alpha=0.9,
            transform=ax.get_transform("fk5"),
            zorder=5,
        )

        # Add annulus circles for detected sources (inner only to reduce clutter)
        if snr >= snr_detection_threshold:
            inner_circle = Circle(
                (ra, dec),
                radius=annulus_inner_deg,
                edgecolor=color,
                facecolor="none",
                linewidth=0.5,
                linestyle="--",
                transform=ax.get_transform("fk5"),
                alpha=0.5,
                zorder=3,
            )
            ax.add_patch(inner_circle)

    # Mark field center
    ax.scatter(
        field_center[0],
        field_center[1],
        marker="x",
        s=200,
        c="white",
        linewidths=3,
        transform=ax.get_transform("fk5"),
        zorder=10,
    )

    # Labels
    ax.coords[0].set_major_formatter("hh:mm:ss")
    ax.coords[1].set_major_formatter("dd:mm:ss")
    ax.set_xlabel("RA (J2000)", fontsize=12)
    ax.set_ylabel("Dec (J2000)", fontsize=12)
    ax.set_title(
        f"Full Field Overview: {len(measurements)} Catalog Sources\n"
        f"Green=Strong (SNR≥{snr_strong_threshold:g}) | "
        f"Orange=Detected (SNR≥{snr_detection_threshold:g}) | Red=Non-detection",
        fontsize=11,
    )

    # Add info text
    info_text = (
        f"Aperture: {box_size_pix}px box\nAnnulus: {annulus_inner_pix}-{annulus_outer_pix}px"
    )
    ax.text(
        0.02,
        0.02,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        color="#333",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#ccc"),
    )

    # Set tick label colors to dark for visibility on light background
    ax.coords[0].set_ticklabel(color="#333")
    ax.coords[1].set_ticklabel(color="#333")
    ax.xaxis.label.set_color("#333")
    ax.yaxis.label.set_color("#333")
    ax.title.set_color("#333")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Jy/beam", fontsize=10, color="#333")
    cbar.ax.tick_params(colors="#333")

    plt.tight_layout()

    # Save to base64
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=100,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    buf.seek(0)
    b64_img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return b64_img


def generate_field_photometry_report(
    fits_image: str | Path,
    measurements: list[dict[str, Any]],
    catalog_sources: list[dict[str, Any]] | None,
    calibrator_info: dict[str, Any] | None,
    calibrator_result: dict[str, Any] | None,
    field_center: tuple[float, float],
    stats: dict[str, Any],
    output_dir: str | Path,
    catalog: str,
    catalog_radius_deg: float,
    min_flux_mjy: float,
    box_size_pix: int,
    annulus_inner_pix: int,
    annulus_outer_pix: int,
    snr_detection_threshold: float = 3.0,
    snr_strong_threshold: float = 5.0,
) -> str:
    """Generate comprehensive HTML report for field photometry."""
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate report filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"field_photometry_report_{timestamp}.html"
    report_path = output_dir / report_name

    # Generate cutouts for top sources
    logger.info("Generating cutouts for top sources...")
    cutouts = _generate_source_cutouts(
        fits_image,
        measurements,
        box_size_pix=box_size_pix,
        annulus_inner_pix=annulus_inner_pix,
        annulus_outer_pix=annulus_outer_pix,
        n_cutouts=5,
        snr_detection_threshold=snr_detection_threshold,
    )
    logger.info("Generated %s cutouts", len(cutouts))

    # Generate full-field overview
    logger.info("Generating full-field overview...")
    field_overview_b64 = _generate_field_overview(
        fits_image,
        measurements,
        field_center,
        box_size_pix=box_size_pix,
        annulus_inner_pix=annulus_inner_pix,
        annulus_outer_pix=annulus_outer_pix,
        snr_detection_threshold=snr_detection_threshold,
        snr_strong_threshold=snr_strong_threshold,
    )

    # Build field overview HTML section
    field_overview_html = ""
    if field_overview_b64:
        field_overview_html = f"""
        <div class="card">
            <h2> Full Field Overview</h2>
            <p style="color: #888; margin-bottom: 15px;">
                All catalog sources marked with detection status. White X marks field center.
                Dashed circles show inner annulus (noise measurement region:
                {annulus_inner_pix}-{annulus_outer_pix} pixels).
            </p>
            <div style="text-align: center;">
                <img src="data:image/png;base64,{field_overview_b64}" alt="Field Overview"
                     style="max-width: 100%; border-radius: 10px; border: 1px solid #333;">
            </div>
        </div>
        """

    # Build cutouts HTML section
    cutouts_html = ""
    if cutouts:
        cutout_cards = []
        for cutout in cutouts:
            cutout_cards.append(
                f"""
            <div class="cutout-card">
                <img src="data:image/png;base64,{cutout["image_b64"]}" alt="Source {cutout["rank"]}">
                <div class="cutout-info">
                    <strong>#{cutout["rank"]}</strong> SNR: {cutout["snr"]:.1f}<br>
                    RA: {cutout["ra_deg"]:.4f}°<br>
                    Dec: {cutout["dec_deg"]:.4f}°<br>
                    Cat: {cutout["catalog_flux_mjy"]:.1f} mJy<br>
                    Meas: {cutout["measured_flux_mjy"]:.3f} mJy
                </div>
            </div>
            """
            )

        cutouts_html = f"""
        <div class="card">
            <h2> Top 5 Source Cutouts (by SNR)</h2>
            <p style="color: #888; margin-bottom: 15px;">
                Cutouts showing measurement aperture (cyan box), noise annulus (yellow/orange circles),
                and beam (white dotted ellipse). Red cross marks catalog position.
            </p>
            <div class="cutouts-grid">
                {"".join(cutout_cards)}
            </div>
        </div>
        """

    # Build source table rows
    source_rows = []
    for measurement in measurements:
        if np.isnan(measurement["measured_flux_mjy"]):
            status = "✗ Failed"
            status_class = "status-failed"
        elif measurement["snr"] < snr_detection_threshold:
            status = "⚠ Low SNR"
            status_class = "status-warn"
        elif abs(measurement["flux_ratio"] - 1.0) < 0.3:
            status = "✓ Good"
            status_class = "status-good"
        else:
            status = "⚠ Offset"
            status_class = "status-warn"

        source_rows.append(
            f"""
        <tr class="{status_class}">
            <td>{measurement["source_idx"]}</td>
            <td>{measurement["ra_deg"]:.5f}</td>
            <td>{measurement["dec_deg"]:.5f}</td>
            <td>{measurement["catalog_flux_mjy"]:.2f}</td>
            <td>{measurement["measured_flux_mjy"]:.2f}</td>
            <td>{measurement["snr"]:.1f}</td>
            <td>{measurement["flux_ratio"]:.3f}</td>
            <td>{status}</td>
        </tr>
        """
        )

    # Calibrator section
    cal_html = ""
    if calibrator_result and calibrator_info:
        cal_ratio = calibrator_result["measured_flux_jy"] / calibrator_result["expected_flux_jy"]
        cal_html = f"""
        <div class="card calibrator-card">
            <h2> Calibrator Source: {calibrator_info["name"]}</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="value">{calibrator_result["measured_flux_jy"] * 1000:.2f} mJy</div>
                    <div class="label">Measured Flux</div>
                </div>
                <div class="stat-box">
                    <div class="value">{calibrator_result["expected_flux_jy"] * 1000:.2f} mJy</div>
                    <div class="label">Expected Flux</div>
                </div>
                <div class="stat-box">
                    <div class="value">{calibrator_result["snr"]:.1f}</div>
                    <div class="label">SNR</div>
                </div>
                <div class="stat-box">
                    <div class="value">{cal_ratio:.3f}</div>
                    <div class="label">Flux Ratio</div>
                </div>
            </div>
        </div>
        """

    # Statistics summary
    n_total = stats.get("n_total")
    if n_total is None and catalog_sources is not None:
        n_total = len(catalog_sources)
    if n_total is None:
        n_total = len(measurements)

    stats_html = f"""
    <div class="card">
        <h2> Field Photometry Statistics</h2>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="value">{stats.get("n_valid", 0)}/{n_total}</div>
                <div class="label">Valid/Total Sources</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats.get("mean_ratio", 0):.3f}</div>
                <div class="label">Mean Flux Ratio</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats.get("median_ratio", 0):.3f}</div>
                <div class="label">Median Flux Ratio</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats.get("scale_factor", 0):.3f}</div>
                <div class="label">Scale Factor</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats.get("std_ratio", 0):.3f}</div>
                <div class="label">Std Deviation</div>
            </div>
        </div>
    </div>
    """

    # Full HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DSA-110 Field Photometry Report</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            color: #333;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            margin-bottom: 30px;
            background: linear-gradient(90deg, #0066cc, #5b4dc9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
        }}
        .card {{
            background: #fff;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .card h2 {{ margin-bottom: 20px; color: #0066cc; }}
        .calibrator-card {{ border: 2px solid #5b4dc9; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .stat-box {{
            background: #e8f4fc;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        .stat-box .value {{
            font-size: 1.8rem;
            font-weight: bold;
            color: #0066cc;
        }}
        .stat-box .label {{
            font-size: 0.85rem;
            color: #666;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #e8f4fc;
            font-weight: 600;
            color: #333;
        }}
        tr:hover {{ background: #f5f7fa; }}
        .status-good {{ color: #27ae60; }}
        .status-warn {{ color: #e67e22; }}
        .status-failed {{ color: #c0392b; }}
        .meta {{
            text-align: center;
            color: #888;
            font-size: 0.9rem;
            margin-top: 30px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }}
        .info-item {{ padding: 8px; }}
        .info-item strong {{ color: #0066cc; }}
        .cutouts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }}
        .cutout-card {{
            background: #f5f7fa;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            border: 1px solid #ddd;
        }}
        .cutout-card img {{
            max-width: 100%;
            border-radius: 5px;
            border: 1px solid #ccc;
        }}
        .cutout-info {{
            margin-top: 10px;
            font-size: 0.85rem;
            color: #555;
        }}
        .cutout-info strong {{
            color: #0066cc;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1> DSA-110 Field Photometry Report</h1>

        <div class="card">
            <h2> Observation Details</h2>
            <div class="info-grid">
                <div class="info-item"><strong>FITS Image:</strong> {Path(fits_image).name}</div>
                <div class="info-item"><strong>Catalog:</strong> {catalog.upper()}</div>
                <div class="info-item"><strong>Field Center:</strong>
                    RA={field_center[0]:.5f}°, Dec={field_center[1]:.5f}°</div>
                <div class="info-item"><strong>Search Radius:</strong> {catalog_radius_deg}°</div>
                <div class="info-item"><strong>Min Flux:</strong> {min_flux_mjy} mJy</div>
                <div class="info-item"><strong>Generated:</strong>
                    {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
            </div>
        </div>

        {cal_html}

        {stats_html}

        {field_overview_html}

        {cutouts_html}

        <div class="card">
            <h2> Source Measurements ({len(measurements)} sources)</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>RA (deg)</th>
                        <th>Dec (deg)</th>
                        <th>Cat Flux (mJy)</th>
                        <th>Meas Flux (mJy)</th>
                        <th>SNR</th>
                        <th>Ratio</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(source_rows)}
                </tbody>
            </table>
        </div>

        <div class="meta">
            Generated by DSA-110 Continuum Imaging Pipeline<br>
            Catalog: Unified Master Catalog (NVSS + FIRST + VLASS + RAX)
        </div>
    </div>
</body>
</html>
    """

    report_path.write_text(html, encoding="utf-8")
    return str(report_path)
