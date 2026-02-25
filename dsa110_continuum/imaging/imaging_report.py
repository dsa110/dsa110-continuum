"""
Imaging Diagnostics HTML Report Generator.

Generates comprehensive HTML reports for imaging quality assessment,
including diagnostic figures, statistics, and recommendations.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from dsa110_contimg.common.utils.template_styles import get_shared_css

logger = logging.getLogger(__name__)


@dataclass
class ImagingReportData:
    """Container for all data needed to generate the imaging report."""

    # Identification
    ms_path: str
    image_path: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Imaging parameters
    imsize: int = 0
    cell_arcsec: float = 0.0
    weighting: str = "briggs"
    robust: float = 0.0
    niter: int = 0
    threshold: str = ""
    gridder: str = "standard"
    quality_tier: str = "standard"

    # Image statistics
    peak_flux_jy: float = 0.0
    rms_jy: float = 0.0
    dynamic_range: float = 0.0
    mean_jy: float = 0.0
    min_jy: float = 0.0
    max_jy: float = 0.0

    # Beam information
    beam_major_arcsec: float = 0.0
    beam_minor_arcsec: float = 0.0
    beam_pa_deg: float = 0.0

    # Quality assessment
    quality_grade: str = "unknown"
    recommendations: list[str] = field(default_factory=list)

    # Output files
    fits_files: list[str] = field(default_factory=list)

    # Figure base64 encoded images
    figure_base64: dict[str, str] = field(default_factory=dict)


def load_imaging_report_data(
    ms_path: str,
    image_path: str,
    imaging_params: dict[str, Any] | None = None,
) -> ImagingReportData:
    """
    Load all data needed to generate the imaging diagnostics report.

    Parameters
    ----------
    ms_path : str
        Path to the measurement set
    image_path : str
        Path to the output image (FITS or CASA image prefix)
    imaging_params : Optional[Dict[str, Any]]
        Dictionary of imaging parameters used

    Returns
    -------
    ImagingReportData
        Data container for report generation
    """
    from dsa110_contimg.common.utils.fits_utils import get_2d_data_and_wcs

    report = ImagingReportData(
        ms_path=ms_path,
        image_path=image_path,
    )

    # Store imaging parameters if provided
    if imaging_params:
        report.imsize = imaging_params.get("imsize", 0)
        report.cell_arcsec = imaging_params.get("cell_arcsec", 0.0)
        report.weighting = imaging_params.get("weighting", "briggs")
        report.robust = imaging_params.get("robust", 0.0)
        report.niter = imaging_params.get("niter", 0)
        report.threshold = imaging_params.get("threshold", "")
        report.gridder = imaging_params.get("gridder", "standard")
        report.quality_tier = imaging_params.get("quality_tier", "standard")

    # Find FITS files
    image_dir = Path(image_path).parent
    image_stem = Path(image_path).stem
    fits_files = list(image_dir.glob(f"{image_stem}*.fits"))
    report.fits_files = [str(f) for f in fits_files]

    # Load primary image statistics
    primary_fits = None
    for suffix in [".image.fits", ".fits", ".image.pbcor.fits"]:
        candidate = Path(f"{image_path}{suffix}")
        if candidate.exists():
            primary_fits = str(candidate)
            break
    if not primary_fits and fits_files:
        primary_fits = str(fits_files[0])

    if primary_fits and Path(primary_fits).exists():
        try:
            data, _, header_info = get_2d_data_and_wcs(primary_fits)

            # Basic statistics
            valid_data = data[np.isfinite(data)]
            if len(valid_data) > 0:
                report.peak_flux_jy = float(np.nanmax(valid_data))
                report.min_jy = float(np.nanmin(valid_data))
                report.max_jy = float(np.nanmax(valid_data))
                report.mean_jy = float(np.nanmean(valid_data))

                # RMS from corner regions
                corner_size = min(data.shape) // 10
                corners = [
                    data[:corner_size, :corner_size],
                    data[:corner_size, -corner_size:],
                    data[-corner_size:, :corner_size],
                    data[-corner_size:, -corner_size:],
                ]
                noise_data = np.concatenate([c.ravel() for c in corners])
                noise_data = noise_data[np.isfinite(noise_data)]
                if len(noise_data) > 0:
                    report.rms_jy = float(np.std(noise_data))

                # Dynamic range
                if report.rms_jy > 0:
                    report.dynamic_range = report.peak_flux_jy / report.rms_jy

            # Beam information from header_info dict
            if header_info.get("bmaj"):
                report.beam_major_arcsec = header_info["bmaj"] * 3600
            if header_info.get("bmin"):
                report.beam_minor_arcsec = header_info["bmin"] * 3600
            if header_info.get("bpa"):
                report.beam_pa_deg = header_info["bpa"]

        except Exception as err:
            logger.warning("Failed to load image statistics: %s", err)

    # Quality assessment
    if report.dynamic_range >= 1000:
        report.quality_grade = "Excellent"
    elif report.dynamic_range >= 100:
        report.quality_grade = "Good"
    elif report.dynamic_range >= 10:
        report.quality_grade = "Acceptable"
    elif report.dynamic_range > 0:
        report.quality_grade = "Poor"
    else:
        report.quality_grade = "Unknown"

    # Generate recommendations
    if report.dynamic_range < 100 and report.dynamic_range > 0:
        report.recommendations.append(
            "Dynamic range is low. Consider deeper cleaning or better calibration."
        )
    if report.rms_jy > 0.01:
        report.recommendations.append(
            f"RMS noise ({report.rms_jy * 1000:.2f} mJy) is high. Check for RFI or calibration issues."
        )
    if report.beam_major_arcsec > 60:
        report.recommendations.append("Beam is large. Consider higher resolution imaging.")

    return report


def generate_imaging_figures(
    report: ImagingReportData,
    save_pngs_dir: str | None = None,
    image_prefix: str = "imaging",
) -> dict[str, str]:
    """
    Generate diagnostic figures and return as base64-encoded strings.

    Parameters
    ----------
    report : ImagingReportData
        The report data container
    save_pngs_dir : Optional[str]
        If provided, save PNG files to this directory in addition to embedding
    image_prefix : str
        Prefix for saved PNG filenames (default: "imaging")

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
        from matplotlib.colors import SymLogNorm
    except ImportError:
        logger.warning("matplotlib not available, skipping figure generation")
        return figures

    # Create PNG output directory if specified
    if save_pngs_dir:
        png_dir = Path(save_pngs_dir)
        png_dir.mkdir(parents=True, exist_ok=True)

    # Find primary FITS file for thumbnail
    primary_fits = None
    for f in report.fits_files:
        if ".image." in f or f.endswith(".fits"):
            primary_fits = f
            break
    if not primary_fits and report.fits_files:
        primary_fits = report.fits_files[0]

    if primary_fits and Path(primary_fits).exists():
        try:
            from matplotlib.patches import Circle

            from dsa110_contimg.common.utils.fits_utils import get_2d_data_and_wcs

            data, wcs, _ = get_2d_data_and_wcs(primary_fits)

            # Figure 1: Image thumbnail with WCS projection
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection=wcs)
            vmin = -3 * report.rms_jy if report.rms_jy > 0 else None
            vmax = report.peak_flux_jy * 0.8 if report.peak_flux_jy > 0 else None

            if vmin is not None and vmax is not None and vmax > 0:
                norm = SymLogNorm(linthresh=report.rms_jy * 3, vmin=vmin, vmax=vmax)
                im = ax.imshow(data, origin="lower", cmap="RdBu_r", norm=norm)
            else:
                im = ax.imshow(data, origin="lower", cmap="RdBu_r")

            # Add primary beam overlay (DSA-110: 4.65m dishes, FWHM ≈ 1.02 λ/D)
            # At 1.4 GHz (λ=0.214m): FWHM ≈ 1.02 * 0.214 / 4.65 ≈ 0.047 rad ≈ 2.7°
            # We draw at FWHM/2 radius (1.35° = 81 arcmin)
            try:
                # Get image center
                center_x, center_y = data.shape[1] // 2, data.shape[0] // 2
                # Primary beam FWHM for DSA-110 at 1.4 GHz: ~2.7 degrees
                pb_fwhm_deg = 2.7
                pb_radius_deg = pb_fwhm_deg / 2.0
                # Convert to pixel radius using cell size
                if wcs.wcs.cdelt is not None and len(wcs.wcs.cdelt) >= 2:
                    cell_deg = abs(wcs.wcs.cdelt[0])  # degrees per pixel
                    pb_radius_pix = pb_radius_deg / cell_deg
                    # Draw primary beam circle at FWHM
                    pb_circle = Circle(
                        (center_x, center_y),
                        pb_radius_pix,
                        fill=False,
                        edgecolor="lime",
                        linewidth=2,
                        linestyle="--",
                        label=f"Primary Beam FWHM ({pb_fwhm_deg:.1f}°)",
                        transform=ax.get_transform("pixel"),
                    )
                    ax.add_patch(pb_circle)
            except Exception as pb_err:
                logger.debug("Could not add primary beam overlay: %s", pb_err)

            plt.colorbar(im, ax=ax, label="Flux (Jy/beam)")
            ax.set_xlabel("Right Ascension (J2000)")
            ax.set_ylabel("Declination (J2000)")
            ax.set_title(f"Image: {Path(primary_fits).name}")
            fig.tight_layout()

            # Save to file if directory specified
            if save_pngs_dir:
                png_path = png_dir / f"{image_prefix}_image_thumbnail.png"
                fig.savefig(png_path, format="png", dpi=120, bbox_inches="tight")
                logger.info("Saved PNG: %s", png_path)

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            figures["image_thumbnail"] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)

            # Figure 2: Histogram of pixel values
            fig, ax = plt.subplots(figsize=(8, 5))
            valid_data = data[np.isfinite(data)].ravel()
            if len(valid_data) > 0:
                # Clip to reasonable range for histogram
                clip_val = (
                    10 * report.rms_jy
                    if report.rms_jy > 0
                    else np.percentile(np.abs(valid_data), 99)
                )
                clipped = valid_data[np.abs(valid_data) < clip_val]
                ax.hist(clipped * 1000, bins=100, alpha=0.7, edgecolor="black", color="#3498db")
                if report.rms_jy > 0:
                    ax.axvline(
                        report.rms_jy * 1000,
                        color="red",
                        linestyle="--",
                        label=f"RMS = {report.rms_jy * 1000:.2f} mJy",
                    )
                    ax.axvline(-report.rms_jy * 1000, color="red", linestyle="--")
                ax.set_xlabel("Flux (mJy/beam)")
                ax.set_ylabel("Count")
                ax.set_title("Pixel Value Distribution")
                ax.legend()
                ax.grid(True, alpha=0.3)
            fig.tight_layout()

            # Save to file if directory specified
            if save_pngs_dir:
                png_path = png_dir / f"{image_prefix}_histogram.png"
                fig.savefig(png_path, format="png", dpi=120, bbox_inches="tight")
                logger.info("Saved PNG: %s", png_path)

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            figures["histogram"] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)

        except Exception as err:
            logger.warning("Failed to generate image figures: %s", err)

    return figures


def generate_html_report(
    report: ImagingReportData,
    output_path: str | None = None,
    embed_figures: bool = True,
    save_pngs_dir: str | None = None,
    image_prefix: str = "imaging",
) -> str:
    """
    Generate HTML report from ImagingReportData.

    Parameters
    ----------
    report : ImagingReportData
        The report data container
    output_path : Optional[str]
        Path to save the HTML report. If None, returns HTML string only.
    embed_figures : bool
        Whether to generate and embed figures (default True)
    save_pngs_dir : Optional[str]
        If provided, save PNG files to this directory
    image_prefix : str
        Prefix for saved PNG filenames (default: "imaging")

    Returns
    -------
    str
        HTML content of the report
    """
    if embed_figures:
        report.figure_base64 = generate_imaging_figures(
            report, save_pngs_dir=save_pngs_dir, image_prefix=image_prefix
        )

    html = _generate_html_content(report)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html, encoding="utf-8")
        logger.info("Imaging diagnostics report saved to: %s", output_path)

    return html


def _generate_html_content(report: ImagingReportData) -> str:
    """Generate the HTML content for the report."""
    grade_colors = {
        "Excellent": "var(--dsa-success)",
        "Good": "var(--dsa-success)",
        "Acceptable": "var(--dsa-warning)",
        "Poor": "var(--dsa-error)",
        "Unknown": "var(--dsa-text-muted)",
        "unknown": "var(--dsa-text-muted)",
    }
    grade_color = grade_colors.get(report.quality_grade, "var(--dsa-text-muted)")

    shared_css = get_shared_css()

    # Build recommendations list
    rec_items = "".join(f"<li>{r}</li>" for r in report.recommendations)
    recommendations_html = f"<ul>{rec_items}</ul>" if rec_items else "<p>No issues detected.</p>"

    # Build figure sections
    figures_html = ""
    if "image_thumbnail" in report.figure_base64:
        figures_html += f"""
        <div class="figure-container">
            <h3>Image Preview</h3>
            <img src="data:image/png;base64,{report.figure_base64["image_thumbnail"]}" alt="Image Thumbnail">
        </div>
        """
    if "histogram" in report.figure_base64:
        figures_html += f"""
        <div class="figure-container">
            <h3>Pixel Distribution</h3>
            <img src="data:image/png;base64,{report.figure_base64["histogram"]}" alt="Histogram">
        </div>
        """

    # Format beam info
    beam_info = f'{report.beam_major_arcsec:.2f}" × {report.beam_minor_arcsec:.2f}" @ {report.beam_pa_deg:.1f}°'

    # Build FITS files list
    fits_list = "".join(f"<li><code>{f}</code></li>" for f in report.fits_files)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Imaging Diagnostics Report</title>
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
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
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
    </style>
</head>
<body>
    <div class="header">
        <h1>Imaging Diagnostics Report</h1>
        <div class="meta">
            <strong>MS:</strong> {report.ms_path}<br>
            <strong>Image:</strong> {report.image_path}<br>
            <strong>Generated:</strong> {report.timestamp}
        </div>
    </div>

    <div class="card">
        <h2>Image Quality</h2>
        <div class="quality-badge">{report.quality_grade}</div>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="value">{report.peak_flux_jy * 1000:.2f} mJy</div>
                <div class="label">Peak Flux</div>
            </div>
            <div class="stat-box">
                <div class="value">{report.rms_jy * 1e6:.1f} µJy</div>
                <div class="label">RMS Noise</div>
            </div>
            <div class="stat-box">
                <div class="value">{report.dynamic_range:.0f}</div>
                <div class="label">Dynamic Range</div>
            </div>
            <div class="stat-box">
                <div class="value">{beam_info}</div>
                <div class="label">Synthesized Beam</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Imaging Parameters</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Image Size</td><td>{report.imsize} pixels</td></tr>
            <tr><td>Cell Size</td><td>{report.cell_arcsec:.2f} arcsec</td></tr>
            <tr><td>Weighting</td><td>{report.weighting} (robust={report.robust})</td></tr>
            <tr><td>Iterations</td><td>{report.niter}</td></tr>
            <tr><td>Threshold</td><td>{report.threshold}</td></tr>
            <tr><td>Gridder</td><td>{report.gridder}</td></tr>
            <tr><td>Quality Tier</td><td>{report.quality_tier}</td></tr>
        </table>
    </div>

    <div class="card">
        <h2>Recommendations</h2>
        <div class="recommendations">
            {recommendations_html}
        </div>
    </div>

    <div class="card">
        <h2>Diagnostic Figures</h2>
        {figures_html if figures_html else "<p>No figures generated.</p>"}
    </div>

    <div class="card">
        <h2>Output Files</h2>
        <ul>
            {fits_list if fits_list else "<li>No FITS files found</li>"}
        </ul>
    </div>
</body>
</html>
"""
    return html


def generate_imaging_report(
    ms_path: str,
    image_path: str,
    output_dir: str,
    imaging_params: dict[str, Any] | None = None,
    save_pngs: bool = True,
) -> str:
    """
    High-level function to generate complete imaging diagnostics report.

    This is the main entry point for the pipeline integration.

    Parameters
    ----------
    ms_path : str
        Path to the measurement set
    image_path : str
        Path to the output image prefix
    output_dir : str
        Directory to save the report
    imaging_params : Optional[Dict[str, Any]]
        Dictionary of imaging parameters used
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
    logger.info("Loading imaging diagnostics data")
    report_data = load_imaging_report_data(ms_path, image_path, imaging_params)

    # Generate report filename based on image name
    image_name = Path(image_path).stem
    report_filename = f"imaging_diagnostics_{image_name}.html"
    report_path = output_dir / report_filename

    # Determine PNG save directory
    save_pngs_dir = str(output_dir) if save_pngs else None

    # Generate HTML report (and optionally save PNGs)
    logger.info("Generating imaging diagnostics report: %s", report_path)
    generate_html_report(
        report_data,
        str(report_path),
        embed_figures=True,
        save_pngs_dir=save_pngs_dir,
        image_prefix=f"imaging_{image_name}",
    )

    return str(report_path)
