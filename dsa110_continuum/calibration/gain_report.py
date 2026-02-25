"""
Gain Diagnostics HTML Report Generator.

Generates comprehensive HTML reports for gain calibration quality assessment,
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
class GainReportData:
    """Container for all data needed to generate the gain report."""

    # Identification
    ms_path: str
    gaintable_paths: list[str] = field(default_factory=list)
    calibrator_name: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Per-table SNR data
    table_snr_data: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Overall summary
    n_tables: int = 0
    n_antennas: int = 0
    overall_snr_median: float = 0.0
    overall_snr_mean: float = 0.0
    overall_snr_min: float = 0.0
    overall_snr_max: float = 0.0

    # Per-antenna summary
    antenna_snr_summary: dict[int, dict[str, float]] = field(default_factory=dict)

    # Quality assessment
    low_snr_antennas: list[int] = field(default_factory=list)
    quality_grade: str = "unknown"
    recommendations: list[str] = field(default_factory=list)

    # Figure base64 encoded images
    figure_base64: dict[str, str] = field(default_factory=dict)


def load_gain_report_data(
    ms_path: str,
    gaintable_paths: list[str],
    calibrator_name: str = "unknown",
    snr_threshold: float = 3.0,
) -> GainReportData:
    """
    Load all data needed to generate the gain diagnostics report.

    Parameters
    ----------
    ms_path : str
        Path to the measurement set
    gaintable_paths : List[str]
        Paths to the gain calibration tables
    calibrator_name : str
        Name of the calibrator source
    snr_threshold : float
        Minimum acceptable SNR (default 3.0)

    Returns
    -------
    GainReportData
        Data container for report generation
    """
    from dsa110_contimg.core.qa.calibration_quality import extract_gain_snr

    report = GainReportData(
        ms_path=ms_path,
        gaintable_paths=list(gaintable_paths),
        calibrator_name=calibrator_name,
        n_tables=len(gaintable_paths),
    )

    all_snr_values = []
    antenna_snr_lists: dict[int, list[float]] = {}

    for gtable in gaintable_paths:
        if not Path(gtable).exists():
            logger.warning(f"Gain table not found: {gtable}")
            continue

        try:
            snr_data = extract_gain_snr(gtable)
            report.table_snr_data[gtable] = snr_data

            # Collect all SNR values
            all_snr_values.extend(snr_data["snr_flat"].tolist())

            # Aggregate per-antenna data
            for ant_data in snr_data["per_antenna"]:
                ant_id = ant_data["antenna"]
                if ant_id not in antenna_snr_lists:
                    antenna_snr_lists[ant_id] = []
                antenna_snr_lists[ant_id].extend(ant_data["snr_values"].tolist())

        except Exception as e:
            logger.warning(f"Failed to extract SNR from {gtable}: {e}")
            continue

    # Compute overall statistics
    if all_snr_values:
        arr = np.array(all_snr_values)
        report.overall_snr_median = float(np.nanmedian(arr))
        report.overall_snr_mean = float(np.nanmean(arr))
        report.overall_snr_min = float(np.nanmin(arr))
        report.overall_snr_max = float(np.nanmax(arr))

    # Compute per-antenna summary
    report.n_antennas = len(antenna_snr_lists)
    for ant_id, snr_list in antenna_snr_lists.items():
        arr = np.array(snr_list)
        report.antenna_snr_summary[ant_id] = {
            "median": float(np.nanmedian(arr)),
            "mean": float(np.nanmean(arr)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "count": len(snr_list),
        }

        # Flag low-SNR antennas
        if np.nanmedian(arr) < snr_threshold:
            report.low_snr_antennas.append(ant_id)

    # Quality assessment
    if report.overall_snr_median >= 10:
        report.quality_grade = "Excellent"
    elif report.overall_snr_median >= 5:
        report.quality_grade = "Good"
    elif report.overall_snr_median >= 3:
        report.quality_grade = "Acceptable"
    else:
        report.quality_grade = "Poor"

    # Generate recommendations
    if report.low_snr_antennas:
        report.recommendations.append(
            f"Antennas with low SNR (<{snr_threshold}): {sorted(report.low_snr_antennas)}"
        )
    if report.overall_snr_median < 3:
        report.recommendations.append(
            "Overall SNR is low. Consider longer integration or check calibrator flux."
        )
    if report.quality_grade == "Poor":
        report.recommendations.append(
            "Gain calibration quality is poor. Review flagging and calibrator selection."
        )

    return report


def generate_gain_figures(
    report: GainReportData,
    save_pngs_dir: str | None = None,
    image_prefix: str = "gain",
) -> dict[str, str]:
    """
    Generate diagnostic figures and return as base64-encoded strings.

    Parameters
    ----------
    report : GainReportData
        The report data container
    save_pngs_dir : Optional[str]
        If provided, save PNG files to this directory in addition to embedding
    image_prefix : str
        Prefix for saved PNG filenames (default: "gain")

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
    except ImportError:
        logger.warning("matplotlib not available, skipping figure generation")
        return figures

    # Create PNG output directory if specified
    if save_pngs_dir:
        png_dir = Path(save_pngs_dir)
        png_dir.mkdir(parents=True, exist_ok=True)

    # Collect all SNR values across tables
    all_snr = []
    for table_data in report.table_snr_data.values():
        all_snr.extend(table_data["snr_flat"].tolist())

    if not all_snr:
        return figures

    # Figure 1: SNR histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_snr, bins=30, alpha=0.7, edgecolor="black", color="#3498db")
    ax.axvline(report.overall_snr_median, color="red", linestyle="--", linewidth=2, label="Median")
    ax.axvline(3.0, color="orange", linestyle=":", linewidth=2, label="Min threshold (3)")
    ax.set_xlabel("SNR", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Gain Calibration SNR Distribution", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Save to file if directory specified
    if save_pngs_dir:
        png_path = png_dir / f"{image_prefix}_snr_histogram.png"
        fig.savefig(png_path, format="png", dpi=120, bbox_inches="tight")
        logger.info("Saved PNG: %s", png_path)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    figures["snr_histogram"] = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    # Figure 2: Per-antenna SNR bar chart
    if report.antenna_snr_summary:
        fig, ax = plt.subplots(figsize=(12, 5))
        antennas = sorted(report.antenna_snr_summary.keys())
        medians = [report.antenna_snr_summary[a]["median"] for a in antennas]
        colors = ["#e74c3c" if a in report.low_snr_antennas else "#2ecc71" for a in antennas]

        ax.bar(range(len(antennas)), medians, color=colors, edgecolor="black", alpha=0.8)
        ax.axhline(3.0, color="orange", linestyle="--", linewidth=2, label="Min threshold")
        ax.axhline(
            report.overall_snr_median,
            color="blue",
            linestyle=":",
            linewidth=2,
            label="Overall median",
        )
        ax.set_xticks(range(len(antennas)))
        ax.set_xticklabels([str(a) for a in antennas], rotation=45, ha="right")
        ax.set_xlabel("Antenna", fontsize=12)
        ax.set_ylabel("Median SNR", fontsize=12)
        ax.set_title("Per-Antenna Median SNR", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()

        # Save to file if directory specified
        if save_pngs_dir:
            png_path = png_dir / f"{image_prefix}_antenna_snr.png"
            fig.savefig(png_path, format="png", dpi=120, bbox_inches="tight")
            logger.info("Saved PNG: %s", png_path)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        figures["antenna_snr"] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

    # Figure 3: SNR vs time (if available)
    has_time_data = False
    for table_data in report.table_snr_data.values():
        if table_data.get("time_min") is not None:
            has_time_data = True
            break

    if has_time_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.tab20(np.linspace(0, 1, report.n_antennas))

        for i, (gtable, table_data) in enumerate(report.table_snr_data.items()):
            for ant_data in table_data["per_antenna"]:
                if ant_data.get("snr_time") is not None:
                    t = (ant_data["snr_time"] - ant_data["snr_time"][0]) / 60.0
                    ax.scatter(
                        t,
                        ant_data["snr_values"],
                        s=15,
                        alpha=0.5,
                        label=f"Ant {ant_data['antenna']}" if i == 0 else None,
                    )

        ax.axhline(3.0, color="orange", linestyle="--", linewidth=2, label="Min threshold")
        ax.set_xlabel("Time (minutes)", fontsize=12)
        ax.set_ylabel("SNR", fontsize=12)
        ax.set_title("Gain SNR vs Time", fontsize=14)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # Save to file if directory specified
        if save_pngs_dir:
            png_path = png_dir / f"{image_prefix}_snr_vs_time.png"
            fig.savefig(png_path, format="png", dpi=120, bbox_inches="tight")
            logger.info("Saved PNG: %s", png_path)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        figures["snr_vs_time"] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

    return figures


def generate_html_report(
    report: GainReportData,
    output_path: str | None = None,
    embed_figures: bool = True,
    save_pngs_dir: str | None = None,
    image_prefix: str = "gain",
) -> str:
    """
    Generate HTML report from GainReportData.

    Parameters
    ----------
    report : GainReportData
        The report data container
    output_path : Optional[str]
        Path to save the HTML report. If None, returns HTML string only.
    embed_figures : bool
        Whether to generate and embed figures (default True)
    save_pngs_dir : Optional[str]
        If provided, save PNG files to this directory
    image_prefix : str
        Prefix for saved PNG filenames (default: "gain")

    Returns
    -------
    str
        HTML content of the report
    """
    if embed_figures:
        report.figure_base64 = generate_gain_figures(
            report, save_pngs_dir=save_pngs_dir, image_prefix=image_prefix
        )

    # Generate HTML using template or inline
    html = _generate_html_content(report)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html, encoding="utf-8")
        logger.info(f"Gain diagnostics report saved to: {output_path}")

    return html


def _generate_html_content(report: GainReportData) -> str:
    """Generate the HTML content for the report."""
    # Quality grade styling
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

    # Build antenna table rows
    antenna_rows = ""
    for ant_id in sorted(report.antenna_snr_summary.keys()):
        stats = report.antenna_snr_summary[ant_id]
        is_low = ant_id in report.low_snr_antennas
        row_class = "low-snr" if is_low else ""
        antenna_rows += f"""
        <tr class="{row_class}">
            <td>{ant_id}</td>
            <td>{stats["median"]:.2f}</td>
            <td>{stats["mean"]:.2f}</td>
            <td>{stats["min"]:.2f}</td>
            <td>{stats["max"]:.2f}</td>
            <td>{stats["count"]}</td>
        </tr>
        """

    # Build recommendations list
    rec_items = "".join(f"<li>{r}</li>" for r in report.recommendations)
    recommendations_html = f"<ul>{rec_items}</ul>" if rec_items else "<p>No issues detected.</p>"

    # Build figure sections
    figures_html = ""
    if "snr_histogram" in report.figure_base64:
        figures_html += f"""
        <div class="figure-container">
            <h3>SNR Distribution</h3>
            <img src="data:image/png;base64,{report.figure_base64["snr_histogram"]}" alt="SNR Histogram">
        </div>
        """
    if "antenna_snr" in report.figure_base64:
        figures_html += f"""
        <div class="figure-container">
            <h3>Per-Antenna SNR</h3>
            <img src="data:image/png;base64,{report.figure_base64["antenna_snr"]}" alt="Per-Antenna SNR">
        </div>
        """
    if "snr_vs_time" in report.figure_base64:
        figures_html += f"""
        <div class="figure-container">
            <h3>SNR vs Time</h3>
            <img src="data:image/png;base64,{report.figure_base64["snr_vs_time"]}" alt="SNR vs Time">
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gain Diagnostics Report</title>
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
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
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
            font-size: 1.5em;
            font-weight: bold;
            color: var(--dsa-primary);
        }}
        .stat-box .label {{
            font-size: 0.9em;
            color: var(--dsa-text-muted);
        }}
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
        th {{
            background: var(--dsa-secondary);
            color: white;
        }}
        tr:hover {{ background: var(--dsa-bg-paper); }}
        tr.low-snr {{ background: var(--dsa-error-bg); }}
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
    </style>
</head>
<body>
    <div class="header">
        <h1>Gain Diagnostics Report</h1>
        <div class="meta">
            <strong>MS:</strong> {report.ms_path}<br>
            <strong>Calibrator:</strong> {report.calibrator_name}<br>
            <strong>Generated:</strong> {report.timestamp}
        </div>
    </div>

    <div class="card">
        <h2>Quality Assessment</h2>
        <div class="quality-badge">{report.quality_grade}</div>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="value">{report.n_tables}</div>
                <div class="label">Gain Tables</div>
            </div>
            <div class="stat-box">
                <div class="value">{report.n_antennas}</div>
                <div class="label">Antennas</div>
            </div>
            <div class="stat-box">
                <div class="value">{report.overall_snr_median:.1f}</div>
                <div class="label">Median SNR</div>
            </div>
            <div class="stat-box">
                <div class="value">{report.overall_snr_min:.1f} - {report.overall_snr_max:.1f}</div>
                <div class="label">SNR Range</div>
            </div>
            <div class="stat-box">
                <div class="value">{len(report.low_snr_antennas)}</div>
                <div class="label">Low-SNR Antennas</div>
            </div>
        </div>
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
        <h2>Per-Antenna Statistics</h2>
        <table>
            <thead>
                <tr>
                    <th>Antenna</th>
                    <th>Median SNR</th>
                    <th>Mean SNR</th>
                    <th>Min SNR</th>
                    <th>Max SNR</th>
                    <th>Samples</th>
                </tr>
            </thead>
            <tbody>
                {antenna_rows}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Gain Tables Analyzed</h2>
        <ul>
            {"".join(f"<li><code>{p}</code></li>" for p in report.gaintable_paths)}
        </ul>
    </div>
</body>
</html>
"""
    return html


def generate_gain_report(
    ms_path: str,
    gaintable_paths: list[str],
    output_dir: str,
    calibrator_name: str = "unknown",
    save_pngs: bool = True,
) -> str:
    """
    High-level function to generate complete gain diagnostics report.

    This is the main entry point for the pipeline integration.

    Parameters
    ----------
    ms_path : str
        Path to the measurement set
    gaintable_paths : List[str]
        Paths to the gain calibration tables
    output_dir : str
        Directory to save the report
    calibrator_name : str
        Name of the calibrator source
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
    logger.info("Loading gain diagnostics data from %d table(s)", len(gaintable_paths))
    report_data = load_gain_report_data(ms_path, gaintable_paths, calibrator_name)

    # Generate report filename based on MS name
    ms_name = Path(ms_path).stem
    report_filename = f"gain_diagnostics_{ms_name}.html"
    report_path = output_dir / report_filename

    # Determine PNG save directory
    save_pngs_dir = str(output_dir) if save_pngs else None

    # Generate HTML report (and optionally save PNGs)
    logger.info("Generating gain diagnostics report: %s", report_path)
    generate_html_report(
        report_data,
        str(report_path),
        embed_figures=True,
        save_pngs_dir=save_pngs_dir,
        image_prefix=f"gain_{ms_name}",
    )

    return str(report_path)
