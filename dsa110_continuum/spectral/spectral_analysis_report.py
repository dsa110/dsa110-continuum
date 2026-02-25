"""
Spectral Analysis Report Generator.

Generates HTML reports for multi-frequency spectral index analysis and source characterization.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dsa110_contimg.infrastructure.database.models import QAPlot, SpectralIndex
from dsa110_contimg.infrastructure.database.session import get_session
from dsa110_contimg.common.utils.template_styles import get_shared_css

logger = logging.getLogger(__name__)


@dataclass
class SpectralAnalysisReportData:
    """Container for spectral analysis report data."""

    # Identification
    field_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Spectral index map metrics
    mean_spectral_index: float = 0.0
    median_spectral_index: float = 0.0
    std_spectral_index: float = 0.0

    # Source SEDs
    sources: list[dict] = field(default_factory=list)

    # Classification statistics
    synchrotron_count: int = 0
    thermal_count: int = 0
    inverted_count: int = 0
    unclassified_count: int = 0

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    # Plot paths
    spectral_index_map: str | None = None
    sed_plots: list[str] = field(default_factory=list)

    # Figure base64 encoded images
    figure_base64: dict[str, str] = field(default_factory=dict)


def load_spectral_analysis_data(field_name: str) -> SpectralAnalysisReportData:
    """
    Load all data needed to generate the spectral analysis report.

    Parameters
    ----------
    field_name : str
        Name of the field/observation

    Returns
    -------
    SpectralAnalysisReportData
        Data container for report generation
    """
    report = SpectralAnalysisReportData(field_name=field_name)

    try:
        with get_session("science") as session:
            # Get all spectral index records for this field
            records = (
                session.query(SpectralIndex)
                .filter(SpectralIndex.field_name == field_name)
                .order_by(SpectralIndex.alpha.desc())
                .all()
            )

            if records:
                # Compute statistics
                alphas = [r.alpha for r in records if r.alpha is not None]
                if alphas:
                    import numpy as np

                    report.mean_spectral_index = float(np.mean(alphas))
                    report.median_spectral_index = float(np.median(alphas))
                    report.std_spectral_index = float(np.std(alphas))

                # Count classifications
                for rec in records:
                    if rec.classification == "synchrotron":
                        report.synchrotron_count += 1
                    elif rec.classification == "thermal":
                        report.thermal_count += 1
                    elif rec.classification == "inverted":
                        report.inverted_count += 1
                    else:
                        report.unclassified_count += 1

                # Store top sources
                report.sources = [
                    {
                        "source_id": rec.source_id,
                        "alpha": rec.alpha,
                        "alpha_err": rec.alpha_err,
                        "classification": rec.classification or "unclassified",
                        "num_frequencies": len(rec.frequencies_hz) if rec.frequencies_hz else 0,
                    }
                    for rec in records[:20]  # Top 20 sources
                ]

                # Recommendations
                if report.inverted_count > 0:
                    report.recommendations.append(
                        f"Found {report.inverted_count} inverted-spectrum sources. "
                        "These may be variable AGN or compact HII regions."
                    )

                if report.mean_spectral_index < -1.5:
                    report.recommendations.append(
                        "Field dominated by steep-spectrum synchrotron sources. "
                        "Typical of extended radio lobes and relic emission."
                    )
                elif report.mean_spectral_index > -0.5:
                    report.recommendations.append(
                        "Field shows relatively flat spectra. "
                        "May indicate compact sources or thermal emission."
                    )

        # Get plots from pipeline database
        with get_session("pipeline") as session:
            # Spectral index map
            map_plot = (
                session.query(QAPlot)
                .filter(QAPlot.field_name == field_name)
                .filter(QAPlot.plot_type == "spectral_index_map")
                .order_by(QAPlot.generated_at.desc())
                .first()
            )
            if map_plot:
                report.spectral_index_map = map_plot.path
                report.figure_base64["map"] = _encode_image(map_plot.path)

            # SED plots (top 5)
            sed_plots = (
                session.query(QAPlot)
                .filter(QAPlot.field_name == field_name)
                .filter(QAPlot.plot_type == "sed")
                .order_by(QAPlot.generated_at.desc())
                .limit(5)
                .all()
            )
            for i, plot in enumerate(sed_plots):
                report.sed_plots.append(plot.path)
                report.figure_base64[f"sed_{i}"] = _encode_image(plot.path)

    except Exception as e:
        logger.error(f"Error loading spectral analysis data: {e}")

    return report


def _encode_image(path: str | None) -> str:
    """Encode image file as base64 string."""
    if not path or not Path(path).exists():
        return ""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to encode image {path}: {e}")
        return ""


def generate_spectral_analysis_report(
    report_data: SpectralAnalysisReportData, output_path: str
) -> None:
    """
    Generate HTML spectral analysis report.

    Parameters
    ----------
    report_data : SpectralAnalysisReportData
        Report data container
    output_path : str
        Path to save HTML report
    """
    # Recommendations section
    recommendations_html = ""
    if report_data.recommendations:
        recommendations_html = '<div class="alert alert-info"><h4>Recommendations</h4><ul>'
        for rec in report_data.recommendations:
            recommendations_html += f"<li>{rec}</li>"
        recommendations_html += "</ul></div>"

    # Source table
    source_rows = ""
    for src in report_data.sources:
        class_color = {
            "synchrotron": "blue",
            "thermal": "orange",
            "inverted": "red",
            "unclassified": "gray",
        }.get(src["classification"], "gray")

        source_rows += f"""
        <tr>
            <td>{src["source_id"]}</td>
            <td>{src["alpha"]:.3f}</td>
            <td>{src["alpha_err"]:.3f if src['alpha_err'] else 'N/A'}</td>
            <td style="color: {class_color}; font-weight: bold;">{src["classification"]}</td>
            <td>{src["num_frequencies"]}</td>
        </tr>
        """

    # Spectral index map
    map_html = ""
    if "map" in report_data.figure_base64:
        map_html = f"""
        <div class="figure">
            <h3>Spectral Index Map</h3>
            <img src="data:image/png;base64,{report_data.figure_base64["map"]}" alt="Spectral Index Map">
        </div>
        """

    # SED plots
    sed_html = ""
    for i in range(len(report_data.sed_plots)):
        if f"sed_{i}" in report_data.figure_base64:
            sed_html += f"""
            <div class="figure">
                <h4>Source SED {i + 1}</h4>
                <img src="data:image/png;base64,{report_data.figure_base64[f"sed_{i}"]}" alt="SED">
            </div>
            """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Spectral Analysis Report: {report_data.field_name}</title>
        <style>
        {get_shared_css()}
        .alert {{
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .alert-info {{
            background-color: #d1ecf1;
            border: 1px solid #0dcaf0;
        }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Spectral Analysis Report</h1>
            <p class="subtitle">Field: {report_data.field_name}</p>
            <p class="subtitle">Generated: {report_data.timestamp}</p>

            <div class="section">
                <h2>Field Spectral Statistics</h2>
                <table>
                    <tr><th>Mean Spectral Index</th><td>{report_data.mean_spectral_index:.3f}</td></tr>
                    <tr><th>Median Spectral Index</th><td>{report_data.median_spectral_index:.3f}</td></tr>
                    <tr><th>Standard Deviation</th><td>{report_data.std_spectral_index:.3f}</td></tr>
                </table>
            </div>

            <div class="section">
                <h2>Source Classification</h2>
                <table>
                    <tr><th>Synchrotron Sources</th><td style="color: blue; font-weight: bold;">{report_data.synchrotron_count}</td></tr>
                    <tr><th>Thermal Sources</th><td style="color: orange; font-weight: bold;">{report_data.thermal_count}</td></tr>
                    <tr><th>Inverted-Spectrum Sources</th><td style="color: red; font-weight: bold;">{report_data.inverted_count}</td></tr>
                    <tr><th>Unclassified Sources</th><td style="color: gray;">{report_data.unclassified_count}</td></tr>
                </table>
            </div>

            {recommendations_html}

            <div class="section">
                <h2>Top Sources by Spectral Index</h2>
                <table>
                    <tr>
                        <th>Source ID</th>
                        <th>Spectral Index (α)</th>
                        <th>Error</th>
                        <th>Classification</th>
                        <th>Frequencies</th>
                    </tr>
                    {source_rows}
                </table>
            </div>

            <div class="section">
                <h2>Diagnostics</h2>
                {map_html}
                {sed_html}
            </div>
        </div>
    </body>
    </html>
    """

    Path(output_path).write_text(html_content)
    logger.info(f"Spectral analysis report written to: {output_path}")
