"""
Source Monitoring Report Generator.

Generates HTML reports for source monitoring, including lightcurves and variability analysis.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dsa110_contimg.infrastructure.database.models import QAPlot
from dsa110_contimg.infrastructure.database.session import get_session
from dsa110_contimg.common.utils.template_styles import get_shared_css

logger = logging.getLogger(__name__)


@dataclass
class SourceMonitoringReportData:
    """Container for source monitoring report data."""

    # Identification
    source_id: str
    source_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Position
    ra_deg: float = 0.0
    dec_deg: float = 0.0

    # Variability metrics
    eta: float = 0.0  # Modulation index
    v_statistic: float = 0.0  # V-statistic
    is_variable: bool = False

    # Flux statistics
    num_epochs: int = 0
    mean_flux_mjy: float = 0.0
    std_flux_mjy: float = 0.0
    min_flux_mjy: float = 0.0
    max_flux_mjy: float = 0.0
    flux_range_percent: float = 0.0

    # Lightcurve data
    epochs: list[dict] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)

    # Plot paths
    lightcurve_plot: str | None = None
    photometry_plot: str | None = None

    # Figure base64 encoded images
    figure_base64: dict[str, str] = field(default_factory=dict)


def load_source_monitoring_data(source_id: str, db_path: str) -> SourceMonitoringReportData:
    """
    Load all data needed to generate the source monitoring report.

    Parameters
    ----------
    source_id : str
        Source identifier
    db_path : str
        Path to products database

    Returns
    -------
    SourceMonitoringReportData
        Data container for report generation
    """
    import sqlite3

    import numpy as np

    report = SourceMonitoringReportData(source_id=source_id)

    try:
        # Connect to products database (using context manager for automatic cleanup)
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get source metadata
            cursor.execute(
                "SELECT name, ra_deg, dec_deg FROM sources WHERE id = ?",
                (source_id,),
            )
            src_row = cursor.fetchone()
            if src_row:
                report.source_name = src_row["name"] or source_id
                report.ra_deg = src_row["ra_deg"]
                report.dec_deg = src_row["dec_deg"]

            # Get photometry data
            cursor.execute(
                """
                SELECT mjd, total_flux, total_flux_err, peak_flux, peak_flux_err,
                       local_rms, aperture_radius
                FROM photometry
                WHERE source_id = ?
                ORDER BY mjd
                """,
                (source_id,),
            )
            rows = cursor.fetchall()

            if rows:
                report.num_epochs = len(rows)

                # Extract flux arrays (use total_flux, convert to mJy)
                fluxes = np.array([r["total_flux"] * 1000 for r in rows if r["total_flux"]])
                errors = np.array([r["total_flux_err"] * 1000 for r in rows if r["total_flux_err"]])

                if len(fluxes) > 0:
                    report.mean_flux_mjy = float(np.mean(fluxes))
                    report.std_flux_mjy = float(np.std(fluxes))
                    report.min_flux_mjy = float(np.min(fluxes))
                    report.max_flux_mjy = float(np.max(fluxes))

                    if report.mean_flux_mjy > 0:
                        report.flux_range_percent = (
                            (report.max_flux_mjy - report.min_flux_mjy) / report.mean_flux_mjy * 100
                        )

                    # Compute variability metrics (if sufficient epochs)
                    if len(fluxes) >= 5:
                        # Modulation index (η)
                        if report.mean_flux_mjy > 0:
                            report.eta = report.std_flux_mjy / report.mean_flux_mjy

                        # V-statistic (weighted variance)
                        if len(errors) == len(fluxes) and np.all(errors > 0):
                            weights = 1.0 / errors**2
                            weighted_mean = np.sum(weights * fluxes) / np.sum(weights)
                            chi2 = np.sum(weights * (fluxes - weighted_mean) ** 2)
                            report.v_statistic = chi2 / (len(fluxes) - 1)

                            # Variability threshold (V > 2.0 is often used)
                            if report.v_statistic > 2.0:
                                report.is_variable = True

                # Store epoch details
                report.epochs = [
                    {
                        "mjd": r["mjd"],
                        "flux_mjy": r["total_flux"] * 1000 if r["total_flux"] else None,
                        "flux_err_mjy": (
                            r["total_flux_err"] * 1000 if r["total_flux_err"] else None
                        ),
                        "local_rms_mjy": r["local_rms"] * 1000 if r["local_rms"] else None,
                    }
                    for r in rows
                ]

                # Generate recommendations
                if report.is_variable:
                    report.alerts.append(
                        f"⚠ Source shows significant variability (V={report.v_statistic:.2f}, "
                        f"η={report.eta:.2f})"
                    )
                    report.recommendations.append(
                        "Increased monitoring cadence recommended. "
                        "Consider multiwavelength follow-up."
                    )

                if report.flux_range_percent > 50:
                    report.alerts.append(f"⚠ Large flux range ({report.flux_range_percent:.1f}%)")

                if report.max_flux_mjy > 5 * report.std_flux_mjy + report.mean_flux_mjy:
                    report.alerts.append("⚠ Potential flare detected")
                    report.recommendations.append(
                        "Review epochs with high flux for transient activity."
                    )

                if not report.is_variable and report.num_epochs >= 5:
                    report.recommendations.append(
                        "Source appears stable. Standard monitoring cadence sufficient."
                    )

        # Get plots from pipeline database
        with get_session("pipeline") as session:
            # Lightcurve plot
            lc_plot = (
                session.query(QAPlot)
                .filter(QAPlot.source_id == source_id)
                .filter(QAPlot.plot_type == "lightcurve")
                .order_by(QAPlot.generated_at.desc())
                .first()
            )
            if lc_plot:
                report.lightcurve_plot = lc_plot.path
                report.figure_base64["lightcurve"] = _encode_image(lc_plot.path)

            # Photometry plot
            phot_plot = (
                session.query(QAPlot)
                .filter(QAPlot.source_id == source_id)
                .filter(QAPlot.plot_type == "photometry")
                .order_by(QAPlot.generated_at.desc())
                .first()
            )
            if phot_plot:
                report.photometry_plot = phot_plot.path
                report.figure_base64["photometry"] = _encode_image(phot_plot.path)

    except Exception as e:
        logger.error(f"Error loading source monitoring data: {e}")

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


def generate_source_monitoring_report(
    report_data: SourceMonitoringReportData, output_path: str
) -> None:
    """
    Generate HTML source monitoring report.

    Parameters
    ----------
    report_data : SourceMonitoringReportData
        Report data container
    output_path : str
        Path to save HTML report
    """
    # Variability badge
    if report_data.is_variable:
        var_badge = '<span class="badge badge-warning">VARIABLE</span>'
    else:
        var_badge = '<span class="badge badge-success">STABLE</span>'

    # Alerts section
    alerts_html = ""
    if report_data.alerts:
        alerts_html = '<div class="alert alert-warning"><h4>Alerts</h4><ul>'
        for alert in report_data.alerts:
            alerts_html += f"<li>{alert}</li>"
        alerts_html += "</ul></div>"

    # Recommendations section
    recommendations_html = ""
    if report_data.recommendations:
        recommendations_html = '<div class="alert alert-info"><h4>Recommendations</h4><ul>'
        for rec in report_data.recommendations:
            recommendations_html += f"<li>{rec}</li>"
        recommendations_html += "</ul></div>"

    # Epoch table (show last 20)
    epoch_rows = ""
    for epoch in report_data.epochs[-20:]:
        epoch_rows += f"""
        <tr>
            <td>{epoch["mjd"]:.5f}</td>
            <td>{epoch["flux_mjy"]:.3f if epoch['flux_mjy'] else 'N/A'}</td>
            <td>{epoch["flux_err_mjy"]:.3f if epoch['flux_err_mjy'] else 'N/A'}</td>
            <td>{epoch["local_rms_mjy"]:.3f if epoch['local_rms_mjy'] else 'N/A'}</td>
        </tr>
        """

    # Lightcurve plot
    lc_html = ""
    if "lightcurve" in report_data.figure_base64:
        lc_html = f"""
        <div class="figure">
            <h3>Lightcurve</h3>
            <img src="data:image/png;base64,{report_data.figure_base64["lightcurve"]}" alt="Lightcurve">
        </div>
        """

    # Photometry plot
    phot_html = ""
    if "photometry" in report_data.figure_base64:
        phot_html = f"""
        <div class="figure">
            <h3>Aperture Photometry</h3>
            <img src="data:image/png;base64,{report_data.figure_base64["photometry"]}" alt="Photometry">
        </div>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Source Monitoring Report: {report_data.source_name}</title>
        <style>
        {get_shared_css()}
        .badge {{
            padding: 5px 10px;
            border-radius: 3px;
            font-weight: bold;
        }}
        .badge-success {{
            background-color: #28a745;
            color: white;
        }}
        .badge-warning {{
            background-color: #ffc107;
            color: black;
        }}
        .alert {{
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .alert-warning {{
            background-color: #fff3cd;
            border: 1px solid #ffc107;
        }}
        .alert-info {{
            background-color: #d1ecf1;
            border: 1px solid #0dcaf0;
        }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Source Monitoring Report</h1>
            <p class="subtitle">Source: {report_data.source_name} ({report_data.source_id})</p>
            <p class="subtitle">Generated: {report_data.timestamp}</p>

            <div class="section">
                <h2>Source Information</h2>
                <table>
                    <tr><th>Source ID</th><td>{report_data.source_id}</td></tr>
                    <tr><th>Source Name</th><td>{report_data.source_name}</td></tr>
                    <tr><th>RA (deg)</th><td>{report_data.ra_deg:.6f}</td></tr>
                    <tr><th>Dec (deg)</th><td>{report_data.dec_deg:.6f}</td></tr>
                    <tr><th>Variability Status</th><td>{var_badge}</td></tr>
                </table>
            </div>

            {alerts_html}
            {recommendations_html}

            <div class="section">
                <h2>Variability Metrics</h2>
                <table>
                    <tr><th>Number of Epochs</th><td>{report_data.num_epochs}</td></tr>
                    <tr><th>Modulation Index (η)</th><td>{report_data.eta:.4f}</td></tr>
                    <tr><th>V-statistic</th><td>{report_data.v_statistic:.4f}</td></tr>
                </table>
            </div>

            <div class="section">
                <h2>Flux Statistics</h2>
                <table>
                    <tr><th>Mean Flux</th><td>{report_data.mean_flux_mjy:.3f} mJy</td></tr>
                    <tr><th>Std Dev</th><td>{report_data.std_flux_mjy:.3f} mJy</td></tr>
                    <tr><th>Min Flux</th><td>{report_data.min_flux_mjy:.3f} mJy</td></tr>
                    <tr><th>Max Flux</th><td>{report_data.max_flux_mjy:.3f} mJy</td></tr>
                    <tr><th>Flux Range</th><td>{report_data.flux_range_percent:.1f}%</td></tr>
                </table>
            </div>

            <div class="section">
                <h2>Recent Epochs (Last 20)</h2>
                <table>
                    <tr>
                        <th>MJD</th>
                        <th>Flux (mJy)</th>
                        <th>Error (mJy)</th>
                        <th>Local RMS (mJy)</th>
                    </tr>
                    {epoch_rows}
                </table>
            </div>

            <div class="section">
                <h2>Diagnostics</h2>
                {lc_html}
                {phot_html}
            </div>
        </div>
    </body>
    </html>
    """

    Path(output_path).write_text(html_content)
    logger.info(f"Source monitoring report written to: {output_path}")
