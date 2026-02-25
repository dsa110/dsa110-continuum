"""
Self-Calibration Report Generator.

Generates HTML reports for self-calibration convergence and image comparison analysis.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dsa110_contimg.infrastructure.database.models import ImageComparison, QAPlot, SelfCalIteration
from dsa110_contimg.infrastructure.database.session import get_session
from dsa110_contimg.common.utils.template_styles import get_shared_css

logger = logging.getLogger(__name__)


@dataclass
class SelfCalReportData:
    """Container for self-calibration report data."""

    # Identification
    ms_path: str
    observation_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Convergence metrics
    total_iterations: int = 0
    initial_snr: float = 0.0
    final_snr: float = 0.0
    initial_rms: float = 0.0
    final_rms: float = 0.0
    total_improvement: float = 0.0
    converged: bool = False

    # Image comparison metrics
    rmse: float = 0.0
    correlation: float = 0.0
    ssim: float = 0.0

    # Iteration history
    iterations: list[dict] = field(default_factory=list)

    # Quality assessment
    recommendations: list[str] = field(default_factory=list)

    # Plot paths
    convergence_plot: str | None = None
    comparison_plot: str | None = None

    # Figure base64 encoded images
    figure_base64: dict[str, str] = field(default_factory=dict)


def load_selfcal_data(ms_path: str) -> SelfCalReportData:
    """
    Load all data needed to generate the self-calibration report.

    Parameters
    ----------
    ms_path : str
        Path to the measurement set

    Returns
    -------
    SelfCalReportData
        Data container for report generation
    """
    obs_id = Path(ms_path).stem
    report = SelfCalReportData(
        ms_path=ms_path,
        observation_id=obs_id,
    )

    try:
        with get_session("pipeline") as session:
            # Get iteration history
            iterations = (
                session.query(SelfCalIteration)
                .filter(SelfCalIteration.ms_path == ms_path)
                .order_by(SelfCalIteration.iteration)
                .all()
            )

            if iterations:
                report.total_iterations = len(iterations)
                report.initial_snr = iterations[0].snr or 0.0
                report.final_snr = iterations[-1].snr or 0.0
                report.initial_rms = iterations[0].rms or 0.0
                report.final_rms = iterations[-1].rms or 0.0
                report.converged = bool(iterations[-1].converged)

                # Calculate improvement
                if report.initial_rms > 0 and report.final_rms > 0:
                    report.total_improvement = (
                        (report.initial_rms - report.final_rms) / report.initial_rms * 100
                    )

                # Store iteration details
                report.iterations = [
                    {
                        "iteration": it.iteration,
                        "snr": it.snr,
                        "rms": it.rms,
                        "peak_flux": it.peak_flux,
                        "dynamic_range": it.dynamic_range,
                        "improvement_percent": it.improvement_percent,
                    }
                    for it in iterations
                ]

                # Assess quality
                if report.total_improvement < 5:
                    report.recommendations.append(
                        "Self-calibration showed minimal improvement (<5%). "
                        "Consider adjusting parameters or using a brighter calibrator."
                    )
                elif report.total_improvement > 50:
                    report.recommendations.append("Excellent self-calibration improvement (>50%)!")

                if not report.converged:
                    report.recommendations.append(
                        "Self-calibration did not converge. "
                        "Consider increasing iteration count or adjusting solution intervals."
                    )

            # Get image comparison data
            comparison = (
                session.query(ImageComparison)
                .filter(ImageComparison.ms_path == ms_path)
                .filter(ImageComparison.comparison_type == "selfcal")
                .order_by(ImageComparison.timestamp.desc())
                .first()
            )

            if comparison:
                report.rmse = comparison.rmse or 0.0
                report.correlation = comparison.correlation or 0.0
                report.ssim = comparison.ssim or 0.0

            # Get plots
            conv_plot = (
                session.query(QAPlot)
                .filter(QAPlot.ms_path == ms_path)
                .filter(QAPlot.plot_type == "selfcal_convergence")
                .order_by(QAPlot.generated_at.desc())
                .first()
            )
            if conv_plot:
                report.convergence_plot = conv_plot.path
                report.figure_base64["convergence"] = _encode_image(conv_plot.path)

            comp_plot = (
                session.query(QAPlot)
                .filter(QAPlot.ms_path == ms_path)
                .filter(QAPlot.plot_type == "image_comparison")
                .order_by(QAPlot.generated_at.desc())
                .first()
            )
            if comp_plot:
                report.comparison_plot = comp_plot.path
                report.figure_base64["comparison"] = _encode_image(comp_plot.path)

    except Exception as e:
        logger.error(f"Error loading self-calibration data: {e}")

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


def generate_selfcal_report(report_data: SelfCalReportData, output_path: str) -> None:
    """
    Generate HTML self-calibration report.

    Parameters
    ----------
    report_data : SelfCalReportData
        Report data container
    output_path : str
        Path to save HTML report
    """
    # Convergence badge
    if report_data.converged:
        conv_badge = '<span class="badge badge-success">CONVERGED</span>'
    else:
        conv_badge = '<span class="badge badge-warning">NOT CONVERGED</span>'

    # Improvement color coding
    if report_data.total_improvement > 20:
        improve_color = "green"
    elif report_data.total_improvement > 5:
        improve_color = "orange"
    else:
        improve_color = "red"

    # Recommendations section
    recommendations_html = ""
    if report_data.recommendations:
        recommendations_html = '<div class="alert alert-info"><h4>Recommendations</h4><ul>'
        for rec in report_data.recommendations:
            recommendations_html += f"<li>{rec}</li>"
        recommendations_html += "</ul></div>"

    # Iteration table
    iteration_rows = ""
    for it in report_data.iterations:
        iteration_rows += f"""
        <tr>
            <td>{it["iteration"]}</td>
            <td>{it["snr"]:.2f if it['snr'] else 'N/A'}</td>
            <td>{it["rms"]:.6f if it['rms'] else 'N/A'}</td>
            <td>{it["peak_flux"]:.6f if it['peak_flux'] else 'N/A'}</td>
            <td>{it["dynamic_range"]:.1f if it['dynamic_range'] else 'N/A'}</td>
            <td>{it["improvement_percent"]:.2f if it['improvement_percent'] else 'N/A'}%</td>
        </tr>
        """

    # Convergence plot
    conv_html = ""
    if "convergence" in report_data.figure_base64:
        conv_html = f"""
        <div class="figure">
            <h3>Self-Calibration Convergence</h3>
            <img src="data:image/png;base64,{report_data.figure_base64["convergence"]}" alt="Convergence">
        </div>
        """

    # Comparison plot
    comp_html = ""
    if "comparison" in report_data.figure_base64:
        comp_html = f"""
        <div class="figure">
            <h3>Pre vs Post Self-Cal Image Comparison</h3>
            <img src="data:image/png;base64,{report_data.figure_base64["comparison"]}" alt="Image Comparison">
        </div>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Self-Calibration Report: {report_data.observation_id}</title>
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
        .alert-info {{
            background-color: #d1ecf1;
            border: 1px solid #0dcaf0;
        }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Self-Calibration Report</h1>
            <p class="subtitle">Generated: {report_data.timestamp}</p>

            <div class="section">
                <h2>Observation Information</h2>
                <table>
                    <tr><th>Observation ID</th><td>{report_data.observation_id}</td></tr>
                    <tr><th>MS Path</th><td>{report_data.ms_path}</td></tr>
                    <tr><th>Convergence Status</th><td>{conv_badge}</td></tr>
                </table>
            </div>

            {recommendations_html}

            <div class="section">
                <h2>Convergence Summary</h2>
                <table>
                    <tr><th>Total Iterations</th><td>{report_data.total_iterations}</td></tr>
                    <tr><th>Initial SNR</th><td>{report_data.initial_snr:.2f}</td></tr>
                    <tr><th>Final SNR</th><td>{report_data.final_snr:.2f}</td></tr>
                    <tr><th>Initial RMS</th><td>{report_data.initial_rms:.6f} Jy</td></tr>
                    <tr><th>Final RMS</th><td>{report_data.final_rms:.6f} Jy</td></tr>
                    <tr><th>Total Improvement</th><td style="color: {improve_color}; font-weight: bold;">{report_data.total_improvement:.2f}%</td></tr>
                </table>
            </div>

            <div class="section">
                <h2>Image Comparison Metrics</h2>
                <table>
                    <tr><th>RMSE</th><td>{report_data.rmse:.6f}</td></tr>
                    <tr><th>Correlation</th><td>{report_data.correlation:.4f}</td></tr>
                    <tr><th>SSIM</th><td>{report_data.ssim:.4f}</td></tr>
                </table>
            </div>

            <div class="section">
                <h2>Iteration History</h2>
                <table>
                    <tr>
                        <th>Iteration</th>
                        <th>SNR</th>
                        <th>RMS (Jy)</th>
                        <th>Peak Flux (Jy)</th>
                        <th>Dynamic Range</th>
                        <th>Improvement</th>
                    </tr>
                    {iteration_rows}
                </table>
            </div>

            <div class="section">
                <h2>Diagnostics</h2>
                {conv_html}
                {comp_html}
            </div>
        </div>
    </body>
    </html>
    """

    Path(output_path).write_text(html_content)
    logger.info(f"Self-calibration report written to: {output_path}")
