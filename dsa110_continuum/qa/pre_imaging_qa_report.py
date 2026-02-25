"""
Pre-Imaging QA Report Generator.

Generates HTML reports for pre-imaging quality assessment,
including UV coverage, baseline distribution, and data quality metrics.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dsa110_contimg.infrastructure.database.models import MSIndex, QAPlot
from dsa110_contimg.infrastructure.database.session import get_session
from dsa110_contimg.common.utils.template_styles import get_shared_css

logger = logging.getLogger(__name__)


@dataclass
class PreImagingQAReportData:
    """Container for pre-imaging QA report data."""

    # Identification
    ms_path: str
    observation_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # UV coverage metrics
    uv_coverage_score: float = 0.0
    baseline_count: int = 0
    shortest_baseline: float = 0.0
    longest_baseline: float = 0.0

    # QA assessment
    uv_qa_passed: bool = False
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Plot paths
    uv_coverage_plot: str | None = None
    uv_density_plot: str | None = None

    # Figure base64 encoded images
    figure_base64: dict[str, str] = field(default_factory=dict)


def load_pre_imaging_qa_data(ms_path: str) -> PreImagingQAReportData:
    """
    Load all data needed to generate the pre-imaging QA report.

    Parameters
    ----------
    ms_path : str
        Path to the measurement set

    Returns
    -------
    PreImagingQAReportData
        Data container for report generation
    """
    obs_id = Path(ms_path).stem
    report = PreImagingQAReportData(
        ms_path=ms_path,
        observation_id=obs_id,
    )

    try:
        with get_session("pipeline") as session:
            # Get UV metrics from MSIndex
            ms_record = session.query(MSIndex).filter(MSIndex.path == ms_path).first()

            if ms_record:
                report.uv_coverage_score = ms_record.uv_coverage_score or 0.0
                report.baseline_count = ms_record.baseline_count or 0
                report.shortest_baseline = ms_record.shortest_baseline or 0.0
                report.longest_baseline = ms_record.longest_baseline or 0.0

                # Assess UV QA status
                if report.uv_coverage_score < 0.3:
                    report.uv_qa_passed = False
                    report.warnings.append(
                        f"UV coverage score ({report.uv_coverage_score:.2f}) below recommended threshold (0.3)"
                    )
                    report.recommendations.append(
                        "Consider additional observations or longer integration times"
                    )
                else:
                    report.uv_qa_passed = True

                # Check baseline coverage
                if report.baseline_count < 100:
                    report.warnings.append(f"Low baseline count ({report.baseline_count})")

            # Get UV plots
            uv_cov_plot = (
                session.query(QAPlot)
                .filter(QAPlot.ms_path == ms_path)
                .filter(QAPlot.plot_type == "uv_coverage")
                .order_by(QAPlot.generated_at.desc())
                .first()
            )
            if uv_cov_plot:
                report.uv_coverage_plot = uv_cov_plot.path
                report.figure_base64["uv_coverage"] = _encode_image(uv_cov_plot.path)

            uv_dens_plot = (
                session.query(QAPlot)
                .filter(QAPlot.ms_path == ms_path)
                .filter(QAPlot.plot_type == "uv_density")
                .order_by(QAPlot.generated_at.desc())
                .first()
            )
            if uv_dens_plot:
                report.uv_density_plot = uv_dens_plot.path
                report.figure_base64["uv_density"] = _encode_image(uv_dens_plot.path)

    except Exception as e:
        logger.error(f"Error loading pre-imaging QA data: {e}")

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


def generate_pre_imaging_qa_report(report_data: PreImagingQAReportData, output_path: str) -> None:
    """
    Generate HTML pre-imaging QA report.

    Parameters
    ----------
    report_data : PreImagingQAReportData
        Report data container
    output_path : str
        Path to save HTML report
    """
    # QA status badge
    if report_data.uv_qa_passed:
        qa_badge = '<span class="badge badge-success">PASSED</span>'
    else:
        qa_badge = '<span class="badge badge-warning">WARNING</span>'

    # Warnings section
    warnings_html = ""
    if report_data.warnings:
        warnings_html = '<div class="alert alert-warning"><h4>Warnings</h4><ul>'
        for warn in report_data.warnings:
            warnings_html += f"<li>{warn}</li>"
        warnings_html += "</ul></div>"

    # Recommendations section
    recommendations_html = ""
    if report_data.recommendations:
        recommendations_html = '<div class="alert alert-info"><h4>Recommendations</h4><ul>'
        for rec in report_data.recommendations:
            recommendations_html += f"<li>{rec}</li>"
        recommendations_html += "</ul></div>"

    # UV coverage plot
    uv_cov_html = ""
    if "uv_coverage" in report_data.figure_base64:
        uv_cov_html = f"""
        <div class="figure">
            <h3>UV Coverage</h3>
            <img src="data:image/png;base64,{report_data.figure_base64["uv_coverage"]}" alt="UV Coverage">
        </div>
        """

    # UV density plot
    uv_dens_html = ""
    if "uv_density" in report_data.figure_base64:
        uv_dens_html = f"""
        <div class="figure">
            <h3>UV Density</h3>
            <img src="data:image/png;base64,{report_data.figure_base64["uv_density"]}" alt="UV Density">
        </div>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pre-Imaging QA Report: {report_data.observation_id}</title>
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
            <h1>Pre-Imaging QA Report</h1>
            <p class="subtitle">Generated: {report_data.timestamp}</p>

            <div class="section">
                <h2>Observation Information</h2>
                <table>
                    <tr><th>Observation ID</th><td>{report_data.observation_id}</td></tr>
                    <tr><th>MS Path</th><td>{report_data.ms_path}</td></tr>
                    <tr><th>QA Status</th><td>{qa_badge}</td></tr>
                </table>
            </div>

            {warnings_html}
            {recommendations_html}

            <div class="section">
                <h2>UV Coverage Metrics</h2>
                <table>
                    <tr><th>UV Coverage Score</th><td>{report_data.uv_coverage_score:.3f}</td></tr>
                    <tr><th>Baseline Count</th><td>{report_data.baseline_count}</td></tr>
                    <tr><th>Shortest Baseline</th><td>{report_data.shortest_baseline:.1f} m</td></tr>
                    <tr><th>Longest Baseline</th><td>{report_data.longest_baseline:.1f} m</td></tr>
                </table>
            </div>

            <div class="section">
                <h2>UV Coverage Diagnostics</h2>
                {uv_cov_html}
                {uv_dens_html}
            </div>
        </div>
    </body>
    </html>
    """

    Path(output_path).write_text(html_content)
    logger.info(f"Pre-imaging QA report written to: {output_path}")
