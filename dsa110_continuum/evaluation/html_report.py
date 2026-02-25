"""
HTML report generator for evaluation results.

Generates browser-viewable HTML reports from JSON evaluation output files,
providing visual examination of stage-by-stage pipeline evaluation results.

Usage:
    # From JSON file
    generate_html_report("eval_20260103_143022.json", "report.html")

    # From StageEvaluationResult objects
    generate_stage_report(results, "pipeline_report.html")

    # CLI
    python -m dsa110_contimg.evaluation.html_report input.json output.html
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .stage_evaluators import CheckType, StageEvaluationResult

logger = logging.getLogger(__name__)


# =============================================================================
# HTML Templates
# =============================================================================

_CSS_STYLES = """
<style>
    :root {
        --pass-color: #22c55e;
        --fail-color: #ef4444;
        --warn-color: #f59e0b;
        --info-color: #3b82f6;
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --border-color: #e2e8f0;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
    }

    * {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        line-height: 1.6;
        color: var(--text-primary);
        background: var(--bg-secondary);
        padding: 2rem;
    }

    .container {
        max-width: 1200px;
        margin: 0 auto;
    }

    header {
        background: var(--bg-primary);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    h1 {
        font-size: 1.875rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        color: var(--text-secondary);
        font-size: 0.95rem;
    }

    .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }

    .summary-card {
        background: var(--bg-secondary);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    .summary-card .value {
        font-size: 2rem;
        font-weight: 700;
    }

    .summary-card .label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .summary-card.pass .value { color: var(--pass-color); }
    .summary-card.fail .value { color: var(--fail-color); }

    .stage-section {
        background: var(--bg-primary);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        overflow: hidden;
    }

    .stage-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.25rem 1.5rem;
        border-bottom: 1px solid var(--border-color);
        cursor: pointer;
    }

    .stage-header:hover {
        background: var(--bg-secondary);
    }

    .stage-title {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .stage-name {
        font-size: 1.25rem;
        font-weight: 600;
        text-transform: capitalize;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge.pass {
        background: #dcfce7;
        color: #166534;
    }

    .badge.fail {
        background: #fee2e2;
        color: #991b1b;
    }

    .badge.warn {
        background: #fef3c7;
        color: #92400e;
    }

    .stage-stats {
        display: flex;
        gap: 1rem;
        font-size: 0.9rem;
        color: var(--text-secondary);
    }

    .stage-content {
        padding: 1.5rem;
    }

    .checks-table {
        width: 100%;
        border-collapse: collapse;
    }

    .checks-table th,
    .checks-table td {
        padding: 0.75rem 1rem;
        text-align: left;
        border-bottom: 1px solid var(--border-color);
    }

    .checks-table th {
        background: var(--bg-secondary);
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
    }

    .checks-table tr:last-child td {
        border-bottom: none;
    }

    .checks-table tr:hover td {
        background: var(--bg-secondary);
    }

    .status-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        font-size: 14px;
    }

    .status-icon.pass {
        background: #dcfce7;
        color: var(--pass-color);
    }

    .status-icon.fail {
        background: #fee2e2;
        color: var(--fail-color);
    }

    .status-icon.warn {
        background: #fef3c7;
        color: var(--warn-color);
    }

    .check-name {
        font-weight: 500;
    }

    .check-type {
        font-size: 0.75rem;
        color: var(--text-secondary);
        background: var(--bg-secondary);
        padding: 0.125rem 0.5rem;
        border-radius: 4px;
        margin-left: 0.5rem;
    }

    .check-message {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }

    .check-details {
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
        font-size: 0.85rem;
        color: var(--text-secondary);
    }

    .errors-section,
    .warnings-section {
        margin-top: 1rem;
        padding: 1rem;
        border-radius: 8px;
    }

    .errors-section {
        background: #fef2f2;
        border: 1px solid #fecaca;
    }

    .warnings-section {
        background: #fffbeb;
        border: 1px solid #fde68a;
    }

    .errors-section h4,
    .warnings-section h4 {
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }

    .errors-section h4 { color: #991b1b; }
    .warnings-section h4 { color: #92400e; }

    .errors-section ul,
    .warnings-section ul {
        list-style: none;
        font-size: 0.9rem;
    }

    .errors-section li { color: #dc2626; }
    .warnings-section li { color: #d97706; }

    .metadata {
        margin-top: 2rem;
        padding: 1.5rem;
        background: var(--bg-primary);
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .metadata h3 {
        font-size: 1rem;
        margin-bottom: 1rem;
        color: var(--text-secondary);
    }

    .metadata-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }

    .metadata-item {
        font-size: 0.9rem;
    }

    .metadata-item .label {
        color: var(--text-secondary);
        font-size: 0.8rem;
    }

    .metadata-item .value {
        font-weight: 500;
    }

    footer {
        margin-top: 2rem;
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.85rem;
    }

    /* Collapsible stages */
    .stage-content {
        display: block;
    }

    .stage-section.collapsed .stage-content {
        display: none;
    }

    .toggle-icon {
        transition: transform 0.2s;
    }

    .stage-section.collapsed .toggle-icon {
        transform: rotate(-90deg);
    }
</style>
"""

_JS_SCRIPT = """
<script>
    document.querySelectorAll('.stage-header').forEach(header => {
        header.addEventListener('click', () => {
            header.parentElement.classList.toggle('collapsed');
        });
    });
</script>
"""


# =============================================================================
# Report Generation
# =============================================================================


def _format_check_details(check: dict[str, Any]) -> str:
    """Format check-type-specific details."""
    check_type = check.get("check_type", "")

    if check_type == CheckType.BOOLEAN:
        return f"value: {check.get('value')}"

    elif check_type == CheckType.COUNT:
        actual = check.get("actual", "?")
        expected = check.get("expected", "?")
        return f"{actual} / {expected}"

    elif check_type == CheckType.THRESHOLD:
        value = check.get("value", "?")
        threshold = check.get("threshold", "?")
        comparison = check.get("comparison", "")
        unit = check.get("unit", "")
        comp_symbol = {"gte": "≥", "lte": "≤", "eq": "="}.get(comparison, comparison)
        return f"{value} {unit} {comp_symbol} {threshold}".strip()

    elif check_type == CheckType.RANGE:
        value = check.get("value", "?")
        min_b = check.get("min_bound", "?")
        max_b = check.get("max_bound", "?")
        unit = check.get("unit", "")
        return f"{min_b} ≤ {value} {unit} ≤ {max_b}".strip()

    elif check_type == CheckType.MATCH:
        value = check.get("value", "?")
        expected = check.get("expected", "?")
        return f"{value} == {expected}"

    return ""


def _render_check_row(check: dict[str, Any]) -> str:
    """Render a single check as a table row."""
    passed = check.get("passed", False)
    required = check.get("required", True)

    # Determine status class
    if passed:
        status_class = "pass"
        status_icon = "✓"
    elif not required:
        status_class = "warn"
        status_icon = "!"
    else:
        status_class = "fail"
        status_icon = "✗"

    name = check.get("name", "unknown")
    check_type = check.get("check_type", "")
    message = check.get("message", "")
    details = _format_check_details(check)

    required_marker = "" if required else " (optional)"

    return f"""
        <tr>
            <td><span class="status-icon {status_class}">{status_icon}</span></td>
            <td>
                <span class="check-name">{name}</span>
                <span class="check-type">{check_type}</span>
                {required_marker}
            </td>
            <td class="check-message">{message}</td>
            <td class="check-details">{details}</td>
        </tr>
    """


def _render_stage_section(stage_result: dict[str, Any]) -> str:
    """Render a stage section with all checks."""
    stage = stage_result.get("stage", "unknown")
    passed = stage_result.get("passed", False)
    checks = stage_result.get("checks", [])
    errors = stage_result.get("errors", [])
    warnings = stage_result.get("warnings", [])
    num_passed = stage_result.get("num_passed", 0)
    num_checks = stage_result.get("num_checks", len(checks))

    status_class = "pass" if passed else "fail"
    status_text = "PASSED" if passed else "FAILED"

    # Render checks table
    checks_html = "\n".join(_render_check_row(c) for c in checks)

    # Render errors
    errors_html = ""
    if errors:
        error_items = "\n".join(f"<li>{e}</li>" for e in errors)
        errors_html = f"""
            <div class="errors-section">
                <h4>Errors</h4>
                <ul>{error_items}</ul>
            </div>
        """

    # Render warnings
    warnings_html = ""
    if warnings:
        warning_items = "\n".join(f"<li>{w}</li>" for w in warnings)
        warnings_html = f"""
            <div class="warnings-section">
                <h4>Warnings</h4>
                <ul>{warning_items}</ul>
            </div>
        """

    return f"""
        <section class="stage-section">
            <div class="stage-header">
                <div class="stage-title">
                    <span class="toggle-icon">▼</span>
                    <span class="stage-name">{stage}</span>
                    <span class="badge {status_class}">{status_text}</span>
                </div>
                <div class="stage-stats">
                    <span>{num_passed}/{num_checks} checks passed</span>
                </div>
            </div>
            <div class="stage-content">
                <table class="checks-table">
                    <thead>
                        <tr>
                            <th style="width: 50px;">Status</th>
                            <th>Check</th>
                            <th>Result</th>
                            <th>Details</th>
                        </tr>
                    </thead>
                    <tbody>
                        {checks_html}
                    </tbody>
                </table>
                {errors_html}
                {warnings_html}
            </div>
        </section>
    """


def generate_html_report(
    json_path: str | Path,
    output_path: str | Path | None = None,
    title: str = "Pipeline Evaluation Report",
) -> Path:
    """Generate HTML report from JSON evaluation results.

    Parameters
    ----------
    json_path : Path
        Path to JSON file with evaluation results
    output_path : optional
        Output HTML path (default: same name with .html extension)
    title : optional
        Report title

    Returns
    -------
        Path
        Path to generated HTML file
    """
    json_path = Path(json_path)
    if output_path is None:
        output_path = json_path.with_suffix(".html")
    else:
        output_path = Path(output_path)

    # Load JSON data
    with open(json_path) as f:
        data = json.load(f)

    # Generate report
    html = render_evaluation_report(data, title)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"Generated HTML report: {output_path}")
    return output_path


def render_evaluation_report(
    data: dict[str, Any],
    title: str = "Pipeline Evaluation Report",
) -> str:
    """Render evaluation data as HTML string.

    Parameters
    ----------
    data : dict
        Evaluation result dictionary (from JSON or to_dict())
    title : str, optional
        Report title

    Returns
    -------
        str
        Complete HTML document as string
    """
    # Extract metadata
    timestamp = data.get("timestamp", datetime.now().isoformat())
    dataset_path = data.get("dataset_path", "")
    num_samples = data.get("num_samples", 0)
    duration = data.get("duration_seconds", 0)

    # Handle both formats: stage_results list or sample_results with stages
    stage_results = data.get("stage_results", [])

    # If we have sample_results with stage data, extract from first sample
    if not stage_results and data.get("sample_results"):
        for sample in data["sample_results"]:
            if "stages" in sample:
                stage_results = sample["stages"]
                break

    # Calculate summary stats
    total_stages = len(stage_results)
    passed_stages = sum(1 for s in stage_results if s.get("passed", False))
    total_checks = sum(s.get("num_checks", 0) for s in stage_results)
    passed_checks = sum(s.get("num_passed", 0) for s in stage_results)

    overall_passed = passed_stages == total_stages and total_stages > 0
    pass_rate = (passed_stages / total_stages * 100) if total_stages > 0 else 0

    # Render stage sections
    stages_html = "\n".join(_render_stage_section(s) for s in stage_results)

    # Build complete HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {_CSS_STYLES}
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="subtitle">Generated {timestamp}</p>

            <div class="summary-grid">
                <div class="summary-card {"pass" if overall_passed else "fail"}">
                    <div class="value">{"PASS" if overall_passed else "FAIL"}</div>
                    <div class="label">Overall Status</div>
                </div>
                <div class="summary-card {"pass" if pass_rate == 100 else "fail" if pass_rate < 80 else ""}">
                    <div class="value">{pass_rate:.0f}%</div>
                    <div class="label">Stage Pass Rate</div>
                </div>
                <div class="summary-card">
                    <div class="value">{passed_stages}/{total_stages}</div>
                    <div class="label">Stages Passed</div>
                </div>
                <div class="summary-card">
                    <div class="value">{passed_checks}/{total_checks}</div>
                    <div class="label">Checks Passed</div>
                </div>
            </div>
        </header>

        <main>
            {stages_html}
        </main>

        <div class="metadata">
            <h3>Run Metadata</h3>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <div class="label">Dataset</div>
                    <div class="value">{dataset_path or "N/A"}</div>
                </div>
                <div class="metadata-item">
                    <div class="label">Samples</div>
                    <div class="value">{num_samples}</div>
                </div>
                <div class="metadata-item">
                    <div class="label">Duration</div>
                    <div class="value">{duration:.2f}s</div>
                </div>
                <div class="metadata-item">
                    <div class="label">Timestamp</div>
                    <div class="value">{timestamp}</div>
                </div>
            </div>
        </div>

        <footer>
            <p>DSA-110 Continuum Imaging Pipeline • Evaluation Framework</p>
        </footer>
    </div>
    {_JS_SCRIPT}
</body>
</html>
"""
    return html


def generate_stage_report(
    results: list[StageEvaluationResult],
    output_path: str | Path,
    title: str = "Pipeline Evaluation Report",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Generate HTML report from StageEvaluationResult objects.

    Parameters
    ----------
    results : list
        List of stage evaluation results
    output_path : str
        Output HTML path
    title : str, optional
        Report title
    metadata : dict, optional
        Additional metadata to include

    Returns
    -------
        str
        Path to generated HTML file
    """
    output_path = Path(output_path)

    # Convert to dict format
    data = {
        "timestamp": datetime.now().isoformat(),
        "stage_results": [r.to_dict() for r in results],
        **(metadata or {}),
    }

    # Generate report
    html = render_evaluation_report(data, title)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"Generated HTML report: {output_path}")
    return output_path


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """Command-line interface for HTML report generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate HTML report from evaluation JSON")
    parser.add_argument("input", help="Input JSON file path")
    parser.add_argument(
        "-o", "--output", help="Output HTML file path (default: input with .html extension)"
    )
    parser.add_argument("-t", "--title", default="Pipeline Evaluation Report", help="Report title")

    args = parser.parse_args()

    output_path = generate_html_report(
        args.input,
        args.output,
        args.title,
    )
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
