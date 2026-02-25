"""
Visualization tools for catalog coverage.

Generates plots showing:
- Catalog coverage limits (declination ranges)
- Current telescope pointing vs coverage
- Database existence status
- Coverage gaps and overlaps
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from dsa110_contimg.core.catalog.builders import (
    CATALOG_COVERAGE_LIMITS,
    check_catalog_database_exists,
)

logger = logging.getLogger(__name__)


def plot_catalog_coverage(
    dec_deg: float | None = None,
    output_path: Path | None = None,
    show_database_status: bool = True,
    ingest_db_path: Path | None = None,
) -> Path:
    """Plot catalog coverage limits and current status.

    Parameters
    ----------
    dec_deg : float or None
        Current declination (if None, tries to get from pointing history)
    output_path : str
        Output file path (default: auto-generated)
    show_database_status : bool
        If True, show database existence status
    ingest_db_path : str
        Path to ingest database for getting current declination

    Returns
    -------
        str
        Path to generated plot
    """
    # Get current declination if not provided
    if dec_deg is None and ingest_db_path:
        try:
            if ingest_db_path.exists():
                with sqlite3.connect(str(ingest_db_path)) as conn:
                    cursor = conn.execute(
                        "SELECT dec_deg FROM pointing_history ORDER BY timestamp DESC LIMIT 1"
                    )
                    result = cursor.fetchone()
                    if result:
                        dec_deg = float(result[0])
        except Exception as e:
            logger.warning(f"Failed to get declination from pointing history: {e}")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot coverage ranges
    y_pos = 0
    colors = {
        "nvss": "#1f77b4",  # blue
        "first": "#ff7f0e",  # orange
        "rax": "#2ca02c",  # green
        "atnf": "#9467bd",  # purple
    }
    labels = {
        "nvss": "NVSS",
        "first": "FIRST",
        "rax": "RACS (RAX)",
        "atnf": "ATNF Pulsars",
    }

    catalog_info = []

    for catalog_type in ["nvss", "first", "rax", "atnf"]:
        limits = CATALOG_COVERAGE_LIMITS.get(catalog_type, {})
        dec_min = limits.get("dec_min", -90.0)
        dec_max = limits.get("dec_max", 90.0)

        # Check database status
        db_exists = False
        within_coverage = True
        if dec_deg is not None:
            within_coverage = dec_deg >= dec_min and dec_deg <= dec_max
            if within_coverage:
                db_exists, _ = check_catalog_database_exists(catalog_type, dec_deg)

        # Draw coverage bar
        width = dec_max - dec_min
        color = colors.get(catalog_type, "gray")
        alpha = 0.7 if db_exists else 0.3

        rect = Rectangle(
            (dec_min, y_pos - 0.4),
            width,
            0.8,
            facecolor=color,
            alpha=alpha,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(rect)

        # Add label
        label = labels.get(catalog_type, catalog_type.upper())
        status_text = ""
        if show_database_status and dec_deg is not None:
            if not within_coverage:
                status_text = " (outside coverage)"
            elif db_exists:
                status_text = " :check: DB exists"
            else:
                status_text = " :cross: DB missing"

        ax.text(
            dec_min + width / 2,
            y_pos,
            f"{label}{status_text}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        # Add coverage limits text
        ax.text(
            dec_min,
            y_pos - 0.6,
            f"{dec_min:.1f}°",
            ha="left",
            va="top",
            fontsize=8,
        )
        ax.text(
            dec_max,
            y_pos - 0.6,
            f"{dec_max:.1f}°",
            ha="right",
            va="top",
            fontsize=8,
        )

        catalog_info.append(
            {
                "type": catalog_type,
                "dec_min": dec_min,
                "dec_max": dec_max,
                "db_exists": db_exists,
                "within_coverage": within_coverage,
            }
        )

        y_pos += 1.5

    # Plot current declination if available
    if dec_deg is not None:
        ax.axvline(
            x=dec_deg,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Current Dec: {dec_deg:.2f}°",
        )
        ax.text(
            dec_deg,
            y_pos - 0.5,
            f"Current: {dec_deg:.2f}°",
            rotation=90,
            ha="right",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="red",
        )

    # Set axis properties
    ax.set_xlim(-95, 95)
    ax.set_ylim(-1, y_pos)
    ax.set_xlabel("Declination (degrees)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Catalog", fontsize=12, fontweight="bold")
    ax.set_title("Catalog Coverage Limits and Database Status", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_yticks([])

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor="gray", alpha=0.7, label="Database exists"),
        plt.Rectangle((0, 0), 1, 1, facecolor="gray", alpha=0.3, label="Database missing"),
    ]
    if dec_deg is not None:
        legend_elements.append(
            plt.Line2D([0], [0], color="red", linestyle="--", label="Current declination")
        )
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    # Save plot
    if output_path is None:
        output_path = Path("state/catalogs/coverage_plot.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Coverage plot saved to: {output_path}")

    plt.close()

    return output_path


def plot_coverage_summary_table(
    dec_deg: float | None = None,
    output_path: Path | None = None,
    ingest_db_path: Path | None = None,
) -> Path:
    """Create a summary table showing catalog coverage status.

    Parameters
    ----------
    dec_deg : float or None
        Current declination (if None, tries to get from pointing history)
    output_path : str
        Output file path (default: auto-generated)
    ingest_db_path : str
        Path to ingest database for getting current declination

    Returns
    -------
        str
        Path to generated plot
    """
    # Get current declination if not provided
    if dec_deg is None and ingest_db_path:
        try:
            if ingest_db_path.exists():
                with sqlite3.connect(str(ingest_db_path)) as conn:
                    cursor = conn.execute(
                        "SELECT dec_deg FROM pointing_history ORDER BY timestamp DESC LIMIT 1"
                    )
                    result = cursor.fetchone()
                    if result:
                        dec_deg = float(result[0])
        except Exception as e:
            logger.warning(f"Failed to get declination from pointing history: {e}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    # Prepare table data
    table_data = []
    headers = [
        "Catalog",
        "Coverage Range",
        "Within Coverage",
        "Database Exists",
        "Status",
    ]

    for catalog_type in ["nvss", "first", "rax", "atnf"]:
        limits = CATALOG_COVERAGE_LIMITS.get(catalog_type, {})
        dec_min = limits.get("dec_min", -90.0)
        dec_max = limits.get("dec_max", 90.0)

        catalog_name = {
            "nvss": "NVSS",
            "first": "FIRST",
            "rax": "RACS (RAX)",
            "atnf": "ATNF Pulsars",
        }.get(catalog_type, catalog_type.upper())

        coverage_range = f"{dec_min:.1f}° to {dec_max:.1f}°"

        if dec_deg is None:
            within_coverage = "N/A"
            db_exists = "N/A"
            status = "No declination data"
        else:
            within_coverage = "Yes" if (dec_deg >= dec_min and dec_deg <= dec_max) else "No"
            if within_coverage == "Yes":
                exists, _ = check_catalog_database_exists(catalog_type, dec_deg)
                db_exists = "Yes" if exists else "No"
                status = ":check: Ready" if exists else ":cross: Missing"
            else:
                db_exists = "N/A"
                status = "Outside coverage"

        table_data.append(
            [
                catalog_name,
                coverage_range,
                within_coverage,
                db_exists,
                status,
            ]
        )

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color code cells
    for i in range(len(table_data)):
        status_cell = table[(i + 1, 4)]  # Status column
        if ":check:" in table_data[i][4]:
            status_cell.set_facecolor("#90EE90")  # Light green
        elif ":cross:" in table_data[i][4]:
            status_cell.set_facecolor("#FFB6C1")  # Light red
        else:
            status_cell.set_facecolor("#D3D3D3")  # Light gray

    # Header styling
    for i in range(len(headers)):
        header_cell = table[(0, i)]
        header_cell.set_facecolor("#4CAF50")
        header_cell.set_text_props(weight="bold", color="white")

    # Title
    title = "Catalog Coverage Status"
    if dec_deg is not None:
        title += f" (Current Dec: {dec_deg:.2f}°)"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()

    # Save plot
    if output_path is None:
        output_path = Path("state/catalogs/coverage_table.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Coverage table saved to: {output_path}")

    plt.close()

    return output_path


def main():
    """CLI entry point for catalog coverage visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize catalog coverage limits and database status"
    )
    parser.add_argument(
        "--dec",
        type=float,
        default=None,
        help="Current declination in degrees (if not provided, tries to get from pointing history)",
    )
    parser.add_argument(
        "--ingest-db",
        type=Path,
        default=None,
        help="Path to ingest database (for getting current declination)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("state/catalogs"),
        help="Output directory for plots (default: state/catalogs)",
    )
    parser.add_argument(
        "--plot-type",
        choices=["both", "coverage", "table"],
        default="both",
        help="Type of plot to generate (default: both)",
    )
    parser.add_argument(
        "--no-db-status",
        action="store_true",
        help="Don't show database existence status",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Find ingest DB if not provided (now uses unified pipeline.sqlite3)
    ingest_db_path = args.ingest_db
    if ingest_db_path is None:
        for path_str in [
            os.environ.get("PIPELINE_DB", "/data/dsa110-contimg/state/db/pipeline.sqlite3"),
            "state/db/pipeline.sqlite3",
        ]:
            candidate = Path(path_str)
            if candidate.exists():
                ingest_db_path = candidate
                break

    # Generate plots
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_type in ["both", "coverage"]:
        plot_path = plot_catalog_coverage(
            dec_deg=args.dec,
            output_path=output_dir / "coverage_plot.png",
            show_database_status=not args.no_db_status,
            ingest_db_path=ingest_db_path,
        )
        print(f":check: Coverage plot: {plot_path}")

    if args.plot_type in ["both", "table"]:
        table_path = plot_coverage_summary_table(
            dec_deg=args.dec,
            output_path=output_dir / "coverage_table.png",
            ingest_db_path=ingest_db_path,
        )
        print(f":check: Coverage table: {table_path}")


if __name__ == "__main__":
    import sys

    sys.exit(main())
