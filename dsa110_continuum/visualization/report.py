"""
Report generation utilities for producing HTML/PDF reports.

Provides:
- HTML report generation with embedded figures
- Multi-page PDF reports
- Diagnostic summary pages

Adapted from ASKAP-continuum-validation/report.py patterns.
"""

from __future__ import annotations

import base64
import html
import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dsa110_contimg.common.utils.templates import render_template

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """A section within a report."""

    title: str
    content: str = ""
    html_blocks: list[str] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    figure_captions: list[str] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    table_captions: list[str] = field(default_factory=list)
    level: int = 2  # Heading level (2 = h2)


@dataclass
class ReportMetadata:
    """Metadata for a report."""

    title: str = "DSA-110 Pipeline Report"
    author: str = "DSA-110 Continuum Imaging Pipeline"
    date: datetime = field(default_factory=datetime.now)
    observation_id: str | None = None
    pipeline_version: str | None = None

    def __post_init__(self):
        if self.pipeline_version is None:
            try:
                from dsa110_contimg import __version__

                self.pipeline_version = __version__
            except ImportError:
                self.pipeline_version = "unknown"


def _figure_to_base64(fig: Figure, image_format: str = "png", dpi: int = 100) -> str:
    """Convert matplotlib figure to base64 string for embedding.

    Parameters
    ----------
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=image_format, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _file_to_base64(path: Path) -> str:
    """Convert a file's contents to base64 for HTML embedding.

    Parameters
    ----------
    """
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _render_png_file_html(path: Path, caption: str) -> str:
    b64_img = _file_to_base64(path)
    escaped_caption = html.escape(caption)
    return (
        f'<div class="figure">'
        f'<img src="data:image/png;base64,{b64_img}" alt="{escaped_caption}">'
        f'<div class="figure-caption">{escaped_caption}</div>'
        f"</div>"
    )


def _append_plot_files(section: ReportSection, plot_paths: Sequence[Path], caption: str) -> None:
    for plot_path in plot_paths:
        if plot_path.exists():
            section.html_blocks.append(_render_png_file_html(plot_path, caption))


def _classify_caltable(caltable: Path) -> str:
    name = caltable.name.lower()
    if "bcal" in name or caltable.suffix == ".B":
        return "bandpass"
    if "gcal" in name or caltable.suffix == ".G":
        return "gains"
    if "kcal" in name or caltable.suffix == ".K":
        return "delays"
    return "unknown"


def _render_figure_html(fig: Figure, caption: str) -> str:
    b64_img = _figure_to_base64(fig)
    return (
        f'<div class="figure">'
        f'<img src="data:image/png;base64,{b64_img}" alt="{caption}">'
        f'<div class="figure-caption">{caption}</div>'
        f"</div>"
    )


def _render_section_html(section: ReportSection) -> list[str]:
    parts: list[str] = []

    heading_tag = f"h{section.level}"
    parts.append(f"<{heading_tag}>{section.title}</{heading_tag}>")

    if section.content:
        parts.append(f"<p>{section.content}</p>")

    for html_block in section.html_blocks:
        parts.append(html_block)

    for idx, fig in enumerate(section.figures):
        caption = section.figure_captions[idx] if idx < len(section.figure_captions) else ""
        parts.append(_render_figure_html(fig, caption))

    for idx, table in enumerate(section.tables):
        caption = section.table_captions[idx] if idx < len(section.table_captions) else ""
        parts.append(_table_to_html(table, caption))

    return parts


def _render_sections_html(sections: Sequence[ReportSection]) -> str:
    content_parts: list[str] = []
    for section in sections:
        content_parts.extend(_render_section_html(section))
    return "\n".join(content_parts)


def _should_include_fits_viewer(sections: Sequence[ReportSection]) -> bool:
    return any(
        "fits-viewer" in block or "fits-image-block" in block
        for section in sections
        for block in section.html_blocks
    )


def _get_extra_css(sections: Sequence[ReportSection]) -> str:
    if not _should_include_fits_viewer(sections):
        return ""

    try:
        from dsa110_contimg.core.visualization.fits_viewer_templates import get_css_styles

        return _strip_style_tags(get_css_styles())
    except Exception:
        return ""


def _table_to_html(data: dict[str, Any], caption: str = "") -> str:
    """Convert dictionary data to HTML table.

    Parameters
    ----------
    data: dict[str :

    Any] :

    """
    lines = ["<table class='data-table'>"]
    if caption:
        lines.append(f"<caption>{caption}</caption>")
    lines.append("<thead><tr><th>Parameter</th><th>Value</th></tr></thead>")
    lines.append("<tbody>")

    for key, value in data.items():
        if isinstance(value, float):
            value_str = f"{value:.4g}"
        elif isinstance(value, (list, tuple)):
            value_str = ", ".join(str(v) for v in value)
        else:
            value_str = str(value)
        lines.append(f"<tr><td>{key}</td><td>{value_str}</td></tr>")

    lines.append("</tbody></table>")
    return "\n".join(lines)


def _strip_style_tags(css: str) -> str:
    css_stripped = css.strip()
    if not css_stripped:
        return ""

    # Remove leading <style ...> tag if present.
    if css_stripped.startswith("<style"):
        _, _, css_stripped = css_stripped.partition(">")

    # Remove trailing </style> tag if present.
    css_stripped = css_stripped.strip()
    if css_stripped.endswith("</style>"):
        css_stripped = css_stripped[: -len("</style>")]

    return css_stripped.strip()


def generate_html_report(
    sections: Sequence[ReportSection],
    output: str | Path,
    metadata: ReportMetadata | None = None,
) -> Path:
    """Generate an HTML report with embedded figures.

    Parameters
    ----------
    sections :
        Report sections to include
    output :
        Output file path
    metadata :
        Report metadata
    sections: Sequence[ReportSection] :

    output : Union[str, Path]
    metadata: Optional[ReportMetadata] :
         (Default value = None)

    Returns
    -------
        Path to generated report

    """
    if metadata is None:
        metadata = ReportMetadata()

    output = Path(output)

    extra_css = _get_extra_css(sections)
    content = _render_sections_html(sections)

    # Build metadata lines
    obs_line = f"<p>Observation: {metadata.observation_id}</p>" if metadata.observation_id else ""
    ver_line = (
        f"<p>Pipeline Version: {metadata.pipeline_version}</p>" if metadata.pipeline_version else ""
    )

    html_document = render_template(
        "diagnostic_report.html",
        title=metadata.title,
        date=metadata.date.strftime("%Y-%m-%d %H:%M:%S"),
        author=metadata.author,
        observation_line=obs_line,
        version_line=ver_line,
        version=metadata.pipeline_version or "unknown",
        content=content,
        extra_css=extra_css,
    )

    output.write_text(html_document, encoding="utf-8")
    logger.info("Generated HTML report: %s", output)

    return output


def generate_pdf_report(
    sections: Sequence[ReportSection],
    output: str | Path,
    metadata: ReportMetadata | None = None,
) -> Path:
    """Generate a PDF report using matplotlib's PdfPages.

    Parameters
    ----------
    sections :
        Report sections to include
    output :
        Output file path
    metadata :
        Report metadata
    sections: Sequence[ReportSection] :

    output : Union[str, Path]
    metadata: Optional[ReportMetadata] :
         (Default value = None)

    Returns
    -------
        Path to generated report

    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if metadata is None:
        metadata = ReportMetadata()

    output = Path(output)

    with PdfPages(output) as pdf:
        # Title page
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.7, metadata.title, ha="center", va="center", fontsize=24, fontweight="bold")
        fig.text(
            0.5,
            0.55,
            f"Generated: {metadata.date.strftime('%Y-%m-%d %H:%M:%S')}",
            ha="center",
            va="center",
            fontsize=12,
        )
        fig.text(0.5, 0.5, f"Author: {metadata.author}", ha="center", va="center", fontsize=12)
        if metadata.observation_id:
            fig.text(
                0.5,
                0.45,
                f"Observation: {metadata.observation_id}",
                ha="center",
                va="center",
                fontsize=12,
            )
        fig.text(
            0.5,
            0.35,
            f"Pipeline Version: {metadata.pipeline_version}",
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Section pages
        for section in sections:
            # Section header page if there's text content
            if section.content:
                fig = plt.figure(figsize=(8.5, 11))
                fig.text(
                    0.5, 0.9, section.title, ha="center", va="top", fontsize=18, fontweight="bold"
                )

                # Wrap text content
                wrapped_text = "\n".join(
                    section.content[i : i + 80] for i in range(0, len(section.content), 80)
                )
                fig.text(0.1, 0.8, wrapped_text, ha="left", va="top", fontsize=10, wrap=True)
                pdf.savefig(fig)
                plt.close(fig)

            # Add figures
            for figure in section.figures:
                pdf.savefig(figure)

        # Set PDF metadata
        d = pdf.infodict()
        d["Title"] = metadata.title
        d["Author"] = metadata.author
        d["CreationDate"] = metadata.date

    logger.info("Generated PDF report: %s", output)

    return output


def _build_overview_section(ms_path: Path) -> ReportSection:
    return ReportSection(
        title="Observation Overview",
        content=f"Diagnostic report for {ms_path.name}",
        tables=[
            {
                "Measurement Set": ms_path.name,
                "Path": str(ms_path),
                "Report Generated": datetime.now().isoformat(),
            }
        ],
        table_captions=["Observation Information"],
    )


def _try_build_calibration_section(ms_path: Path, config: Any) -> ReportSection | None:
    cal_section = ReportSection(
        title="Calibration Diagnostics",
        content="Calibration solution quality metrics.",
    )

    caltable_dir = ms_path.parent
    for pattern in ["*.bcal", "*.gcal", "*.kcal", "*.B", "*.G", "*.K"]:
        for caltable in caltable_dir.glob(pattern):
            try:
                from dsa110_contimg.core.visualization.calibration_plots import (
                    plot_bandpass,
                    plot_gains,
                    plot_kcal_delays,
                )

                kind = _classify_caltable(caltable)
                if kind == "bandpass":
                    _append_plot_files(
                        cal_section,
                        plot_bandpass(caltable, config=config),
                        f"Bandpass: {caltable.name}",
                    )
                elif kind == "gains":
                    _append_plot_files(
                        cal_section,
                        plot_gains(caltable, config=config),
                        f"Gains: {caltable.name}",
                    )
                elif kind == "delays":
                    _append_plot_files(
                        cal_section,
                        plot_kcal_delays(caltable, ms_path=ms_path, config=config),
                        f"Delays: {caltable.name}",
                    )
            except Exception as exc:
                logger.warning("Failed to plot %s: %s", caltable, exc)

    return cal_section if cal_section.figures or cal_section.html_blocks else None


def _try_build_imaging_section(
    output_dir: Path,
    config: Any,
    observation_id: str,
    ms_stem: str,
) -> ReportSection | None:
    img_section = ReportSection(
        title="Imaging Diagnostics",
        content="Image quality metrics and diagnostics.",
    )

    image_dir = output_dir / "images"
    if not image_dir.exists():
        return None

    viewer_manager = None
    render_fits_image_block = None
    try:
        from dsa110_contimg.core.visualization.fits_viewer import (
            FITSViewerConfig,
            FITSViewerManager,
        )
        from dsa110_contimg.core.visualization.fits_viewer_templates import (
            render_fits_image_block as _render_fits_image_block,
        )

        viewer_config = FITSViewerConfig.from_env()
        viewer_config.safe_mode = True
        viewer_config.safe_directories = list(
            {
                *viewer_config.safe_directories,
                str(image_dir.resolve()),
            }
        )
        viewer_manager = FITSViewerManager(viewer_config)
        render_fits_image_block = _render_fits_image_block
    except Exception:
        viewer_manager = None
        render_fits_image_block = None

    for fits_file in image_dir.glob("*.fits"):
        try:
            from dsa110_contimg.core.visualization.fits_plots import plot_fits_image

            fig = plot_fits_image(fits_file, config=config)

            caption = f"Image: {fits_file.name}"
            if viewer_manager is not None and render_fits_image_block is not None:
                b64_img = _figure_to_base64(fig)
                escaped_caption = html.escape(caption)
                png_html = (
                    '<div class="figure">'
                    f'<img src="data:image/png;base64,{b64_img}" alt="{escaped_caption}">'
                    f'<div class="figure-caption">{escaped_caption}</div>'
                    "</div>"
                )
                viewer_buttons = viewer_manager.get_viewer_buttons(
                    str(fits_file),
                    context={"observation_id": observation_id or ms_stem},
                    include_inline=False,
                )
                img_section.html_blocks.append(
                    render_fits_image_block(png_html, viewer_buttons, description="")
                )
            else:
                img_section.figures.append(fig)
                img_section.figure_captions.append(caption)
        except Exception as exc:
            logger.warning("Failed to plot %s: %s", fits_file, exc)

    if img_section.figures or img_section.html_blocks:
        return img_section

    return None


def create_diagnostic_report(
    ms_path: str | Path,
    output_dir: str | Path,
    include_calibration: bool = True,
    include_imaging: bool = True,
) -> Path:
    """Create a comprehensive diagnostic report for an observation.

    This is a convenience function that generates all relevant plots
    and assembles them into an HTML report.

    Parameters
    ----------
    ms_path :
        Path to measurement set
    output_dir :
        Directory for output files
    include_calibration :
        Include calibration diagnostic plots
    include_imaging :
        Include imaging diagnostic plots
    ms_path : Union[str, Path]
    output_dir: Union[str :

    Returns
    -------
        Path to generated report

    """
    from dsa110_contimg.core.visualization import (
        FigureConfig,
        PlotStyle,
    )

    ms_path = Path(ms_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = ReportMetadata(
        title=f"Diagnostic Report: {ms_path.name}",
        observation_id=ms_path.stem,
    )

    config = FigureConfig(style=PlotStyle.QUICKLOOK)
    sections = []

    sections.append(_build_overview_section(ms_path))

    if include_calibration:
        cal_section = _try_build_calibration_section(ms_path, config=config)
        if cal_section is not None:
            sections.append(cal_section)

    if include_imaging:
        img_section = _try_build_imaging_section(
            output_dir=output_dir,
            config=config,
            observation_id=metadata.observation_id or "",
            ms_stem=ms_path.stem,
        )
        if img_section is not None:
            sections.append(img_section)

    # Generate report
    report_path = output_dir / f"{ms_path.stem}_diagnostic_report.html"

    return generate_html_report(sections, report_path, metadata)
