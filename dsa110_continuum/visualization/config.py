"""
Configuration and styling for visualization module.

Provides consistent styling across all figure types with support for:
- Quicklook (web dashboard)
- Publication (journal-ready PDF)
- Presentation (slides)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PlotStyle(Enum):
    """Predefined plot styles."""

    QUICKLOOK = "quicklook"  # Fast, web-ready, PNG
    PUBLICATION = "publication"  # High-quality PDF for papers
    PRESENTATION = "presentation"  # Large fonts for slides
    INTERACTIVE = "interactive"  # For Jupyter notebooks


@dataclass
class FigureConfig:
    """Configuration for figure generation."""

    style: PlotStyle = PlotStyle.QUICKLOOK
    figsize: tuple[float, float] = (8, 6)
    dpi: int = 150
    cmap: str = "inferno"
    font_size: int = 12
    title_size: int | None = None
    label_size: int | None = None
    tick_size: int | None = None
    colorbar: bool = True
    grid: bool = False
    tight_layout: bool = True
    output_format: str = "png"

    # Additional style options
    line_width: float = 1.5
    marker_size: float = 6.0
    alpha: float = 0.8

    # Astronomy-specific
    flux_unit: str = "Jy/beam"
    coord_format: str = "hms"  # hms or deg

    def __post_init__(self) -> None:
        """Apply style presets after initialization."""
        if isinstance(self.style, str):
            self.style = PlotStyle(self.style)
        self._apply_style_preset()

    def _apply_style_preset(self) -> None:
        """Apply predefined style settings.

        Uses values from dsa110_contimg.common.utils.plotting utilities for consistency.
        """
        # Import utilities for consistency (circular imports are safe here)
        try:
            from dsa110_contimg.common.utils.plotting import (
                get_dpi,
                get_figure_size,
                get_font_sizes,
                get_output_format,
            )
        except ImportError:
            # Fallback if utils not available
            get_dpi = lambda s: {
                "quicklook": 140,
                "publication": 300,
                "presentation": 150,
                "interactive": 100,
            }.get(s, 140)
            get_figure_size = lambda pt, s: {
                "quicklook": (6, 5),
                "publication": (8, 6),
                "presentation": (12, 9),
                "interactive": (10, 8),
            }.get(s, (8, 6))
            get_font_sizes = lambda b, s: {
                "title": 12,
                "label": 10,
                "tick": 9,
                "legend": 9,
                "annotation": 8,
            }
            get_output_format = lambda s: {
                "quicklook": "png",
                "publication": "pdf",
                "presentation": "png",
                "interactive": "png",
            }.get(s, "png")

        style_name = self.style.value

        # Get standardized values
        self.dpi = get_dpi(style_name)
        self.figsize = get_figure_size("single", style_name)
        self.output_format = get_output_format(style_name)
        font_sizes = get_font_sizes(10, style_name)
        self.font_size = font_sizes["label"]
        self.title_size = font_sizes["title"]
        self.tick_size = font_sizes["tick"]

        # Style-specific adjustments
        if self.style == PlotStyle.QUICKLOOK:
            self.cmap = "inferno"
            self.line_width = 1.5
            self.marker_size = 6.0

        elif self.style == PlotStyle.PUBLICATION:
            self.cmap = "viridis"
            self.line_width = 1.0
            self.marker_size = 5.0

        elif self.style == PlotStyle.PRESENTATION:
            self.line_width = 2.0
            self.marker_size = 10.0
            self.cmap = "plasma"

        elif self.style == PlotStyle.INTERACTIVE:
            self.cmap = "viridis"
            self.line_width = 1.5
            self.marker_size = 6.0
            self.tight_layout = True

    @property
    def effective_title_size(self) -> int:
        """Get title font size, with auto-scaling."""
        return self.title_size or int(self.font_size * 1.2)

    @property
    def effective_label_size(self) -> int:
        """Get label font size, with auto-scaling."""
        return self.label_size or self.font_size

    @property
    def effective_tick_size(self) -> int:
        """Get tick label font size, with auto-scaling."""
        return self.tick_size or int(self.font_size * 0.9)

    def to_mpl_params(self) -> dict[str, Any]:
        """Convert to matplotlib rcParams dictionary."""
        return {
            "figure.figsize": self.figsize,
            "figure.dpi": self.dpi,
            "font.size": self.font_size,
            "axes.titlesize": self.effective_title_size,
            "axes.labelsize": self.effective_label_size,
            "xtick.labelsize": self.effective_tick_size,
            "ytick.labelsize": self.effective_tick_size,
            "lines.linewidth": self.line_width,
            "lines.markersize": self.marker_size,
            "image.cmap": self.cmap,
        }

    def apply_to_mpl(self) -> None:
        """Apply configuration to matplotlib."""
        import matplotlib.pyplot as plt

        plt.rcParams.update(self.to_mpl_params())


# Default configurations for common use cases
QUICKLOOK_CONFIG = FigureConfig(style=PlotStyle.QUICKLOOK)
PUBLICATION_CONFIG = FigureConfig(style=PlotStyle.PUBLICATION)
PRESENTATION_CONFIG = FigureConfig(style=PlotStyle.PRESENTATION)


def get_config(style: str | PlotStyle = "quicklook") -> FigureConfig:
    """Get a FigureConfig for a given style name.

    Parameters
    ----------
    style :
        Style name or PlotStyle enum
    style: str | PlotStyle :
         (Default value = "quicklook")

    Returns
    -------
        Configured FigureConfig instance

    """
    if isinstance(style, str):
        style = PlotStyle(style.lower())

    return FigureConfig(style=style)
