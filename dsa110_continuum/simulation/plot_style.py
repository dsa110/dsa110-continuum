"""Plotting style helpers for DSA-110 continuum pipeline figures.

All pipeline diagnostics and output figures must use the SciencePlots
``science + notebook`` style as the default standard.  Import and call
``apply_pipeline_style()`` before creating any figure.

Style rationale
---------------
- ``science``:   clean serif axes, proper tick marks, no grid by default,
                 proportions suitable for journal figures.
- ``notebook``:  larger fonts (14 pt base) appropriate for interactive
                 review and pipeline reports.
- ``no-latex``:  fallback if a system LaTeX is unavailable; uses matplotlib's
                 built-in mathtext renderer.  Applied automatically when
                 LaTeX is not found on PATH.

Usage
-----
    from dsa110_continuum.simulation.plot_style import apply_pipeline_style
    apply_pipeline_style()
    fig, ax = plt.subplots()
    ...

Reference
---------
SciencePlots: https://github.com/garrettj403/SciencePlots
              https://pypi.org/project/SciencePlots/
"""
from __future__ import annotations

import logging
import shutil

import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Detect LaTeX availability once at import time
_LATEX_AVAILABLE: bool = shutil.which("latex") is not None

# Base styles applied to every pipeline figure
_BASE_STYLES: list[str] = ["science", "notebook"]


def apply_pipeline_style(*, extra_styles: list[str] | None = None) -> None:
    """Apply the DSA-110 pipeline matplotlib style.

    Uses SciencePlots ``science + notebook``.  If LaTeX is not installed,
    automatically appends ``no-latex`` so matplotlib's mathtext renderer is
    used instead of an external TeX engine.

    Parameters
    ----------
    extra_styles:
        Additional style names to append (e.g. ``["dark_background"]``).
        Applied after the base styles.
    """
    try:
        import scienceplots  # noqa: F401 — registers styles on import
    except ImportError:
        logger.warning(
            "scienceplots not installed; falling back to default matplotlib style. "
            "Install with: pip install scienceplots"
        )
        return

    styles = list(_BASE_STYLES)
    if not _LATEX_AVAILABLE:
        styles.append("no-latex")
    if extra_styles:
        styles.extend(extra_styles)

    plt.style.use(styles)
    logger.debug("Applied matplotlib styles: %s", styles)


def figure_and_axes(*args, **kwargs):
    """Thin wrapper around ``plt.subplots`` that applies pipeline style first.

    All positional and keyword arguments are forwarded to ``plt.subplots``.

    Example
    -------
    >>> fig, axes = figure_and_axes(1, 3, figsize=(15, 5))
    """
    apply_pipeline_style()
    return plt.subplots(*args, **kwargs)
