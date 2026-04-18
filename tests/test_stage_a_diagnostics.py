# tests/test_stage_a_diagnostics.py
import matplotlib
matplotlib.use("Agg")

import pytest
import matplotlib.pyplot as plt

from dsa110_continuum.visualization.config import FigureConfig, PlotStyle


def test_style_context_publication():
    """FigureConfig(PUBLICATION).style_context() applies scienceplots rcParams."""
    config = FigureConfig(style=PlotStyle.PUBLICATION)
    before = plt.rcParams.get("font.size", None)
    with config.style_context():
        inside = plt.rcParams.get("font.size", None)
    assert inside is not None
    after = plt.rcParams.get("font.size", None)
    assert after == before
