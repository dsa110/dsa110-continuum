"""
Vega-Lite specification generators for interactive visualizations.

Generates JSON specs that can be rendered by the frontend with pan/zoom/hover.
Uses Vega-Lite schema: https://vega.github.io/vega-lite/
"""

import json
from pathlib import Path
from typing import Any

import numpy as np


def _numpy_to_json_serializable(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable types.

    Parameters
    ----------
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.complexfloating):
        return str(obj)  # Convert complex to string representation
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _numpy_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_to_json_serializable(item) for item in obj]
    return obj


def save_vega_spec(spec: dict[str, Any], output_path: str | Path) -> None:
    """Save Vega-Lite spec to JSON file.

    Parameters
    ----------
    spec : dict
        Vega-Lite specification dictionary.
    output_path : str or Path
        Path to save JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to JSON-serializable
    spec_clean = _numpy_to_json_serializable(spec)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(spec_clean, f, indent=2)


def create_rfi_spectrum_spec(
    freqs_mhz: np.ndarray,
    occupancy: np.ndarray,
    title: str = "RFI Occupancy Spectrum",
    width: int = 800,
    height: int = 400,
) -> dict[str, Any]:
    """Generate Vega-Lite spec for RFI occupancy spectrum.

    Parameters
    ----------
    freqs_mhz :
        Frequencies in MHz.
    occupancy :
        Occupancy percentages (0-100).
    title :
        Plot title.
    width :
        Plot width in pixels.
    height :
        Plot height in pixels.
    freqs_mhz: np.ndarray :

    occupancy: np.ndarray :

    Returns
    -------
        Vega-Lite specification dictionary.

    """
    # Prepare data
    data_values = [
        {"freq_mhz": float(f), "occupancy": float(o)} for f, o in zip(freqs_mhz, occupancy)
    ]

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "RFI Occupancy vs Frequency",
        "title": title,
        "width": width,
        "height": height,
        "data": {"values": data_values},
        "mark": {"type": "line", "point": True, "tooltip": True},
        "encoding": {
            "x": {
                "field": "freq_mhz",
                "type": "quantitative",
                "title": "Frequency (MHz)",
                "scale": {"zero": False},
            },
            "y": {
                "field": "occupancy",
                "type": "quantitative",
                "title": "RFI Occupancy (%)",
                "scale": {"domain": [0, 100]},
            },
            "tooltip": [
                {
                    "field": "freq_mhz",
                    "type": "quantitative",
                    "title": "Frequency (MHz)",
                    "format": ".2f",
                },
                {
                    "field": "occupancy",
                    "type": "quantitative",
                    "title": "Occupancy (%)",
                    "format": ".2f",
                },
            ],
        },
        "config": {
            "view": {"stroke": "transparent"},
            "axis": {"grid": True, "gridOpacity": 0.3},
        },
    }

    return spec


def create_rfi_waterfall_spec(
    times: list[str],
    freqs_mhz: np.ndarray,
    occupancy_2d: np.ndarray,
    title: str = "RFI Time-Frequency Waterfall",
    width: int = 800,
    height: int = 500,
) -> dict[str, Any]:
    """Generate Vega-Lite spec for RFI time-frequency waterfall heatmap.

    Parameters
    ----------
    times :
        List of timestamp strings (ISO format).
    freqs_mhz :
        Frequencies in MHz.
    occupancy_2d :
        2D array of occupancy (time × frequency).
    title :
        Plot title.
    width :
        Plot width in pixels.
    height :
        Plot height in pixels.
    times: List[str] :

    freqs_mhz: np.ndarray :

    occupancy_2d: np.ndarray :

    Returns
    -------
        Vega-Lite specification dictionary.

    """
    if occupancy_2d.shape != (len(times), len(freqs_mhz)):
        raise ValueError(
            f"Occupancy shape {occupancy_2d.shape} does not match "
            f"times ({len(times)}) and freqs ({len(freqs_mhz)})"
        )

    # Flatten data for Vega-Lite
    data_values = []
    for i, time in enumerate(times):
        for j, freq in enumerate(freqs_mhz):
            data_values.append(
                {
                    "time": time,
                    "freq_mhz": float(freq),
                    "occupancy": float(occupancy_2d[i, j]),
                }
            )

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "RFI Time-Frequency Waterfall",
        "title": title,
        "width": width,
        "height": height,
        "data": {"values": data_values},
        "mark": "rect",
        "encoding": {
            "x": {
                "field": "time",
                "type": "temporal",
                "title": "Time (UTC)",
                "timeUnit": "yearmonthdatehoursminutes",
            },
            "y": {
                "field": "freq_mhz",
                "type": "quantitative",
                "title": "Frequency (MHz)",
                "scale": {"zero": False},
            },
            "color": {
                "field": "occupancy",
                "type": "quantitative",
                "title": "Occupancy (%)",
                "scale": {
                    "scheme": "viridis",
                    "domain": [0, 100],
                },
            },
            "tooltip": [
                {"field": "time", "type": "temporal", "title": "Time"},
                {
                    "field": "freq_mhz",
                    "type": "quantitative",
                    "title": "Frequency (MHz)",
                    "format": ".2f",
                },
                {
                    "field": "occupancy",
                    "type": "quantitative",
                    "title": "Occupancy (%)",
                    "format": ".2f",
                },
            ],
        },
        "config": {
            "view": {"stroke": "transparent", "continuousWidth": width, "continuousHeight": height},
        },
    }

    return spec


def create_residual_histogram_spec(
    residuals: np.ndarray,
    bin_count: int = 50,
    gaussian_fit: dict[str, float] | None = None,
    title: str = "Residual Distribution",
    width: int = 600,
    height: int = 400,
) -> dict[str, Any]:
    """Generate Vega-Lite spec for residual histogram with optional Gaussian overlay.

    Parameters
    ----------
    residuals :
        Flattened array of residual values.
    bin_count :
        Number of histogram bins.
    gaussian_fit :
        Dict with 'mean', 'std', 'amplitude' for Gaussian overlay (optional).
    title :
        Plot title.
    width :
        Plot width in pixels.
    height :
        Plot height in pixels.
    residuals: np.ndarray :

    Returns
    -------
        Vega-Lite specification dictionary.

    """
    # Compute histogram
    hist, bin_edges = np.histogram(residuals, bins=bin_count)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Prepare histogram data
    hist_data = [{"value": float(bc), "count": int(h)} for bc, h in zip(bin_centers, hist)]

    layers = [
        {
            "mark": {"type": "bar", "opacity": 0.7, "tooltip": True},
            "encoding": {
                "x": {
                    "field": "value",
                    "type": "quantitative",
                    "title": "Residual Value",
                    "bin": False,
                },
                "y": {
                    "field": "count",
                    "type": "quantitative",
                    "title": "Count",
                },
                "tooltip": [
                    {"field": "value", "type": "quantitative", "title": "Value", "format": ".3f"},
                    {"field": "count", "type": "quantitative", "title": "Count"},
                ],
            },
        }
    ]

    # Add Gaussian overlay if provided
    if gaussian_fit:
        x_range = np.linspace(residuals.min(), residuals.max(), 200)
        gaussian_y = gaussian_fit["amplitude"] * np.exp(
            -0.5 * ((x_range - gaussian_fit["mean"]) / gaussian_fit["std"]) ** 2
        )
        gaussian_data = [
            {"value": float(x), "gaussian": float(y)} for x, y in zip(x_range, gaussian_y)
        ]

        layers.append(
            {
                "data": {"values": gaussian_data},
                "mark": {"type": "line", "color": "red", "strokeWidth": 2},
                "encoding": {
                    "x": {"field": "value", "type": "quantitative"},
                    "y": {"field": "gaussian", "type": "quantitative"},
                },
            }
        )

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Residual Histogram with Gaussian Fit",
        "title": title,
        "width": width,
        "height": height,
        "data": {"values": hist_data},
        "layer": layers,
        "config": {
            "view": {"stroke": "transparent"},
            "axis": {"grid": True, "gridOpacity": 0.3},
        },
    }

    return spec


def create_scatter_spec(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_label: str = "X",
    y_label: str = "Y",
    title: str = "Scatter Plot",
    color_field: np.ndarray | None = None,
    color_label: str | None = None,
    width: int = 600,
    height: int = 600,
) -> dict[str, Any]:
    """Generate Vega-Lite spec for scatter plot (used for PSF correlation, etc.).

    Parameters
    ----------
    x_data :
        X-axis values.
    y_data :
        Y-axis values.
    x_label :
        X-axis label.
    y_label :
        Y-axis label.
    title :
        Plot title.
    color_field :
        Optional array for color encoding.
    color_label :
        Label for color field.
    width :
        Plot width in pixels.
    height :
        Plot height in pixels.
    x_data: np.ndarray :

    y_data: np.ndarray :

    Returns
    -------
        Vega-Lite specification dictionary.

    """
    # Prepare data
    data_values = []
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        point = {"x": float(x), "y": float(y)}
        if color_field is not None:
            point["color"] = float(color_field[i])
        data_values.append(point)

    encoding = {
        "x": {
            "field": "x",
            "type": "quantitative",
            "title": x_label,
            "scale": {"zero": False},
        },
        "y": {
            "field": "y",
            "type": "quantitative",
            "title": y_label,
            "scale": {"zero": False},
        },
        "tooltip": [
            {"field": "x", "type": "quantitative", "title": x_label, "format": ".3f"},
            {"field": "y", "type": "quantitative", "title": y_label, "format": ".3f"},
        ],
    }

    if color_field is not None and color_label:
        encoding["color"] = {
            "field": "color",
            "type": "quantitative",
            "title": color_label,
            "scale": {"scheme": "viridis"},
        }
        encoding["tooltip"].append(
            {"field": "color", "type": "quantitative", "title": color_label, "format": ".3f"}
        )

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Scatter plot with optional color encoding",
        "title": title,
        "width": width,
        "height": height,
        "data": {"values": data_values},
        "mark": {"type": "circle", "size": 60, "opacity": 0.7, "tooltip": True},
        "encoding": encoding,
        "config": {
            "view": {"stroke": "transparent"},
            "axis": {"grid": True, "gridOpacity": 0.3},
        },
    }

    return spec
