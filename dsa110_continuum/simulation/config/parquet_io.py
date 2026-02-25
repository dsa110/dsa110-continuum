"""Parquet I/O for simulation reference layout.

This module provides efficient Parquet-based storage for simulation reference layouts,
replacing the previous JSON format with ~10x size reduction and ~20x faster loading.

Example usage:
    # Write (one-time migration)
    from dsa110_contimg.core.simulation.config.parquet_io import write_reference_layout_parquet
    write_reference_layout_parquet(json_data, Path("reference_layout.parquet"))

    # Read
    from dsa110_contimg.core.simulation.config.parquet_io import load_reference_layout_parquet
    layout = load_reference_layout_parquet(Path("reference_layout.parquet"))
    
    # Selective column loading
    layout = load_reference_layout_parquet(path, columns=["nfreqs", "freq_array_hz"])
"""

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def write_reference_layout_parquet(data: dict[str, Any], output_path: Path) -> Path:
    """Write reference layout data to Parquet format.

    Parameters
    ----------
    data : dict
        Reference layout dictionary with fields:
        - Scalars: channel_width_hz, filename, integration_time_sec, lst_step_rad,
                   nbls, nblts, nfreqs, npol, nspws, time_span_sec, uvw_order
        - Arrays: freq_array_hz, lst_array_rad, polarization_array, time_array_mjd
        - Nested: extra_keywords (dict with applied_delays_ns, ha_phase_center,
                  phase_center_dec, phase_center_epoch)
    output_path : Path
        Output Parquet file path

    Returns
    -------
    Path
        Path to written Parquet file
    """
    # Define schema with explicit types
    # Using list types for arrays to enable predicate pushdown on scalar columns
    schema = pa.schema([
        # Scalar fields
        ("channel_width_hz", pa.float64()),
        ("filename", pa.string()),
        ("integration_time_sec", pa.float64()),
        ("lst_step_rad", pa.float64()),
        ("nbls", pa.int64()),
        ("nblts", pa.int64()),
        ("nfreqs", pa.int64()),
        ("npol", pa.int64()),
        ("nspws", pa.int64()),
        ("time_span_sec", pa.float64()),
        ("uvw_order", pa.string()),
        # Array fields - stored as lists for compact storage
        ("freq_array_hz", pa.list_(pa.float64())),
        ("lst_array_rad", pa.list_(pa.float64())),
        ("polarization_array", pa.list_(pa.int64())),
        ("time_array_mjd", pa.list_(pa.float64())),
        # Nested struct for extra_keywords
        ("extra_keywords_applied_delays_ns", pa.string()),
        ("extra_keywords_ha_phase_center", pa.float64()),
        ("extra_keywords_phase_center_dec", pa.float64()),
        ("extra_keywords_phase_center_epoch", pa.string()),
    ])

    # Extract extra_keywords fields
    extra = data.get("extra_keywords", {})

    # Build single-row table (reference layout is one configuration)
    table = pa.table(
        {
            "channel_width_hz": [data["channel_width_hz"]],
            "filename": [data["filename"]],
            "integration_time_sec": [data["integration_time_sec"]],
            "lst_step_rad": [data["lst_step_rad"]],
            "nbls": [data["nbls"]],
            "nblts": [data["nblts"]],
            "nfreqs": [data["nfreqs"]],
            "npol": [data["npol"]],
            "nspws": [data["nspws"]],
            "time_span_sec": [data["time_span_sec"]],
            "uvw_order": [data["uvw_order"]],
            "freq_array_hz": [data["freq_array_hz"]],
            "lst_array_rad": [data["lst_array_rad"]],
            "polarization_array": [data["polarization_array"]],
            "time_array_mjd": [data["time_array_mjd"]],
            "extra_keywords_applied_delays_ns": [extra.get("applied_delays_ns", "")],
            "extra_keywords_ha_phase_center": [extra.get("ha_phase_center", 0.0)],
            "extra_keywords_phase_center_dec": [extra.get("phase_center_dec", 0.0)],
            "extra_keywords_phase_center_epoch": [extra.get("phase_center_epoch", "")],
        },
        schema=schema,
    )

    # Write with compression (ZSTD is fast and efficient for this data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="zstd")

    return output_path


def load_reference_layout_parquet(
    path: Path,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Load reference layout from Parquet format.

    Parameters
    ----------
    path : Path
        Path to Parquet file
    columns : list[str], optional
        Specific columns to load. If None, loads all columns.
        For selective loading, specify column names like:
        ["nfreqs", "freq_array_hz", "extra_keywords"]

    Returns
    -------
    dict
        Reference layout dictionary matching the original JSON structure
    """
    # Map user-friendly column names to actual Parquet columns
    extra_keywords_cols = [
        "extra_keywords_applied_delays_ns",
        "extra_keywords_ha_phase_center",
        "extra_keywords_phase_center_dec",
        "extra_keywords_phase_center_epoch",
    ]

    # Expand "extra_keywords" to individual columns if requested
    if columns is not None:
        expanded_columns = []
        for col in columns:
            if col == "extra_keywords":
                expanded_columns.extend(extra_keywords_cols)
            else:
                expanded_columns.append(col)
        columns = expanded_columns

    # Read table with optional column projection
    table = pq.read_table(path, columns=columns)

    # Convert to dict (single row)
    row = table.to_pydict()

    # Extract values from single-element lists
    result: dict[str, Any] = {}
    for key, values in row.items():
        if key.startswith("extra_keywords_"):
            continue  # Handle separately
        result[key] = values[0] if len(values) == 1 else values

    # Reconstruct extra_keywords nested dict if any of its columns are present
    extra_cols_present = [k for k in row.keys() if k.startswith("extra_keywords_")]
    if extra_cols_present:
        result["extra_keywords"] = {
            "applied_delays_ns": row.get("extra_keywords_applied_delays_ns", [""])[0],
            "ha_phase_center": row.get("extra_keywords_ha_phase_center", [0.0])[0],
            "phase_center_dec": row.get("extra_keywords_phase_center_dec", [0.0])[0],
            "phase_center_epoch": row.get("extra_keywords_phase_center_epoch", [""])[0],
        }

    return result


def convert_json_to_parquet(json_path: Path, parquet_path: Path | None = None) -> Path:
    """Convert existing JSON reference layout to Parquet format.

    Parameters
    ----------
    json_path : Path
        Path to existing JSON file
    parquet_path : Path, optional
        Output Parquet path. If None, uses same name with .parquet extension.

    Returns
    -------
    Path
        Path to created Parquet file
    """
    import json

    if parquet_path is None:
        parquet_path = json_path.with_suffix(".parquet")

    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    return write_reference_layout_parquet(data, parquet_path)
