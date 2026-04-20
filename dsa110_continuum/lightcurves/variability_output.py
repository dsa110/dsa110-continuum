"""Write variability summary CSV from a metrics DataFrame."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_CSV_COLUMNS = [
    "ra_deg", "dec_deg", "n_epochs", "mean_flux", "std_flux",
    "m", "Vs", "eta", "is_variable_candidate", "catalog_flux_jy",
]


def write_variability_summary(
    metrics: pd.DataFrame,
    out_dir: str | Path,
    filename: str = "variability_summary.csv",
) -> Path:
    """Write variability metrics DataFrame to a CSV, sorted by eta descending.

    Parameters
    ----------
    metrics : DataFrame
        Output of :func:`dsa110_continuum.lightcurves.metrics.compute_metrics`.
        Indexed by source_id.
    out_dir : str or Path
        Directory in which to write the CSV.
    filename : str
        Output filename (default: variability_summary.csv).

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    # Reset index so source_id becomes a column
    df = metrics.reset_index()

    # Select and order columns; include source_id plus available _CSV_COLUMNS
    cols = ["source_id"] + [c for c in _CSV_COLUMNS if c in df.columns]
    df = df[cols].sort_values("eta", ascending=False, na_position="last")

    df.to_csv(str(out_path), index=False)

    n_cand = int(df["is_variable_candidate"].sum()) if "is_variable_candidate" in df.columns else 0
    log.info(
        "Variability summary written: %s (%d sources, %d candidates)",
        out_path, len(df), n_cand,
    )
    return out_path
