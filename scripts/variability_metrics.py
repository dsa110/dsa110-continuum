#!/opt/miniforge/envs/casa6/bin/python
"""Compute per-source variability metrics from stacked light curve Parquet.

Metrics (Mooley et al. 2016, ApJ 818, 105):
  m   = sigma_S / mean_S           (modulation index)
  Vs  = (S_max - S_min) / sqrt(sigma_max^2 + sigma_min^2)
  eta = reduced chi^2 against constant-flux null hypothesis

Usage:
    python scripts/variability_metrics.py [--products-dir ...]

Delegates to dsa110_continuum.lightcurves.metrics for all logic.
"""
import argparse
import os
from pathlib import Path

import pandas as pd

from dsa110_continuum.lightcurves.metrics import (
    VS_THRESHOLD,
    ETA_THRESHOLD,
    compute_metrics,
    flag_candidates,
)

PRODUCTS_DIR = Path(os.environ.get("DSA110_PRODUCTS_BASE", "/data/dsa110-proc/products/mosaics")).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--products-dir", default=str(PRODUCTS_DIR))
    args = parser.parse_args()

    products = Path(args.products_dir)
    lc_path = products / "lightcurves" / "lightcurves.parquet"
    if not lc_path.exists():
        raise FileNotFoundError(f"Run stack_lightcurves.py first: {lc_path}")

    lc = pd.read_parquet(lc_path)
    print(f"Loaded {len(lc)} source-epoch rows, {lc['source_id'].nunique()} sources.")

    metrics = compute_metrics(lc)
    metrics = flag_candidates(metrics)

    n_candidates = metrics["is_variable_candidate"].sum()
    print(f"Variability metrics computed for {len(metrics)} sources.")
    print(f"Variable candidates (Vs>{VS_THRESHOLD} or eta>{ETA_THRESHOLD}): {n_candidates}")
    if n_candidates > 0:
        top = metrics[metrics["is_variable_candidate"]].sort_values("eta", ascending=False)
        print(top[["ra_deg", "dec_deg", "n_epochs", "mean_flux", "Vs", "eta"]].head(10).to_string())

    out_path = products / "lightcurves" / "variability_metrics.parquet"
    metrics.reset_index().to_parquet(out_path, index=False)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
