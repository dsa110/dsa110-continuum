#!/opt/miniforge/envs/casa6/bin/python
"""Compute per-source variability metrics from stacked light curve Parquet.

Metrics (Mooley et al. 2016, ApJ 818, 105):
  m   = sigma_S / mean_S           (modulation index)
  Vs  = (S_max - S_min) / sqrt(sigma_max^2 + sigma_min^2)
  eta = reduced chi^2 against constant-flux null hypothesis

Usage:
    python scripts/variability_metrics.py [--products-dir ...]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PRODUCTS_DIR = Path("/data/dsa110-continuum/products")
VS_THRESHOLD = 4.0
ETA_THRESHOLD = 2.5


def compute_metrics(lc: pd.DataFrame) -> pd.DataFrame:
    """Compute m, Vs, eta for each source. Returns DataFrame indexed by source_id."""
    records = []
    for sid, group in lc.groupby("source_id"):
        n = len(group)
        fluxes = group["dsa_peak_jyb"].values
        errors = group["dsa_peak_err_jyb"].values
        ra = group["ra_deg"].iloc[0]
        dec = group["dec_deg"].iloc[0]
        nvss = group["nvss_flux_jy"].iloc[0]

        mean_s = fluxes.mean()
        std_s = fluxes.std(ddof=1) if n > 1 else np.nan

        if n >= 2:
            m = std_s / mean_s if mean_s > 0 else np.nan
            idx_max = np.argmax(fluxes)
            idx_min = np.argmin(fluxes)
            Vs = (fluxes[idx_max] - fluxes[idx_min]) / np.sqrt(
                errors[idx_max] ** 2 + errors[idx_min] ** 2
            )
            weights = 1.0 / errors ** 2
            mean_w = np.average(fluxes, weights=weights)
            chi2 = np.sum(((fluxes - mean_w) / errors) ** 2)
            eta = chi2 / (n - 1)
        else:
            m = Vs = eta = np.nan

        records.append({
            "source_id": sid,
            "ra_deg": ra,
            "dec_deg": dec,
            "nvss_flux_jy": nvss,
            "n_epochs": n,
            "mean_flux": mean_s,
            "std_flux": std_s,
            "m": m,
            "Vs": Vs,
            "eta": eta,
        })
    return pd.DataFrame(records).set_index("source_id")


def flag_candidates(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    metrics["is_variable_candidate"] = (
        (metrics["Vs"] > VS_THRESHOLD) | (metrics["eta"] > ETA_THRESHOLD)
    )
    return metrics


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
