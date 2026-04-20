#!/opt/miniforge/envs/casa6/bin/python
"""Generate per-source light curve plots and a variable candidates summary HTML.

Usage:
    python scripts/plot_lightcurves.py [--products-dir ...] [--top-n 50]
"""
import argparse
import base64
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime

PRODUCTS_DIR = Path(os.environ.get("DSA110_PRODUCTS_BASE", "/data/dsa110-proc/products/mosaics")).parent


def plot_source_lightcurve(group: pd.DataFrame, nvss_flux: float, out_path: str) -> None:
    """Save a flux-vs-time plot for one source to out_path."""
    times = [datetime.fromisoformat(e) for e in group["epoch_utc"]]
    fluxes = group["measured_flux_jy"].values
    errors = group["flux_err_jy"].values

    import scienceplots  # noqa
    with plt.style.context(["science", "notebook"]):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.errorbar(times, fluxes, yerr=errors, fmt="o", color="#1f77b4",
                    capsize=3, elinewidth=1, markersize=5, label="DSA-110")
        ax.axhline(nvss_flux, color="gray", linestyle="--", linewidth=1,
                   label=f"Catalog ref. {nvss_flux:.3f} Jy")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate(rotation=30)
        ax.set_xlabel("Epoch (UTC)")
        ax.set_ylabel("Peak flux density (Jy/beam)")
        sid = group["source_id"].iloc[0]
        ra = group["ra_deg"].iloc[0]
        dec = group["dec_deg"].iloc[0]
        ax.set_title(f"Source {int(sid):06d}  RA={ra:.3f}\u00b0  Dec={dec:.3f}\u00b0")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)


def build_summary_html(
    metrics: pd.DataFrame,
    lc: pd.DataFrame,
    plots_dir: str,
    top_n: int = 50,
) -> str:
    """Build HTML string ranking variable candidates by eta, with embedded plot thumbnails."""
    candidates = (
        metrics[metrics["is_variable_candidate"]]
        .sort_values("eta", ascending=False)
        .head(top_n)
    )
    plots_path = Path(plots_dir)

    rows_html = ""
    for _, row in candidates.iterrows():
        sid = int(row["source_id"])
        plot_file = plots_path / f"{sid:06d}.png"
        img_tag = ""
        if plot_file.exists():
            with open(plot_file, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            img_tag = f'<img src="data:image/png;base64,{b64}" width="350">'
        n_ep = int(row["n_epochs"]) if not pd.isna(row.get("n_epochs", float("nan"))) else "?"
        Vs_val = row.get("Vs", float("nan"))
        Vs_str = f"{Vs_val:.2f}" if not pd.isna(Vs_val) else "\u2014"
        eta_val = row.get("eta", float("nan"))
        eta_str = f"{eta_val:.2f}" if not pd.isna(eta_val) else "\u2014"
        rows_html += f"""
        <tr>
            <td>{sid:06d}</td>
            <td>{row['ra_deg']:.4f}</td>
            <td>{row['dec_deg']:.4f}</td>
            <td>{row['catalog_flux_jy']:.3f}</td>
            <td>{n_ep}</td>
            <td>{row['mean_flux']:.4f}</td>
            <td>{Vs_str}</td>
            <td>{eta_str}</td>
            <td>{img_tag}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DSA-110 Variable Source Candidates</title>
<style>
  body {{ font-family: monospace; font-size: 12px; background: #fafafa; }}
  h2 {{ color: #333; }}
  table {{ border-collapse: collapse; margin-top: 1em; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; vertical-align: middle; }}
  th {{ background: #e8e8e8; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
</style>
</head>
<body>
<h2>DSA-110 Variable Source Candidates (top {len(candidates)} by &eta;)</h2>
<p>Criteria: V<sub>s</sub> &gt; 4.0 OR &eta; &gt; 2.5 &nbsp;|&nbsp; Metrics: Mooley et al. 2016</p>
<table>
<tr>
  <th>Source ID</th><th>RA (deg)</th><th>Dec (deg)</th>
  <th>Cat. flux (Jy)</th><th>N epochs</th><th>Mean flux (Jy/bm)</th>
  <th>V<sub>s</sub></th><th>&eta;</th><th>Light curve</th>
</tr>
{rows_html}
</table>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--products-dir", default=str(PRODUCTS_DIR))
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument(
        "--candidates-only",
        action="store_true",
        default=False,
        help="Only plot variable candidate sources (faster; suitable for summary HTML)",
    )
    args = parser.parse_args()

    products = Path(args.products_dir)
    lc_path = products / "lightcurves" / "lightcurves.parquet"
    metrics_path = products / "lightcurves" / "variability_metrics.parquet"

    if not lc_path.exists():
        raise FileNotFoundError(f"Run stack_lightcurves.py first: {lc_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Run variability_metrics.py first: {metrics_path}")

    lc = pd.read_parquet(lc_path)
    metrics = pd.read_parquet(metrics_path)

    plots_dir = products / "lightcurves" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.candidates_only:
        plot_ids = metrics[metrics["is_variable_candidate"]]["source_id"].values
        print(f"Plotting {len(plot_ids)} variable candidate sources (--candidates-only)...")
    else:
        plot_ids = metrics[metrics["n_epochs"] >= 2]["source_id"].values
        print(f"Plotting {len(plot_ids)} sources with \u22652 epochs...")

    for sid in plot_ids:
        group = lc[lc["source_id"] == sid]
        cat_flux = group["catalog_flux_jy"].iloc[0]
        out_path = plots_dir / f"{int(sid):06d}.png"
        plot_source_lightcurve(group, nvss_flux=float(cat_flux), out_path=str(out_path))

    html = build_summary_html(metrics, lc, plots_dir=str(plots_dir), top_n=args.top_n)
    summary_path = products / "lightcurves" / "variable_candidates_summary.html"
    summary_path.write_text(html)
    print(f"Written: {summary_path}")

    n_candidates = int(metrics["is_variable_candidate"].sum())
    n_plots = len(list(plots_dir.glob("*.png")))
    print(f"Summary: {n_candidates} variable candidates, {n_plots} plots generated.")

    # ── Variability metrics summary ───────────────────────────────────────────
    try:
        from dsa110_continuum.lightcurves.metrics import compute_metrics, variability_summary
        from dsa110_continuum.lightcurves.variability_output import write_variability_summary

        computed = compute_metrics(lc)
        summary = variability_summary(computed)
        log.info(
            "Variability summary: %d sources  %d candidates (%.1f%%)  "
            "median_Vs=%.2f  median_eta=%.2f",
            summary["n_sources"],
            summary["n_candidates"],
            summary["fraction_variable"] * 100,
            summary["median_Vs"],
            summary["median_eta"],
        )
        for sid, row in computed.iterrows():
            if row["is_variable_candidate"]:
                log.warning(
                    "Variable candidate: source_id=%s  Vs=%.2f  eta=%.2f",
                    sid, row["Vs"], row["eta"],
                )
        csv_path = write_variability_summary(computed, out_dir=str(plots_dir.parent))
        print(f"Variability summary CSV: {csv_path}")
    except Exception as exc:
        log.warning("Variability metrics computation failed (non-fatal): %s", exc)


if __name__ == "__main__":
    main()
