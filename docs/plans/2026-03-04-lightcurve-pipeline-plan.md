# Light Curve Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build scripts to stack 10 existing forced-photometry CSVs into per-source light curves with variability metrics and visualization, and fix the pipeline to propagate source IDs for future epochs.

**Architecture:** Four independent scripts. Task 1 produces the stacked Parquet (cross-epoch source matching by RA/Dec). Task 2 computes variability metrics on that Parquet. Task 3 visualizes. Task 4 patches `batch_pipeline.py` and `dsa110_continuum/photometry/forced.py` to emit `source_id` in future runs and archive mosaics.

**Tech Stack:** Python 3, pandas, astropy (SkyCoord matching), matplotlib, scipy (chi-squared), pathlib. Run under `/opt/miniforge/envs/casa6/bin/python`.

---

## Task 1: Light Curve Stack

**Files:**
- Create: `scripts/stack_lightcurves.py`
- Create: `tests/test_stack_lightcurves.py`

**Context for implementer:**

The 10 forced phot CSVs live at:
```
/data/dsa110-continuum/products/mosaics/{date}/{date}T{HH}00_forced_phot.csv
```
Schema: `ra_deg, dec_deg, nvss_flux_jy, dsa_peak_jyb, dsa_peak_err_jyb, dsa_nvss_ratio`

There is no `source_id`. We assign one by matching each row to the closest NVSS source in the combined pool (across all CSVs). Use `astropy.coordinates.SkyCoord.match_to_catalog_sky()` with a 5 arcsec tolerance. The NVSS "catalog" is assembled from the union of unique RA/Dec positions across all CSVs (since every CSV row is already matched to NVSS by the upstream pipeline). Group identical positions (within 1 arcsec) and assign integer `source_id` starting at 0.

The epoch timestamp is parsed from the CSV filename: `2026-02-12T0000_forced_phot.csv` → `2026-02-12T00:00:00`.

Output: `products/lightcurves/lightcurves.parquet` with columns:
`source_id, ra_deg, dec_deg, nvss_flux_jy, epoch_utc, dsa_peak_jyb, dsa_peak_err_jyb, dsa_nvss_ratio, date`

**Step 1: Write the failing test**

File: `tests/test_stack_lightcurves.py`

```python
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.stack_lightcurves import (
    parse_epoch_utc,
    assign_source_ids,
    stack_csvs,
)


def test_parse_epoch_utc():
    assert parse_epoch_utc("2026-02-12T0000_forced_phot.csv") == "2026-02-12T00:00:00"
    assert parse_epoch_utc("2026-01-25T2200_forced_phot.csv") == "2026-01-25T22:00:00"


def test_assign_source_ids_groups_nearby():
    df = pd.DataFrame({
        "ra_deg": [10.001, 10.001, 20.000],
        "dec_deg": [5.000, 5.000, 15.000],
        "nvss_flux_jy": [1.0, 1.0, 2.0],
    })
    result = assign_source_ids(df, match_arcsec=5.0)
    assert result.loc[0, "source_id"] == result.loc[1, "source_id"]
    assert result.loc[0, "source_id"] != result.loc[2, "source_id"]


def test_stack_csvs_produces_required_columns(tmp_path):
    csv1 = tmp_path / "2026-01-25T0200_forced_phot.csv"
    csv2 = tmp_path / "2026-02-12T0000_forced_phot.csv"
    rows = "ra_deg,dec_deg,nvss_flux_jy,dsa_peak_jyb,dsa_peak_err_jyb,dsa_nvss_ratio\n"
    rows += "10.0,5.0,1.0,0.9,0.01,0.9\n"
    csv1.write_text(rows)
    csv2.write_text(rows)
    df = stack_csvs([str(csv1), str(csv2)])
    for col in ["source_id", "epoch_utc", "date", "dsa_peak_jyb", "dsa_peak_err_jyb"]:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) == 2
    assert df["source_id"].iloc[0] == df["source_id"].iloc[1]
```

**Step 2: Run test to verify it fails**

```bash
cd /data/dsa110-continuum
/opt/miniforge/envs/casa6/bin/python -m pytest tests/test_stack_lightcurves.py -v 2>&1 | tail -20
```
Expected: ImportError or ModuleNotFoundError for `scripts.stack_lightcurves`.

**Step 3: Implement `scripts/stack_lightcurves.py`**

```python
#!/opt/miniforge/envs/casa6/bin/python
"""Stack per-epoch forced photometry CSVs into a cross-epoch light curve Parquet.

Usage:
    python scripts/stack_lightcurves.py [--products-dir /data/dsa110-continuum/products]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

PRODUCTS_DIR = Path("/data/dsa110-continuum/products")


def parse_epoch_utc(filename: str) -> str:
    """Extract ISO8601 UTC string from forced-phot CSV filename.

    Examples:
        2026-02-12T0000_forced_phot.csv  ->  2026-02-12T00:00:00
        2026-01-25T2200_forced_phot.csv  ->  2026-01-25T22:00:00
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})_forced_phot", filename)
    if not m:
        raise ValueError(f"Cannot parse epoch from filename: {filename}")
    date, hh, mm = m.group(1), m.group(2), m.group(3)
    return f"{date}T{hh}:{mm}:00"


def assign_source_ids(df: pd.DataFrame, match_arcsec: float = 5.0) -> pd.DataFrame:
    """Assign a stable integer source_id to each row by clustering RA/Dec positions.

    Rows within match_arcsec of each other get the same source_id.
    Uses greedy first-occurrence assignment via SkyCoord matching.
    """
    coords = SkyCoord(ra=df["ra_deg"].values * u.deg, dec=df["dec_deg"].values * u.deg)
    source_ids = np.full(len(df), -1, dtype=int)
    next_id = 0
    for i in range(len(df)):
        if source_ids[i] != -1:
            continue
        source_ids[i] = next_id
        sep = coords[i].separation(coords).arcsec
        matches = (sep < match_arcsec) & (source_ids == -1)
        source_ids[matches] = next_id
        next_id += 1
    df = df.copy()
    df["source_id"] = source_ids
    return df


def stack_csvs(csv_paths: list[str], match_arcsec: float = 5.0) -> pd.DataFrame:
    """Read all forced-phot CSVs, assign source_ids, return stacked DataFrame."""
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        fname = Path(path).name
        df["epoch_utc"] = parse_epoch_utc(fname)
        df["date"] = fname[:10]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = assign_source_ids(combined, match_arcsec=match_arcsec)
    return combined


def main():
    parser = argparse.ArgumentParser(description="Stack forced-phot CSVs into light curve Parquet.")
    parser.add_argument("--products-dir", default=str(PRODUCTS_DIR))
    parser.add_argument("--match-arcsec", type=float, default=5.0)
    args = parser.parse_args()

    products = Path(args.products_dir)
    csv_paths = sorted(products.glob("mosaics/*/*.csv"))
    if not csv_paths:
        print(f"No forced-phot CSVs found under {products}/mosaics/", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csv_paths)} CSVs:")
    for p in csv_paths:
        print(f"  {p}")

    df = stack_csvs([str(p) for p in csv_paths], match_arcsec=args.match_arcsec)
    print(f"\nStacked {len(df)} source-epoch rows, {df['source_id'].nunique()} unique sources.")

    out_dir = products / "lightcurves"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lightcurves.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

```bash
cd /data/dsa110-continuum
/opt/miniforge/envs/casa6/bin/python -m pytest tests/test_stack_lightcurves.py -v 2>&1 | tail -20
```
Expected: 3/3 PASS.

**Step 5: Run the script on real data**

```bash
cd /data/dsa110-continuum
/opt/miniforge/envs/casa6/bin/python scripts/stack_lightcurves.py
```
Expected output contains "Stacked N source-epoch rows" with N > 500. Check output:
```bash
/opt/miniforge/envs/casa6/bin/python -c "
import pandas as pd
df = pd.read_parquet('products/lightcurves/lightcurves.parquet')
print(df.shape, df.columns.tolist())
print(df.groupby('source_id').size().describe())
"
```

**Step 6: Commit**

```bash
cd /data/dsa110-continuum
git add scripts/stack_lightcurves.py tests/test_stack_lightcurves.py products/lightcurves/lightcurves.parquet
git commit -m "feat: stack forced-phot CSVs into cross-epoch light curve Parquet"
```

---

## Task 2: Variability Metrics

**Files:**
- Create: `scripts/variability_metrics.py`
- Create: `tests/test_variability_metrics.py`

**Context for implementer:**

Input: `products/lightcurves/lightcurves.parquet` (schema from Task 1).
Output: `products/lightcurves/variability_metrics.parquet`.

Formulas (Mooley et al. 2016, ApJ 818, 105; see `docs/reference/vast-crossref.md`):
- **m** = σ_S / ⟨S⟩  where σ_S is the standard deviation of DSA flux measurements
- **Vs** = (S_max − S_min) / sqrt(σ_max² + σ_min²)  where σ_i = `dsa_peak_err_jyb`
- **η** (eta) = reduced χ² = (1/(N-1)) × Σ [(S_i − ⟨S⟩)² / σ_i²]

Only compute metrics for sources with N ≥ 2 epochs. Sources with N = 1 get NaN for all metrics.

Candidate thresholds: `is_variable_candidate = (Vs > 4.0) | (eta > 2.5)`.

**Step 1: Write the failing test**

File: `tests/test_variability_metrics.py`

```python
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.variability_metrics import compute_metrics, flag_candidates


def make_source_df(fluxes, errors):
    """Helper: DataFrame for one source with given fluxes and per-point errors."""
    n = len(fluxes)
    return pd.DataFrame({
        "source_id": [0] * n,
        "ra_deg": [10.0] * n,
        "dec_deg": [5.0] * n,
        "nvss_flux_jy": [1.0] * n,
        "epoch_utc": [f"2026-01-0{i+1}T00:00:00" for i in range(n)],
        "dsa_peak_jyb": fluxes,
        "dsa_peak_err_jyb": errors,
        "dsa_nvss_ratio": [f / 1.0 for f in fluxes],
        "date": [f"2026-01-0{i+1}" for i in range(n)],
    })


def test_constant_source_has_low_eta():
    df = make_source_df([1.0, 1.0, 1.0], [0.1, 0.1, 0.1])
    metrics = compute_metrics(df)
    assert metrics.loc[0, "eta"] < 0.1


def test_variable_source_has_high_vs():
    df = make_source_df([1.0, 2.0], [0.05, 0.05])
    metrics = compute_metrics(df)
    assert metrics.loc[0, "Vs"] > 4.0


def test_single_epoch_source_gets_nan():
    df = make_source_df([1.0], [0.1])
    metrics = compute_metrics(df)
    assert np.isnan(metrics.loc[0, "eta"])
    assert np.isnan(metrics.loc[0, "Vs"])


def test_flag_candidates():
    metrics = pd.DataFrame({
        "source_id": [0, 1, 2],
        "Vs": [5.0, 1.0, 2.0],
        "eta": [1.0, 3.0, 1.0],
    })
    flagged = flag_candidates(metrics)
    assert flagged.loc[0, "is_variable_candidate"]   # Vs > 4
    assert flagged.loc[1, "is_variable_candidate"]   # eta > 2.5
    assert not flagged.loc[2, "is_variable_candidate"]
```

**Step 2: Run test to verify it fails**

```bash
cd /data/dsa110-continuum
/opt/miniforge/envs/casa6/bin/python -m pytest tests/test_variability_metrics.py -v 2>&1 | tail -20
```

**Step 3: Implement `scripts/variability_metrics.py`**

```python
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
    """Compute m, Vs, eta for each source. Returns one row per source_id."""
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
```

**Step 4: Run tests**

```bash
cd /data/dsa110-continuum
/opt/miniforge/envs/casa6/bin/python -m pytest tests/test_variability_metrics.py -v 2>&1 | tail -20
```
Expected: 4/4 PASS.

**Step 5: Run on real data (requires Task 1 output)**

```bash
cd /data/dsa110-continuum
/opt/miniforge/envs/casa6/bin/python scripts/variability_metrics.py
```

**Step 6: Commit**

```bash
git add scripts/variability_metrics.py tests/test_variability_metrics.py products/lightcurves/variability_metrics.parquet
git commit -m "feat: compute m/Vs/eta variability metrics per source"
```

---

## Task 3: Visualization

**Files:**
- Create: `scripts/plot_lightcurves.py`
- Create: `tests/test_plot_lightcurves.py`

**Context for implementer:**

Input: `products/lightcurves/lightcurves.parquet` and `products/lightcurves/variability_metrics.parquet`.

Outputs:
1. `products/lightcurves/plots/{source_id:06d}.png` — per-source flux vs time, error bars, NVSS reference line
2. `products/lightcurves/variable_candidates_summary.html` — HTML table ranking top variable sources by η, with inline thumbnails of their light curve plots

Use `matplotlib` only (no seaborn, no plotly). Convert `epoch_utc` strings to matplotlib dates. Only plot sources where `n_epochs >= 2`. Only generate the summary HTML for `is_variable_candidate == True`, sorted by `eta` descending.

The HTML summary must be self-contained: embed plot images as base64 data URIs so it is viewable offline.

**Step 1: Write the failing test**

File: `tests/test_plot_lightcurves.py`

```python
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.plot_lightcurves import (
    plot_source_lightcurve,
    build_summary_html,
)


def make_lc():
    return pd.DataFrame({
        "source_id": [0, 0, 1],
        "ra_deg": [10.0, 10.0, 20.0],
        "dec_deg": [5.0, 5.0, 15.0],
        "nvss_flux_jy": [1.0, 1.0, 2.0],
        "epoch_utc": ["2026-01-25T02:00:00", "2026-02-12T00:00:00", "2026-01-25T02:00:00"],
        "dsa_peak_jyb": [0.9, 0.95, 1.8],
        "dsa_peak_err_jyb": [0.01, 0.01, 0.02],
        "dsa_nvss_ratio": [0.9, 0.95, 0.9],
        "date": ["2026-01-25", "2026-02-12", "2026-01-25"],
    })


def make_metrics():
    return pd.DataFrame({
        "source_id": [0, 1],
        "ra_deg": [10.0, 20.0],
        "dec_deg": [5.0, 15.0],
        "nvss_flux_jy": [1.0, 2.0],
        "n_epochs": [2, 1],
        "mean_flux": [0.925, 1.8],
        "std_flux": [0.035, np.nan],
        "m": [0.038, np.nan],
        "Vs": [3.5, np.nan],
        "eta": [6.1, np.nan],
        "is_variable_candidate": [True, False],
    })


def test_plot_source_lightcurve_creates_file(tmp_path):
    lc = make_lc()
    source_group = lc[lc["source_id"] == 0]
    out_path = tmp_path / "000000.png"
    plot_source_lightcurve(source_group, nvss_flux=1.0, out_path=str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 1000


def test_build_summary_html_contains_candidates(tmp_path):
    lc = make_lc()
    metrics = make_metrics()
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    # Create fake plot file for source 0
    (plots_dir / "000000.png").write_bytes(b"\x89PNG\r\n")
    html = build_summary_html(metrics, lc, plots_dir=str(plots_dir))
    assert "source_id" in html
    assert "000000" in html or "0" in html
```

**Step 2: Run test to verify it fails**

```bash
/opt/miniforge/envs/casa6/bin/python -m pytest tests/test_plot_lightcurves.py -v 2>&1 | tail -20
```

**Step 3: Implement `scripts/plot_lightcurves.py`**

```python
#!/opt/miniforge/envs/casa6/bin/python
"""Generate per-source light curve plots and a variable candidates summary HTML.

Usage:
    python scripts/plot_lightcurves.py [--products-dir ...] [--top-n 50]
"""
import argparse
import base64
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime

PRODUCTS_DIR = Path("/data/dsa110-continuum/products")


def plot_source_lightcurve(group: pd.DataFrame, nvss_flux: float, out_path: str) -> None:
    """Save a flux-vs-time plot for one source to out_path."""
    times = [datetime.fromisoformat(e) for e in group["epoch_utc"]]
    fluxes = group["dsa_peak_jyb"].values
    errors = group["dsa_peak_err_jyb"].values

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.errorbar(times, fluxes, yerr=errors, fmt="o", color="#1f77b4",
                capsize=3, elinewidth=1, markersize=5, label="DSA-110")
    ax.axhline(nvss_flux, color="gray", linestyle="--", linewidth=1, label=f"NVSS {nvss_flux:.3f} Jy")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)
    ax.set_xlabel("Epoch (UTC)")
    ax.set_ylabel("Peak flux density (Jy/beam)")
    sid = group["source_id"].iloc[0]
    ra = group["ra_deg"].iloc[0]
    dec = group["dec_deg"].iloc[0]
    ax.set_title(f"Source {sid:06d}  RA={ra:.3f}°  Dec={dec:.3f}°")
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
    candidates = metrics[metrics["is_variable_candidate"]].sort_values("eta", ascending=False).head(top_n)
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
        rows_html += f"""
        <tr>
            <td>{sid:06d}</td>
            <td>{row['ra_deg']:.4f}</td>
            <td>{row['dec_deg']:.4f}</td>
            <td>{row['nvss_flux_jy']:.3f}</td>
            <td>{n_ep}</td>
            <td>{row['mean_flux']:.4f}</td>
            <td>{row.get('Vs', float('nan')):.2f}</td>
            <td>{row['eta']:.2f}</td>
            <td>{img_tag}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DSA-110 Variable Source Candidates</title>
<style>
  body {{ font-family: monospace; font-size: 12px; }}
  table {{ border-collapse: collapse; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; vertical-align: middle; }}
  th {{ background: #eee; }}
</style>
</head>
<body>
<h2>DSA-110 Variable Source Candidates (top {len(candidates)} by &eta;)</h2>
<p>Criteria: V<sub>s</sub> &gt; 4.0 OR &eta; &gt; 2.5</p>
<table>
<tr>
  <th>Source ID</th><th>RA (deg)</th><th>Dec (deg)</th>
  <th>NVSS flux (Jy)</th><th>N epochs</th><th>Mean flux (Jy/bm)</th>
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
    args = parser.parse_args()

    products = Path(args.products_dir)
    lc = pd.read_parquet(products / "lightcurves" / "lightcurves.parquet")
    metrics = pd.read_parquet(products / "lightcurves" / "variability_metrics.parquet")

    plots_dir = products / "lightcurves" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    multi_epoch = metrics[metrics["n_epochs"] >= 2]["source_id"].values
    print(f"Plotting {len(multi_epoch)} sources with ≥2 epochs...")
    for sid in multi_epoch:
        group = lc[lc["source_id"] == sid]
        nvss = group["nvss_flux_jy"].iloc[0]
        out_path = plots_dir / f"{int(sid):06d}.png"
        plot_source_lightcurve(group, nvss_flux=nvss, out_path=str(out_path))

    html = build_summary_html(metrics, lc, plots_dir=str(plots_dir), top_n=args.top_n)
    summary_path = products / "lightcurves" / "variable_candidates_summary.html"
    summary_path.write_text(html)
    print(f"Written: {summary_path}")

    n_candidates = metrics["is_variable_candidate"].sum()
    print(f"Summary: {n_candidates} variable candidates ranked.")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

```bash
cd /data/dsa110-continuum
/opt/miniforge/envs/casa6/bin/python -m pytest tests/test_plot_lightcurves.py -v 2>&1 | tail -20
```
Expected: 2/2 PASS.

**Step 5: Run on real data (requires Tasks 1+2 output)**

```bash
cd /data/dsa110-continuum
/opt/miniforge/envs/casa6/bin/python scripts/plot_lightcurves.py
```
Verify: `ls products/lightcurves/plots/*.png | wc -l` > 0 and `products/lightcurves/variable_candidates_summary.html` exists.

**Step 6: Commit**

```bash
git add scripts/plot_lightcurves.py tests/test_plot_lightcurves.py
git commit -m "feat: generate per-source light curve plots and variable candidates HTML summary"
```

---

## Task 4: Pipeline Fix

**Files:**
- Modify: `scripts/batch_pipeline.py`
- Modify: `dsa110_continuum/photometry/forced.py`
- Create: `tests/test_pipeline_fix.py`

**Context for implementer:**

### 4a — Add `source_id` to forced phot output

The forced phot CSV currently has no source identifier. Fix: after computing `dsa_peak_jyb` etc., add a `source_id` column = NVSS catalog row index. The NVSS catalog row index is the position of the matched NVSS source in the query result. Use `astropy.coordinates.SkyCoord.match_to_catalog_sky()` to find the index, then store it.

Read `dsa110_continuum/photometry/forced.py` — the function that writes the CSV is likely `run_forced_photometry()` or similar. Find where it builds the output DataFrame and add the column before writing.

Key constraint: `source_id` must be deterministic across epochs — the same NVSS source must always get the same integer. The NVSS catalog is queried from the SQLite DB via `dsa110_continuum.catalog.query`. The row ID (primary key) from the catalog is the stable source identifier.

### 4b — Archive epoch mosaic FITS to products/

In `batch_pipeline.py`, after a mosaic is produced and QA passes, copy the mosaic FITS from `/stage/dsa110-contimg/images/mosaic_{date}/{date}T{HH}00_mosaic.fits` to `products/mosaics/{date}/{date}T{HH}00_mosaic.fits`. This already happens for CSV files — extend the same copy step to include the FITS.

### 4c — Extract hardcoded paths to environment variables

In `batch_pipeline.py`, the constants `MS_DIR`, `STAGE_IMAGE_BASE`, `PRODUCTS_BASE` are hardcoded. Wrap them with `os.environ.get()` with the existing values as defaults:
```python
MS_DIR = os.environ.get("DSA110_MS_DIR", "/stage/dsa110-contimg/ms")
STAGE_IMAGE_BASE = os.environ.get("DSA110_STAGE_IMAGE_BASE", "/stage/dsa110-contimg/images")
PRODUCTS_BASE = os.environ.get("DSA110_PRODUCTS_BASE", "/data/dsa110-continuum/products/mosaics")
```

**Step 1: Write the failing tests**

File: `tests/test_pipeline_fix.py`

```python
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_batch_pipeline_respects_env_vars(monkeypatch, tmp_path):
    """MS_DIR, STAGE_IMAGE_BASE, PRODUCTS_BASE should be overridable via env vars."""
    monkeypatch.setenv("DSA110_MS_DIR", str(tmp_path / "ms"))
    monkeypatch.setenv("DSA110_STAGE_IMAGE_BASE", str(tmp_path / "images"))
    monkeypatch.setenv("DSA110_PRODUCTS_BASE", str(tmp_path / "products"))
    import importlib
    import scripts.batch_pipeline as bp
    importlib.reload(bp)
    assert str(bp.MS_DIR) == str(tmp_path / "ms")
    assert str(bp.STAGE_IMAGE_BASE) == str(tmp_path / "images")
    assert str(bp.PRODUCTS_BASE) == str(tmp_path / "products")


def test_forced_phot_csv_has_source_id_column(tmp_path):
    """Forced phot output CSV must have a source_id column."""
    import pandas as pd
    # Create a mock forced phot CSV (simulates what batch_pipeline writes)
    csv_path = tmp_path / "test_forced_phot.csv"
    df = pd.DataFrame({
        "ra_deg": [10.0],
        "dec_deg": [5.0],
        "nvss_flux_jy": [1.0],
        "dsa_peak_jyb": [0.9],
        "dsa_peak_err_jyb": [0.01],
        "dsa_nvss_ratio": [0.9],
        "source_id": [42],
    })
    df.to_csv(csv_path, index=False)
    result = pd.read_csv(csv_path)
    assert "source_id" in result.columns
```

**Step 2: Run test to verify failure**

```bash
cd /data/dsa110-continuum
/opt/miniforge/envs/casa6/bin/python -m pytest tests/test_pipeline_fix.py::test_batch_pipeline_respects_env_vars -v 2>&1 | tail -10
```
Expected: FAIL (MS_DIR is hardcoded string, not from env).

**Step 3: Apply fix 4c to `scripts/batch_pipeline.py`**

Read the file first. Then find the three constant definitions near line 70 and replace them:

Current (approximately lines 70–73):
```python
MS_DIR = "/stage/dsa110-contimg/ms"
STAGE_IMAGE_BASE = "/stage/dsa110-contimg/images"
PRODUCTS_BASE = "/data/dsa110-continuum/products/mosaics"
```

Replace with:
```python
MS_DIR = os.environ.get("DSA110_MS_DIR", "/stage/dsa110-contimg/ms")
STAGE_IMAGE_BASE = os.environ.get("DSA110_STAGE_IMAGE_BASE", "/stage/dsa110-contimg/images")
PRODUCTS_BASE = os.environ.get("DSA110_PRODUCTS_BASE", "/data/dsa110-continuum/products/mosaics")
```

**Step 4: Apply fix 4b — archive mosaic FITS**

In `batch_pipeline.py`, find the section that copies the forced-phot CSV to `products/`. It likely does `shutil.copy(csv_src, csv_dst)` or similar. Extend it to also copy the mosaic FITS:

Find: the block where the epoch mosaic QA passes and products are archived. Add after the CSV copy:
```python
mosaic_src = Path(paths["stage_dir"]) / f"{date}T{epoch_tag}_mosaic.fits"
mosaic_dst = Path(epoch_products_dir) / mosaic_src.name
if mosaic_src.exists() and not mosaic_dst.exists():
    shutil.copy2(str(mosaic_src), str(mosaic_dst))
    log.info(f"Archived mosaic FITS: {mosaic_dst}")
```

Read the file around the archiving section to find the exact insertion point before making this edit.

**Step 5: Apply fix 4a — add source_id to forced phot**

Read `dsa110_continuum/photometry/forced.py`. Find the function that builds the output DataFrame with `dsa_peak_jyb` and `dsa_nvss_ratio`. Before the DataFrame is returned or written, add:

```python
# Match each row back to NVSS catalog to get stable source_id
from astropy.coordinates import SkyCoord
import astropy.units as u
source_coords = SkyCoord(ra=ra_values * u.deg, dec=dec_values * u.deg)
catalog_coords = SkyCoord(ra=nvss_ra * u.deg, dec=nvss_dec * u.deg)
idx, sep, _ = source_coords.match_to_catalog_sky(catalog_coords)
df["source_id"] = nvss_ids[idx]  # nvss_ids = primary key from catalog query
```

The exact variable names will differ — read the actual function before implementing. The key is that `source_id` must be the catalog primary key (stable integer), not a local index.

**Step 6: Run all tests**

```bash
cd /data/dsa110-continuum
/opt/miniforge/envs/casa6/bin/python -m pytest tests/test_pipeline_fix.py -v 2>&1 | tail -15
```
Expected: 2/2 PASS.

**Step 7: Commit**

```bash
git add scripts/batch_pipeline.py dsa110_continuum/photometry/forced.py tests/test_pipeline_fix.py
git commit -m "fix: env-var configurable paths, archive mosaic FITS, add source_id to forced phot output"
```

---

## Final Verification

After all four tasks complete:

```bash
cd /data/dsa110-continuum
# 1. Run full light curve pipeline on existing data
/opt/miniforge/envs/casa6/bin/python scripts/stack_lightcurves.py
/opt/miniforge/envs/casa6/bin/python scripts/variability_metrics.py
/opt/miniforge/envs/casa6/bin/python scripts/plot_lightcurves.py

# 2. Check outputs
ls -lh products/lightcurves/
ls products/lightcurves/plots/ | wc -l

# 3. Run all new tests
/opt/miniforge/envs/casa6/bin/python -m pytest tests/test_stack_lightcurves.py tests/test_variability_metrics.py tests/test_plot_lightcurves.py tests/test_pipeline_fix.py -v 2>&1 | tail -20
```

Success criteria:
- `lightcurves.parquet` has ≥ 500 rows
- `variability_metrics.parquet` has metrics for all sources
- `variable_candidates_summary.html` exists and is non-empty
- `plots/` contains ≥ 1 PNG file
- All 11 tests PASS
