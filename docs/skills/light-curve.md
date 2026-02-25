# Light Curve Skill

Expert guidance for light curve computation in the DSA-110 pipeline.

## Overview

The light curve stage queries photometry measurements from the database and computes variability metrics (η, V, σ-deviation) for each source. This is the core analysis step for ESE detection.

## Key Modules

| Module | Purpose |
|--------|---------|
| `workflow/pipeline/stages/light_curve.py` | `LightCurveStage` pipeline stage |
| `core/photometry/variability.py` | Variability metric computation |
| `core/photometry/lightcurve_io.py` | Light curve I/O |
| `infrastructure/database/unified.py` | Database access |

## Variability Metrics

### η (Eta) - Weighted Variance

Variance-based metric accounting for measurement errors:

```python
# η = (1/N) Σ [(f_i - f̄)² / σ_i²]
# η > 1 indicates variability beyond measurement noise

eta = np.sum((fluxes - mean_flux)**2 / flux_errors**2) / n_epochs
```

### V - Coefficient of Variation

Fractional variability (std/mean):

```python
# V = σ / μ
# V > 0.1 (10%) typically indicates variable source

v_index = np.std(fluxes) / np.mean(fluxes)
```

### σ-Deviation - Maximum Deviation

Maximum excursion in units of standard deviation:

```python
# σ_dev = max(|f_i - f̄|) / σ
# Used for ESE detection (sudden dramatic changes)

sigma_dev = np.max(np.abs(fluxes - mean_flux)) / np.std(fluxes)
```

### χ²/ν - Reduced Chi-Squared

Test against constant flux model:

```python
# χ²/ν = (1/(N-1)) Σ [(f_i - f̄)² / σ_i²]
# χ²/ν >> 1 rejects constant flux hypothesis

chi2_reduced = np.sum((fluxes - mean_flux)**2 / flux_errors**2) / (n_epochs - 1)
```

## Pipeline Stage Usage

```python
from dsa110_contimg.workflow.pipeline.stages import LightCurveStage

stage = LightCurveStage(config)
context = PipelineContext(
    config=config,
    outputs={
        "mosaic_path": "/data/mosaics/mosaic_2025-01-01.fits",
        "source_ids": ["NVSS_J123456+420312"],  # Optional
    }
)

result = stage.execute(context)
variable_sources = result.outputs["variable_sources"]
ese_candidates = result.outputs["ese_candidates"]
```

## Operation Modes

### 1. Per-Mosaic Mode

Compute metrics for sources in a specific mosaic:

```python
context = PipelineContext(
    outputs={"mosaic_path": "/data/mosaics/mosaic_2025-01-15.fits"}
)
# Processes only sources detected in this mosaic
```

### 2. Source-Specific Mode

Compute metrics for specific sources:

```python
context = PipelineContext(
    outputs={"source_ids": ["NVSS_J123456+420312", "FIRST_J123500+420400"]}
)
# Processes only specified sources
```

### 3. Full Catalog Mode

Recompute metrics for all sources with sufficient epochs:

```python
context = PipelineContext(
    outputs={}  # Empty - processes all sources
)
# Warning: Can be slow for large databases
```

## Light Curve Query

```python
from dsa110_contimg.core.photometry.lightcurve_io import query_light_curve

# Get photometry history for a source
lightcurve = query_light_curve(
    source_id="NVSS_J123456+420312",
    db_path=Path("/data/dsa110-contimg/state/db/pipeline.sqlite3"),
    min_epochs=5,
)
# Returns: DataFrame with mjd, flux_jy, flux_err_jy, mosaic_id
```

## Computing Variability

```python
from dsa110_contimg.core.photometry.variability import compute_variability_metrics

metrics = compute_variability_metrics(
    mjd=lightcurve["mjd"],
    flux=lightcurve["flux_jy"],
    flux_err=lightcurve["flux_err_jy"],
)
# Returns: {
#     "eta": 2.5,
#     "v_index": 0.15,
#     "sigma_deviation": 3.2,
#     "chi2_reduced": 2.1,
#     "n_epochs": 15,
#     "mean_flux_jy": 0.045,
#     "std_flux_jy": 0.007,
# }
```

## ESE Detection Criteria

Extreme Scattering Events are identified by:

```python
# ESE detection thresholds (from config/evaluation_thresholds.yaml)
ESE_CRITERIA = {
    "sigma_deviation_min": 5.0,    # ≥5σ from mean
    "duration_days_min": 1.0,      # Lasts at least 1 day
    "duration_days_max": 365.0,    # Less than 1 year
    "symmetry_ratio": 0.5,         # Rise/fall asymmetry
}

def is_ese_candidate(metrics, lightcurve):
    if metrics["sigma_deviation"] < 5.0:
        return False
    if metrics["n_epochs"] < 10:
        return False  # Insufficient data
    # Check for characteristic ESE shape...
    return True
```

## Database Tables

```sql
-- Variability statistics (computed by light curve stage)
CREATE TABLE variability_stats (
    source_id TEXT PRIMARY KEY,
    mean_flux_jy REAL,
    std_flux_jy REAL,
    n_epochs INTEGER,
    eta REAL,
    v_index REAL,
    sigma_deviation REAL,
    chi2_reduced REAL,
    last_updated TIMESTAMP,
    is_variable BOOLEAN,
    is_ese_candidate BOOLEAN
);

-- Individual photometry measurements
CREATE TABLE photometry (
    id INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL,
    mosaic_id INTEGER REFERENCES mosaics(id),
    mjd REAL NOT NULL,
    flux_jy REAL NOT NULL,
    flux_err_jy REAL NOT NULL,
    snr REAL,
    created_at TIMESTAMP
);
```

## Light Curve Output

```python
# Save light curve to file
from dsa110_contimg.core.photometry.lightcurve_io import save_light_curve

save_light_curve(
    source_id="NVSS_J123456+420312",
    lightcurve=lightcurve_df,
    output_dir=Path("/data/dsa110-contimg/products/lightcurves"),
    format="csv",  # or "json", "fits"
)
```

## Configuration

```python
from dsa110_contimg.common.unified_config import settings

# Light curve settings
settings.lightcurve.min_epochs = 5
settings.lightcurve.eta_threshold = 1.5
settings.lightcurve.v_threshold = 0.1
settings.lightcurve.sigma_deviation_threshold = 3.0
settings.lightcurve.ese_sigma_threshold = 5.0
```

## CLI Commands

```bash
# Compute light curves for mosaic
dsa110 lightcurve compute --mosaic /data/mosaic.fits

# Recompute for specific source
dsa110 lightcurve compute --source NVSS_J123456+420312

# Export light curve
dsa110 lightcurve export NVSS_J123456+420312 --output /data/lc.csv

# Plot light curve
dsa110 lightcurve plot NVSS_J123456+420312 --output /data/lc.png
```

## Variability Classification

| Class | η | V | σ-dev | Interpretation |
|-------|---|---|-------|----------------|
| Stable | <1.2 | <5% | <2 | No variability |
| Low | 1.2-2 | 5-10% | 2-3 | Mild variability |
| Moderate | 2-5 | 10-20% | 3-4 | Clear variability |
| High | 5-10 | 20-50% | 4-5 | Strong variability |
| Extreme | >10 | >50% | >5 | ESE candidate |

## Performance Considerations

| N Sources | N Epochs | Time (estimate) |
|-----------|----------|-----------------|
| 1,000 | 10 | ~5 seconds |
| 10,000 | 30 | ~1 minute |
| 100,000 | 100 | ~15 minutes |

Use `--parallel` flag for large catalogs.

## Related Resources

- Photometry skill: `.agent/skills/photometry/SKILL.md`
- Variable source detection: `.agent/skills/variable-source-detection/SKILL.md`
- Mosaic skill: `.agent/skills/mosaic/SKILL.md`
