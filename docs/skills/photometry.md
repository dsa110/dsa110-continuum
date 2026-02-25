# Photometry Skill

Expert guidance for source photometry and flux measurement in the DSA-110 pipeline.

## Overview

Photometry is **core to ESE detection**: we measure fluxes of ~10⁴ compact sources daily and track variability over time. The pipeline uses adaptive channel binning to optimize SNR while preserving spectral information.

## Key Modules

| Module | Purpose |
|--------|---------|
| `core/photometry/adaptive_binning.py` | Adaptive channel binning for optimal SNR |
| `core/photometry/variability.py` | Variability metrics (η, V, σ-deviation) |
| `core/photometry/thresholds.py` | Detection threshold presets |
| `core/photometry/worker.py` | Parallel batch processing |
| `workflow/pipeline/stages/adaptive_photometry.py` | Pipeline stage |

## Adaptive Binning Photometry

The adaptive binning algorithm dynamically groups frequency channels to achieve a target SNR:

```python
from dsa110_contimg.core.photometry.adaptive_binning import AdaptiveBinningConfig

config = AdaptiveBinningConfig(
    target_snr=10.0,      # Target SNR per bin
    max_width=8,          # Maximum channels per bin
    min_channels=2,       # Minimum channels per bin
)
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_snr` | 10.0 | Target SNR threshold for adaptive binning |
| `max_width` | 8 | Maximum number of channels in a single bin |
| `min_channels` | 2 | Minimum channels required to form a valid bin |
| `imsize` | 256 | Image size for cutout photometry |
| `quality_tier` | "standard" | Imaging quality tier |

## Variability Metrics

The pipeline computes several variability metrics for ESE detection:

### η (Eta) Metric
```python
from dsa110_contimg.core.photometry.variability import calculate_eta_metric

# Weighted χ² per degree of freedom
eta = calculate_eta_metric(fluxes, errors, baseline_flux)
# ESE candidate if η > 3.0 (configurable)
```

**Interpretation**: η > 3 indicates flux variation exceeds measurement errors.

### V (Variability Index)
```python
from dsa110_contimg.core.photometry.variability import calculate_v_metric

# Fractional RMS variability
v = calculate_v_metric(fluxes)
# ESE candidate if V > 0.1 (10% variability)
```

### σ-Deviation
```python
from dsa110_contimg.core.photometry.variability import calculate_sigma_deviation

# Deviation from baseline in sigma
sigma_dev = calculate_sigma_deviation(
    current_flux=flux,
    current_error=error,
    baseline_flux=baseline,
    baseline_error=baseline_err,
)
# ESE candidate if |σ_dev| > 5
```

### M Metric (Two-Epoch)
```python
from dsa110_contimg.core.photometry.variability import calculate_m_metric

# Modulation index between two epochs
m = calculate_m_metric(flux_a=1.2, flux_b=0.8)  # Returns ~0.4
```

## Threshold Presets

Three preset levels for ESE candidate flagging:

```python
from dsa110_contimg.core.photometry.thresholds import get_threshold_preset

# Conservative (fewer false positives)
thresholds = get_threshold_preset("conservative")
# {'min_sigma': 5.0, 'min_chi2_nu': 4.0, 'min_eta': 3.0}

# Moderate (balanced)
thresholds = get_threshold_preset("moderate")
# {'min_sigma': 3.5, 'min_chi2_nu': 2.5, 'min_eta': 2.0}

# Sensitive (more candidates, more false positives)
thresholds = get_threshold_preset("sensitive")
# {'min_sigma': 2.5, 'min_chi2_nu': 1.5, 'min_eta': 1.0}
```

**Recommendation**: Use `conservative` for production, `sensitive` for candidate discovery.

## Pipeline Stage Usage

```python
from dsa110_contimg.common.unified_config import settings

# Configure photometry
settings.photometry.enabled = True
settings.photometry.target_snr = 10.0
settings.photometry.catalog = "nvss"  # or "vlass", "first"
settings.photometry.catalog_radius_deg = 2.0
settings.photometry.parallel = True
settings.photometry.max_workers = 4
```

### Stage Execution

```python
from dsa110_contimg.workflow.pipeline.stages import AdaptivePhotometryStage

stage = AdaptivePhotometryStage(config)
context = PipelineContext(
    config=config,
    outputs={"ms_path": "/data/calibrated.ms"}
)
result = stage.execute(context)

# Results include flux measurements for all sources
photometry_results = result.outputs["photometry_results"]
```

## Database Storage

Photometry measurements are stored in `pipeline.sqlite3`:

```sql
-- Query recent photometry for a source
SELECT mjd, flux_mjy, flux_err_mjy, spectral_index
FROM photometry
WHERE source_name = 'NVSS_J083000+550000'
ORDER BY mjd DESC
LIMIT 30;
```

## Performance Considerations

| Scenario | Parallel | Workers | Time (100 sources) |
|----------|----------|---------|-------------------|
| Single-threaded | No | 1 | ~30 min |
| Parallel | Yes | 4 | ~8 min |
| Parallel + GPU | Yes | 4 | ~5 min |

**Recommendations**:
- Enable `parallel=True` with `max_workers=4` for production
- Use `serialize_ms_access=True` if encountering table locking issues

## ESE Detection Workflow

1. **Daily Mosaics**: Create mosaic from 5-min tiles
2. **Source Extraction**: Get positions from NVSS/VLASS catalog
3. **Adaptive Photometry**: Measure flux at each position
4. **Variability Calculation**: Compute η, V, σ-deviation
5. **Threshold Check**: Flag sources exceeding preset thresholds
6. **Light Curve Update**: Append to `variable_source_lightcurves` table

## CLI Commands

```bash
# Run photometry on a measurement set
dsa110 photometry run /data/obs.ms --catalog nvss --parallel

# Generate source monitoring report
dsa110 photometry report --source-id NVSS_J083000+550000 --output report.html

# Check light curve for a source
dsa110 photometry lightcurve NVSS_J083000+550000 --days 30
```

## Related Resources

- Variable source detection: `.agent/skills/variable-source-detection/SKILL.md`
- Pipeline advisor: `.agent/skills/pipeline-advisor/SKILL.md`
- Photometry CLI: `backend/src/dsa110_contimg/core/photometry/cli.py`
