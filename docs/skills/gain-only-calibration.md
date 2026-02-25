# Gain-Only Calibration Skill

Expert guidance for gain-only calibration in the DSA-110 mosaic calibration workflow.

## Overview

The gain-only calibration stage is optimized for **mosaic observations** where a valid bandpass (BP) calibration table already exists. Instead of re-solving bandpass (which is stable over 24h), this stage:

1. **Builds a multi-source sky model** from the unified source catalog
2. **Populates MODEL_DATA** using GPU-accelerated prediction (with fallbacks)
3. **Reuses an existing BP table** (found via registry lookup or context)
4. **Solves gain-only calibration** (phase+amplitude, all 24 fields combined)

This enables efficient calibration of mosaic tiles without redundant BP solves.

## When to Use

| Scenario | Use Gain-Only? | Reason |
|----------|----------------|--------|
| Fresh observation with no BP | ❌ | Need full calibration |
| Mosaic tile after BP solve | ✅ | BP valid for ~24h |
| Calibrator transit detected | ❌ | Run full BP calibration |
| Science field (no calibrator) | ✅ | Multi-source sky model + BP reuse |

## Key Modules

| Module | Purpose |
|--------|---------|
| `pipeline/stages/gain_only_calibration.py` | `GainOnlyCalibrationStage` |
| `core/imaging/gpu_predict.py` | GPU-accelerated model prediction |
| `core/calibration/model.py` | Sky model source selection |
| `core/calibration/hardening.py` | BP table validity and lookup |

## Three-Tier Model Prediction

The stage uses a **three-tier fallback strategy** for populating MODEL_DATA:

### Tier 1: GPU Prediction (Default)

GPU-accelerated degridding with spectral index lookup.

```python
from dsa110_contimg.core.imaging.gpu_predict import predict_model_from_catalog

result = predict_model_from_catalog(
    ms_path="/stage/observation.ms",
    catalog_sources=sources,
    phase_center_ra=pointing_ra,
    phase_center_dec=pointing_dec,
    min_flux_jy=0.005,  # 5 mJy threshold
    max_sources=50,
    write_model=True,  # Writes to MODEL_DATA
)

if result.success:
    print(f"Predicted {result.n_sources} sources in {result.processing_time_s:.3f}s")
```

**Performance**: ~30-60× faster than WSClean `-predict` (~160ms for 1M visibilities)

**Requirements**:
- NVIDIA GPU with CUDA
- Spectral index data in master_sources.sqlite3 or VLA calibrators

### Tier 2: WSClean Prediction (Fallback)

CPU-based prediction using WSClean `-predict` mode.

```python
# Automatic fallback if GPU unavailable or fails
# Uses same source list, no spectral index required
```

**When used**:
- GPU not available (no CUDA, out of memory)
- Spectral index lookup fails for all sources
- GPU prediction returns error

### Tier 3: Single Brightest Source (Last Resort)

Simple single-source model using CASA `setjy`.

```python
# Automatic fallback if WSClean fails
# Uses brightest source from catalog only
```

**When used**:
- WSClean not available or fails
- Very simple/quick calibration needed

## Spectral Index Lookup Hierarchy

GPU prediction requires spectral indices for accurate model prediction:

```
1. Source dict 'alpha' column → Direct use
       ↓ (not found)
2. master_sources.sqlite3 → Query by position (0.5" crossmatch)
       ↓ (not found)
3. VLA calibrators → Compute from L/C-band fluxes
       ↓ (not found)
4. Skip source with warning
```

**Code path**:
```python
from dsa110_contimg.core.imaging.gpu_predict import CatalogSourceAdapter

# Adapter handles spectral index lookup
adapter = CatalogSourceAdapter(
    ra_deg=source["ra_deg"],
    dec_deg=source["dec_deg"],
    flux_jy=source["flux_mjy"] / 1000,
    spectral_index=source.get("alpha"),  # May be None
    source_dict=source,  # Full dict for fallback lookup
)

# If spectral_index is None, from_dict() will query databases
model = adapter.to_source_model()
```

## BP Table Discovery

When no BP table is provided in context, the stage **automatically searches** the caltables registry:

```python
from dsa110_contimg.core.calibration.hardening import (
    BP_VALIDITY_HOURS,  # Default: 24.0 hours
    get_active_applylist_bidirectional,
)

# Query for BP tables within ±24h of observation
selection = get_active_applylist_bidirectional(
    db_path=pipeline_db,
    target_mjd=observation_mjd,
    table_types=["BP"],
    validity_hours=BP_VALIDITY_HOURS,
    prefer_nearest=True,  # Select closest in time
)

# Returns closest valid BP table path
bp_table = selection.paths[0]  # e.g., "/stage/caltables/3C286_20260203.BP"
```

**Validity window**: ±24 hours centered on observation time

**Staleness warnings**: If using a BP table beyond ideal validity (but still within window), warnings are logged.

## Pipeline Stage Usage

```python
from dsa110_contimg.workflow.pipeline.stages.gain_only_calibration import (
    GainOnlyCalibrationStage,
)
from dsa110_contimg.workflow.pipeline.context import PipelineContext

# Initialize stage
stage = GainOnlyCalibrationStage(config)

# Create context with MS path
context = PipelineContext(
    config=config,
    outputs={
        "ms_path": "/stage/science_field_t05.ms",
        # BP table is OPTIONAL - stage will search registry if not provided
        # "bp_table_path": "/stage/caltables/3C286_20260203.BP",
    }
)

# Validate
is_valid, error = stage.validate(context)
if not is_valid:
    raise ValueError(f"Validation failed: {error}")

# Execute
result = stage.execute(context)

# Outputs
gain_table = result.outputs["gain_table_path"]  # New gain table
sky_model = result.outputs["sky_model_sources"]  # Sources used
method = result.outputs.get("model_prediction_method")  # "gpu", "wsclean", or "setjy"
```

## Dagster Integration

The stage is triggered by the **MosaicGainCalibrationSensor**:

```python
from dsa110_contimg.workflow.dagster.sensors.mosaic_sensors import (
    mosaic_gain_calibration_sensor,
)

# Sensor monitors for complete 12-tile mosaics
# When detected:
# 1. Selects center tile as calibration target
# 2. Triggers gain_only_calibration_job
# 3. Job runs GainOnlyCalibrationStage
```

**Sensor flow**:
```
12 tiles complete → MosaicGainCalibrationSensor triggers
                           ↓
                   Select center tile MS
                           ↓
                   gain_only_calibration_job
                           ↓
                   GainOnlyCalibrationStage.execute()
                           ↓
                   Gain table stored in registry
```

## Configuration

### PredictConfig

```python
from dsa110_contimg.core.imaging.gpu_predict import PredictConfig

config = PredictConfig(
    image_size=512,           # Image size in pixels
    cell_size_arcsec=12.0,    # Cell size
    phase_center_ra=None,     # Auto-detect from MS
    phase_center_dec=None,    # Auto-detect from MS
    w_planes=32,              # W-projection planes
    ref_freq_hz=1.4e9,        # Reference frequency
    max_gpu_gb=8.0,           # GPU memory limit
)
```

### Stage Parameters

```python
# Via context params
context = PipelineContext(
    config=config,
    outputs={
        "ms_path": ms_path,
    },
    params={
        "min_flux_jy": 0.005,    # 5 mJy threshold
        "max_sources": 50,       # Maximum sources in model
        "bp_validity_hours": 24, # BP table validity window
    }
)
```

## Metrics

The stage records metrics to the pipeline database:

| Metric | Description |
|--------|-------------|
| `model_prediction_method` | "gpu", "wsclean", or "setjy" |
| `model_prediction_time_s` | Time to populate MODEL_DATA |
| `n_sky_model_sources` | Number of sources in sky model |
| `bp_table_age_hours` | Age of reused BP table |

## Troubleshooting

### GPU Prediction Fails

**Symptoms**: Falls back to WSClean, logs "GPU prediction not available"

**Causes**:
- CUDA not available (`nvidia-smi` fails)
- GPU out of memory (try reducing `max_sources`)
- No spectral indices found for any source

**Solutions**:
```bash
# Check GPU status
nvidia-smi

# Check spectral index coverage
python -c "
from dsa110_contimg.core.imaging.gpu_predict import CatalogSourceAdapter
# Test spectral index lookup for a known source
"
```

### No Valid BP Table Found

**Symptoms**: Logs "No valid BP table found within ±24h"

**Causes**:
- No bandpass calibration run recently
- DB path incorrect
- BP tables not registered in caltables table

**Solutions**:
```bash
# Check caltables registry
sqlite3 /data/dsa110-contimg/state/db/pipeline.sqlite3 \
    "SELECT * FROM caltables WHERE table_type='BP' ORDER BY created_at DESC LIMIT 5"

# Run bandpass calibration job
dsa110 calibrate run --preset standard /stage/calibrator.ms
```

### Model Prediction Slow

**Symptoms**: Prediction takes >10s instead of <1s

**Causes**:
- Fell back to WSClean (check logs)
- Too many sources (>100)
- Database queries slow

**Solutions**:
- Ensure GPU is available
- Reduce `max_sources` parameter
- Check catalog database indexes

## Related Documentation

- [Calibration Skill](../calibration/SKILL.md) - Full calibration workflow
- [GPU Acceleration Skill](../gpu-acceleration/SKILL.md) - GPU configuration
- [Mosaic Skill](../mosaic/SKILL.md) - Mosaic construction
- [docs/how-to/mosaic-calibration-scheduler.md](../../docs/how-to/mosaic-calibration-scheduler.md) - Scheduler setup
