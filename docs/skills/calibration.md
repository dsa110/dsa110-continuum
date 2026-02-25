# Calibration Skill

Expert guidance for radio interferometer calibration in the DSA-110 pipeline.

## ⚠️ Output MS Naming: TWO Different Phaseshifts!

> **The calibration workflow creates TWO phaseshifted MS files - understand the difference!**

| MS Name | Purpose | Phase Center | Recommended for Imaging? |
|---------|---------|--------------|-------------------------|
| `*_cal.ms` | Solve calibration tables | Calibrator position | Use `*_meridian.ms` |
| `*_meridian.ms` | Science imaging | Median meridian | ✅ YES |

**Use `*_meridian.ms` for science imaging** so your field is centered. `*_cal.ms` is valid but image will be centered on calibrator.

See **Imaging Skill** (`imaging/SKILL.md`) for validated examples.

---

## Overview

The DSA-110 calibration system corrects for instrumental and atmospheric effects in visibility data. The pipeline uses CASA tasks wrapped in a safe execution environment, with optional GPU acceleration for gain application.

### Calibration Types

| Type | CASA Task | Purpose | Typical Cadence |
|------|-----------|---------|-----------------|
| **K (Delay)** | `gaincal` | Antenna delay corrections | Per-observation (**REQUIRED FIRST!**) |
| **BP (Bandpass)** | `bandpass` | Frequency-dependent gains | 24-hour cadence |
| **G (Gain)** | `gaincal` | Time-dependent amplitude/phase | Hourly |
| **Self-cal** | `gaincal` | Iterative improvement from sky model | Per-observation |

## DSA-110 Calibration Order

> **ℹ️ DSA-110 Note: Delay calibration (K) is NOT needed for routine observations!**
>
> DSA-110 has stable delays with <3° phase slope across the band (threshold: 30°).
> The default calibration preset correctly omits K-cal. See CASA_REFERENCE.md for empirical evidence.

### Standard DSA-110 Calibration Order (No K-Cal)

```
1. setjy() / populate_model_from_catalog()  - Establish flux model
2. gaincal(calmode='p')                     - Pre-BP phase (optional, improves BP)
3. bandpass(solnorm=True)                   - Frequency-dependent gains
4. gaincal(calmode='ap')                    - Amplitude + phase gains
5. applycal()                               - Apply all corrections
```

### Generic CASA Order (Arrays With Unstable Delays)

For arrays that need delay calibration (NOT DSA-110):

```
1. setjy()                    - Establish flux model
2. gaincal(gaintype='K')      - DELAY calibration FIRST
3. bandpass()                 - Frequency-dependent gains
4. gaincal(gaintype='G')      - Time-dependent amplitude/phase
5. applycal()                 - Apply all corrections
```

### DSA-110 K-Cal Diagnostic

```python
# Check if K-cal is needed (DSA-110: typically <3°, threshold 30°)
import numpy as np
from casacore.tables import table

with table(ms_path, readonly=True, ack=False) as tb:
    corr = tb.getcol('CORRECTED_DATA', nrow=10000)
    flag = tb.getcol('FLAG', nrow=10000)

phase_std = np.std(np.angle(corr[~flag], deg=True))
print(f"Phase std: {phase_std:.1f}°")  # DSA-110 typical: 25-30°
```

## ⚠️ CRITICAL: Bandpass ≠ Flux Calibration

> **Bandpass calibration is NORMALIZED and does NOT set absolute flux scale!**

### What Bandpass Does
- Corrects frequency-dependent gain **shape** (ripple, roll-off)
- Normalizes each antenna/polarization to mean amplitude ~1.0 (`solnorm=True` default)
- Equalizes relative gains across the band

### What Bandpass Does NOT Do
- Set absolute flux density scale
- Convert correlator units to Jy
- Apply known calibrator flux model

### For Absolute Flux Calibration

For flux calibrators (3C286, 3C48, 3C454.3, etc.), also run:

```python
from dsa110_contimg.core.calibration.casa_service import CASAService

service = CASAService()

# Option 1: setjy() for known flux calibrators (preferred)
service.setjy(
    vis="observation.ms",
    field="3C286",
    standard="Perley-Butler 2017",  # CASA flux model
)
# This sets MODEL_DATA to known flux values

# Then solve bandpass AGAINST the model
service.bandpass(
    vis="observation.ms",
    caltable="observation.BP",
    ...
)
```

### Verification: Did Flux Calibration Work?

```python
from casacore.tables import table
import numpy as np

with table(ms_path, readonly=True, ack=False) as tb:
    raw = tb.getcol('DATA', nrow=10000)
    corr = tb.getcol('CORRECTED_DATA', nrow=10000)
    flag = tb.getcol('FLAG', nrow=10000)
    
good = ~flag
ratio = np.mean(np.abs(corr[good])) / np.mean(np.abs(raw[good]))

# If ratio ≈ 1.0, calibration only normalized (no flux scale)
# If ratio >> 1 or << 1, flux scaling was applied
print(f"CORRECTED/RAW ratio: {ratio:.4f}")
assert ratio != 1.0, "Flux calibration may not have been applied!"
```
### Pipeline Flow - TWO PHASESHIFTS

> **⚠️ Critical: There are TWO different phaseshift operations!**

```
Original MS (24 fields at different meridian RAs)
        │
        ├──▶ [CALIBRATION PATH]
        │         ↓
        │    Split calibrator fields → Phaseshift to CALIBRATOR
        │         ↓
        │    Solve BP + gains on *_cal.ms
        │         ↓
        │    Cal tables (stored in database)
        │
        └──▶ [IMAGING PATH]
                  ↓
             Phaseshift to MEDIAN MERIDIAN (different from above!)
                  ↓
             Apply cal tables → *_meridian.ms
                  ↓
             Image *_meridian.ms
```

**Key Distinction**:
- `*_cal.ms`: Phaseshifted to **calibrator** position - for SOLVING cal tables only
- `*_meridian.ms`: Phaseshifted to **median meridian** - for IMAGING

Do NOT image `*_cal.ms` - it has the wrong phase center!

## Calibration Configuration

Use the default preset for all DSA-110 calibration (see [presets.py](backend/src/dsa110_contimg/core/calibration/presets.py)):

```python
from dsa110_contimg.core.calibration.presets import get_preset, DEFAULT_PRESET

# Get the default preset
preset = get_preset()  # Returns DEFAULT_PRESET

# Customize for specific observation
custom = preset.with_overrides(
    calibrator_name="3C286",
    refant="105",
)

# Convert to dict for pipeline stage
params = custom.to_dict()
```

### Default Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| K (delay) | ❌ | NOT needed for DSA-110 |
| BP (bandpass) | ✅ | Essential |
| Pre-BP phase | ✅ | Improves BP quality |
| G mode | `ap` | Full amplitude + phase |
| Adaptive flagging | ✅ | RFI mitigation |

### Key Parameters

```python
from dataclasses import fields
from dsa110_contimg.core.calibration.presets import CalibrationPreset

# View all parameters
for f in fields(CalibrationPreset):
    print(f"{f.name}: {f.type}")
```

**Core parameters:**
- `solve_delay`: Enable K (delay) calibration
- `solve_bandpass`: Enable BP calibration
- `solve_gains`: Enable G calibration
- `gain_calmode`: `"p"` (phase-only), `"a"` (amplitude), `"ap"` (both)
- `gain_solint`: Solution interval (`"inf"`, `"60s"`, etc.)
- `bp_minsnr`, `gain_minsnr`: Minimum SNR thresholds
- `refant`: Reference antenna (default: `"103"`)

## CASA Service

All CASA tasks are executed via `CASAService` for safe, logged execution:

```python
from dsa110_contimg.core.calibration.casa_service import CASAService

service = CASAService()

# Run gaincal
service.gaincal(
    vis="observation.ms",
    caltable="observation.G",
    field="0~23",
    refant="103",
    calmode="p",
    solint="inf",
    minsnr=3.0,
)

# Run bandpass
service.bandpass(
    vis="observation.ms",
    caltable="observation.BP",
    field="0~23",
    refant="103",
    combine="scan,field",
    solnorm=True,
    minsnr=5.0,
)

# Apply calibration
service.applycal(
    vis="observation.ms",
    field="",
    gaintable=["observation.BP", "observation.G"],
    interp=["nearest", "linear"],
)
```

### Process Isolation

For concurrent operations, enable CASA process isolation to avoid log file conflicts:

```bash
export DSA110_CASA_PROCESS_ISOLATION=true
```

Or in Python:
```python
service = CASAService(use_process_isolation=True)
```

## Calibrator Detection

The pipeline auto-detects calibrators in drift-scan observations:

```python
from dsa110_contimg.core.calibration.selection import select_bandpass_from_catalog

# Find calibrator in MS
field_sel, indices, wflux, cal_info, peak_idx = select_bandpass_from_catalog(
    "observation.ms",
    search_radius_deg=1.0,
)

name, ra_deg, dec_deg, flux_jy = cal_info
print(f"Calibrator: {name} at field {peak_idx}")
# Output: "Calibrator: 3C286 at field 17"
```

### Field Naming

After detection, rename the field to include calibrator name:

```python
from dsa110_contimg.core.calibration.field_naming import (
    rename_calibrator_field,
    rename_calibrator_fields_from_catalog,
)

# Auto-detect and rename all calibrators
rename_calibrator_fields_from_catalog("observation.ms")
# Result: Field 17 renamed from "meridian_icrs_t17" to "3C286_t17"

# Manual rename
rename_calibrator_field("observation.ms", "3C286", 17, include_time_suffix=True)
```

## VLA Calibrator Catalog

The VLA calibrator catalog is stored in SQLite:

```python
from dsa110_contimg.core.calibration.catalogs import load_vla_catalog

df = load_vla_catalog()
print(df.head())
#                 ra_deg    dec_deg  flux_jy
# 0134+329        24.42    32.912    26.2
# 3C286          202.78    30.509    15.0
```

**Catalog location**: `/data/dsa110-contimg/state/catalogs/vla_calibrators.sqlite3`

## Self-Calibration

Self-calibration iteratively improves calibration using the sky model from imaging:

```python
from dsa110_contimg.core.calibration.selfcal import (
    SelfCalConfig,
    selfcal_ms,
)

config = SelfCalConfig(
    max_iterations=5,
    phase_solints=["300s", "120s", "60s"],  # Start long, shorten progressively
    do_amplitude=True,
    amp_solint="inf",
    backend="wsclean",
    use_galvin_clip=True,  # Adaptive artifact suppression
)

result = selfcal_ms("observation.ms", config)
print(f"Status: {result.status}, Improvement: {result.improvement_factor:.1f}x")
```

### Self-Cal Workflow

1. **Initial imaging** → Create sky model
2. **Phase self-cal** (3 iterations):
   - `solint="300s"` → `solint="120s"` → `solint="60s"`
   - Phase-only (`calmode="p"`)
3. **Amplitude+phase self-cal** (2 iterations):
   - `solint="inf"` with `calmode="ap"`
4. **Convergence check**: SNR improvement <5% stops iteration

### Key Self-Cal Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 5 | Maximum total iterations |
| `phase_solints` | `["300s", "120s", "60s"]` | Progressive phase solution intervals |
| `phase_antenna_snr` | 3.0 | Per-antenna minimum SNR (phase) |
| `amp_antenna_snr` | 10.0 | Per-antenna minimum SNR (amplitude) |
| `use_galvin_clip` | True | Adaptive minimum clip for artifact suppression |
| `min_snr_improvement` | 1.05 | Stop if SNR improvement <5% |

## GPU-Accelerated Calibration

Gain application can be GPU-accelerated (~10× speedup on large datasets):

```python
from dsa110_contimg.core.calibration.gpu_calibration import (
    apply_gains,
    CUPY_AVAILABLE,
)

if CUPY_AVAILABLE:
    result = apply_gains(
        visibilities=vis_data,  # Complex array
        gains=gain_solutions,   # Complex gains
        antenna1=ant1,
        antenna2=ant2,
    )
    print(f"Processed {result.n_vis_calibrated} visibilities in {result.processing_time_s:.2f}s")
```

**Note**: GPU calibration requires CuPy and is only beneficial for >100k visibilities.

## Pipeline Stages

Calibration is organized into pipeline stages:

### CalibrationSolveStage

Solves for calibration solutions (K, BP, G):

```python
from dsa110_contimg.workflow.pipeline.stages import CalibrationSolveStage

stage = CalibrationSolveStage(config)
result = stage.run(context)
# Produces: K, BP, G calibration tables
```

### CalibrationApplyStage (CalibrationStage)

Applies calibration tables to target MS:

```python
from dsa110_contimg.workflow.pipeline.stages import CalibrationStage

stage = CalibrationStage(config)
result = stage.run(context)
# Applies BP, G tables to CORRECTED_DATA
```

### SelfCalibrationStage

Runs iterative self-calibration:

```python
from dsa110_contimg.workflow.pipeline.stages import SelfCalibrationStage

stage = SelfCalibrationStage(config)
result = stage.run(context)
# Iteratively improves calibration
```

### GainOnlyCalibrationStage

Optimized for mosaic tiles with existing BP tables:

```python
from dsa110_contimg.workflow.pipeline.stages.gain_only_calibration import (
    GainOnlyCalibrationStage,
)

stage = GainOnlyCalibrationStage(config)
context = PipelineContext(
    config=config,
    outputs={
        "ms_path": "/stage/science_field.ms",
        # BP table auto-discovered from registry if not provided
    },
)
result = stage.execute(context)

# Outputs:
# - gain_table_path: New gain table
# - sky_model_sources: Sources used in model
# - model_prediction_method: "gpu", "wsclean", or "setjy"
```

**Key features**:
- GPU-accelerated model prediction (30-60× faster than WSClean)
- Automatic BP table discovery via caltables registry
- Three-tier fallback: GPU → WSClean → single source
- Spectral index lookup from unified catalog

**See also**: [Gain-Only Calibration Skill](../gain-only-calibration/SKILL.md)

## Quality Assurance

Calibration tables are validated before use:

```python
from dsa110_contimg.core.calibration.validate import validate_caltables_for_use

is_valid, issues = validate_caltables_for_use(
    ["observation.BP", "observation.G"],
    ms_path="observation.ms",
)

if not is_valid:
    for issue in issues:
        print(f"⚠️ {issue}")
```

### QA Thresholds

| Metric | Critical | Warning |
|--------|----------|---------|
| Mean SNR | <3.0 | <10.0 |
| Flagged fraction | >50% | >20% |
| Minimum antennas | <10 | <50 |

## Best Practices

### 1. Always Use Presets

```python
# ✅ Good - use preset with overrides
params = get_preset("standard").with_overrides(calibrator_name="3C286").to_dict()

# ❌ Bad - manual 20+ parameter dict
params = {"field": "0~23", "refant": "103", ...}  # Easy to miss parameters
```

### 2. Progressive Self-Cal Solution Intervals

```python
# ✅ Good - start long, shorten progressively
config = SelfCalConfig(phase_solints=["300s", "120s", "60s"])

# ❌ Bad - start short (unstable bootstrap)
config = SelfCalConfig(phase_solints=["60s", "30s", "15s"])
```

### 3. Enable Process Isolation for Concurrent Operations

```bash
# In production with multiple parallel calibration jobs
export DSA110_CASA_PROCESS_ISOLATION=true
```

### 4. Use Galvin Clip for Self-Cal Imaging

```python
# Suppresses imaging artifacts during self-cal iterations
config = SelfCalConfig(use_galvin_clip=True, galvin_box_size=100)
```

## Key Files

| File | Purpose |
|------|---------|
| [calibration.py](backend/src/dsa110_contimg/core/calibration/calibration.py) | Core gaincal/bandpass routines |
| [presets.py](backend/src/dsa110_contimg/core/calibration/presets.py) | Calibration configuration presets |
| [casa_service.py](backend/src/dsa110_contimg/core/calibration/casa_service.py) | Safe CASA task execution |
| [selfcal.py](backend/src/dsa110_contimg/core/calibration/selfcal.py) | Self-calibration routines |
| [selection.py](backend/src/dsa110_contimg/core/calibration/selection.py) | Calibrator field selection |
| [field_naming.py](backend/src/dsa110_contimg/core/calibration/field_naming.py) | Field renaming utilities |
| [applycal.py](backend/src/dsa110_contimg/core/calibration/applycal.py) | Apply calibration to target MS |
| [gpu_calibration.py](backend/src/dsa110_contimg/core/calibration/gpu_calibration.py) | GPU-accelerated gain application |
| [catalogs.py](backend/src/dsa110_contimg/core/calibration/catalogs.py) | VLA calibrator catalog loading |

## CLI Reference

```bash
# List calibration presets
python -m dsa110_contimg.core.calibration.presets_cli list

# Show preset details
python -m dsa110_contimg.core.calibration.presets_cli show standard

# Show preset with overrides
python -m dsa110_contimg.core.calibration.presets_cli show standard refant=105
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DSA110_CASA_PROCESS_ISOLATION` | `false` | Run CASA tasks in isolated processes |
| `CASA_DATADIR` | (system) | CASA data directory for flux models |

## Troubleshooting

### CASA Log File Conflicts

**Symptom**: `RuntimeError: LogSink::postGlobally` when running concurrent calibration

**Solution**: Enable process isolation:
```bash
export DSA110_CASA_PROCESS_ISOLATION=true
```

### Low SNR Calibration

**Symptom**: Many flagged solutions, poor image quality

**Solution**: Use `low_snr` preset:
```python
params = get_preset("low_snr").with_overrides(calibrator_name="faint_cal").to_dict()
```

### Self-Cal Divergence

**Symptom**: SNR decreases during self-cal iterations

**Solution**: 
1. Start with longer solution intervals: `phase_solints=["600s", "300s", "120s"]`
2. Increase `min_snr_improvement` threshold
3. Check initial image quality before self-cal
