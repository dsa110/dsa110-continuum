# Validation Skill

Expert guidance for image validation in the DSA-110 pipeline.

## Overview

The validation stage performs comprehensive quality checks on images including astrometry validation, flux scale validation, and source count completeness analysis. Optionally generates HTML reports with diagnostic plots.

## ⚠️ CRITICAL: Pre-Imaging MS Validation - Check the RIGHT MS!

> **Warning**: The pipeline creates TWO phaseshifted MS files:
> - `*_cal.ms` → Phaseshifted to **calibrator** (for solving only - DON'T image!)
> - `*_meridian.ms` → Phaseshifted to **median meridian** (for imaging - USE THIS!)

Before imaging, verify you're using the **meridian** MS, and that it's properly phaseshifted and calibrated:

```python
from casacore.tables import table
import numpy as np

def validate_ms_ready_for_imaging(ms_path: str) -> list[str]:
    """Check MS is ready for imaging. Returns list of issues."""
    issues = []
    
    # Check 0: Is this the right MS type?
    if "_cal.ms" in ms_path.lower():
        issues.append(
            "This is a CALIBRATION MS (*_cal.ms), not an imaging MS! "
            "Use *_meridian.ms for imaging instead."
        )
    
    # Check 1: Phase centers aligned (phaseshift was done)
    with table(f"{ms_path}::FIELD", readonly=True, ack=False) as f:
        phase_dir = f.getcol('PHASE_DIR')
        ra_spread = np.ptp(np.degrees(phase_dir[:, 0, 0]))
        if ra_spread > 0.001:  # More than ~3.6 arcsec
            issues.append(
                f"Phase centers NOT aligned (spread={ra_spread*3600:.1f} arcsec). "
                f"Run phaseshift_ms(mode='median_meridian') first!"
            )
    
    # Check 2: CORRECTED_DATA differs from DATA (calibration applied)
    with table(ms_path, readonly=True, ack=False) as tb:
        if 'CORRECTED_DATA' in tb.colnames():
            raw = tb.getcol('DATA', nrow=10000)
            corr = tb.getcol('CORRECTED_DATA', nrow=10000)
            flag = tb.getcol('FLAG', nrow=10000)
            good = ~flag
            ratio = np.mean(np.abs(corr[good])) / np.mean(np.abs(raw[good]))
            if 0.99 < ratio < 1.01:
                issues.append(
                    f"CORRECTED_DATA ≈ DATA (ratio={ratio:.4f}). "
                    f"Calibration may not have been applied or is normalized only."
                )
    
    return issues

# Usage
issues = validate_ms_ready_for_imaging("/stage/observation_meridian.ms")  # Note: _meridian.ms!
if issues:
    for issue in issues:
        print(f"⚠️ {issue}")
    raise RuntimeError("MS not ready for imaging!")
```

### Common Pre-Imaging Issues

| Issue | Symptom | Impact | Fix |
|-------|---------|--------|-----|
| **Imaging wrong MS** | Using `*_cal.ms` instead of `*_meridian.ms` | Image centered on calibrator | Use `*_meridian.ms` |
| Missing phaseshift | Phase centers spread >0.001° | Phase incoherence | `phaseshift_ms(mode="median_meridian")` |
| Missing flux cal | CORRECTED/DATA ratio ≈ 1.0 | ~100x flux error | `setjy()` + recalibrate |
| Unpopulated CORRECTED_DATA | All zeros or identical to DATA | No calibration | Re-run `applycal()` |

### ✅ Validated Example: Imaging Wrong MS vs Correct MS

**Validated 2026-02-02** on 3C454.3 observation (~10 Jy calibrator):

| Metric | ❌ Imaged `*_cal.ms` | ✅ Imaged `*_meridian.ms` |
|--------|---------------------|---------------------------|
| Peak flux | 22.4 Jy (inflated!) | 8.88 Jy (correct) |
| Source offset | 5.99" | 0.00" |
| Dynamic range | 891:1 (fake) | 258:1 (real) |

**Lesson**: The artificially high peak from the wrong MS was from sidelobe structure
adding constructively — scientifically useless despite looking "better."

**Validation script**: `tests/manual/01d_image_one_ms.py`

## Key Modules

| Module | Purpose |
|--------|---------|
| `workflow/pipeline/stages/validation.py` | `ValidationStage` pipeline stage |
| `core/qa/catalog_validation.py` | Core validation logic |
| `core/qa/astrometry_qa.py` | Astrometric accuracy checks |
| `core/qa/flux_qa.py` | Flux calibration checks |
| `core/qa/completeness.py` | Source count analysis |

## Validation Types

Three primary validation types:

| Type | Description | Key Metric |
|------|-------------|------------|
| `astrometry` | Positional accuracy | RMS offset < 2" |
| `flux` | Flux calibration | Ratio within 10% |
| `completeness` | Source detection | >80% match rate |

## Pipeline Stage Usage

```python
from dsa110_contimg.workflow.pipeline.stages import ValidationStage

stage = ValidationStage(config)
context = PipelineContext(
    config=config,
    outputs={"image_path": "/data/image.fits"}
)

result = stage.execute(context)
validation_results = result.outputs["validation_results"]
```

## Auto-Archive to HDD After Validation

After successful validation, MS files are automatically archived from SSD (`/stage/`) to HDD (`/data/stage/`) to free up fast storage for active processing.

### Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `validation.archive_to_hdd` | `True` | Enable/disable auto-archive after validation |

**Disable auto-archive:**
```python
config.validation.archive_to_hdd = False
```

Or in YAML:
```yaml
validation:
  archive_to_hdd: false
```

### What Gets Archived

MS files found in pipeline context outputs:
- `calibrated_ms_path` (`*_cal.ms`)
- `meridian_ms_path` (`*_meridian.ms`)
- `ms_path` (original MS)

**Only files on SSD staging** (`/stage/dsa110-contimg/`) are moved to archive.

### Manual Archive

```python
from dsa110_contimg.common.utils.paths import archive_ms_to_hdd
from pathlib import Path

archived_path = archive_ms_to_hdd(
    Path("/stage/dsa110-contimg/ms/2026-01-25T22:26:05.ms")
)
# Returns: Path("/data/stage/dsa110-contimg/ms/2026-01-25T22:26:05.ms")
```

### Storage Tiers

| Location | Type | Purpose |
|----------|------|--------|
| `/stage/dsa110-contimg/ms/` | SSD (1 TB) | Active processing |
| `/data/stage/dsa110-contimg/ms/` | HDD (13 TB) | Long-term archive |

## Validation Results Structure

```python
# validation_results dict
{
    "status": "passed",  # "passed", "warning", "failed"
    "astrometry": {
        "ra_offset_arcsec": 0.3,
        "dec_offset_arcsec": -0.2,
        "rms_arcsec": 0.8,
        "n_matches": 150,
        "status": "passed",
    },
    "flux": {
        "ratio_median": 1.02,
        "ratio_std": 0.05,
        "n_sources": 130,
        "status": "passed",
    },
    "completeness": {
        "detected_fraction": 0.87,
        "expected_sources": 150,
        "detected_sources": 130,
        "status": "passed",
    },
    "report_path": "/data/qa/reports/validation_2025-01-15.html",
}
```

## Full Validation

```python
from dsa110_contimg.core.qa.catalog_validation import run_full_validation

results = run_full_validation(
    fits_image="/data/image.fits",
    catalog="nvss",
    validation_types=["astrometry", "flux", "completeness"],
    output_dir=Path("/data/qa/reports"),
    generate_report=True,
)
```

## Astrometry Validation

```python
from dsa110_contimg.core.qa.astrometry_qa import validate_astrometry

astrometry_result = validate_astrometry(
    fits_image="/data/image.fits",
    reference_catalog="nvss",
    max_separation_arcsec=10.0,
    min_matches=50,
)
# Returns: {
#     "ra_offset_arcsec": 0.3,
#     "dec_offset_arcsec": -0.2,
#     "rms_arcsec": 0.8,
#     "n_matches": 150,
#     "outlier_fraction": 0.02,
# }
```

### Astrometry Thresholds

| Metric | Pass | Warning | Fail |
|--------|------|---------|------|
| RMS offset | <2" | 2-5" | >5" |
| Systematic offset | <1" | 1-3" | >3" |
| Outlier fraction | <5% | 5-10% | >10% |

## Flux Validation

```python
from dsa110_contimg.core.qa.flux_qa import validate_flux_scale

flux_result = validate_flux_scale(
    fits_image="/data/image.fits",
    reference_catalog="nvss",
    min_flux_jy=0.01,  # Only use sources >10 mJy
)
# Returns: {
#     "ratio_median": 1.02,
#     "ratio_std": 0.05,
#     "bootstrapped_error": 0.01,
#     "offset_percent": 2.0,
# }
```

### Flux Thresholds

| Metric | Pass | Warning | Fail |
|--------|------|---------|------|
| Flux ratio | 0.9-1.1 | 0.8-1.2 | Outside |
| Scatter | <10% | 10-20% | >20% |

## Completeness Validation

```python
from dsa110_contimg.core.qa.completeness import validate_completeness

completeness_result = validate_completeness(
    fits_image="/data/image.fits",
    reference_catalog="nvss",
    flux_bins=[0.01, 0.03, 0.1, 0.3, 1.0],  # Jy
)
# Returns: {
#     "overall_completeness": 0.87,
#     "by_flux_bin": {
#         "0.01-0.03 Jy": 0.75,
#         "0.03-0.1 Jy": 0.85,
#         "0.1-0.3 Jy": 0.92,
#         "0.3-1.0 Jy": 0.98,
#     },
#     "detection_limit_jy": 0.005,
# }
```

## HTML Report Generation

```python
from dsa110_contimg.core.qa.catalog_validation import generate_validation_report

report_path = generate_validation_report(
    results=validation_results,
    image_path="/data/image.fits",
    output_dir=Path("/data/qa/reports"),
    include_plots=True,
)
# Generates: validation_2025-01-15T12:00:00.html
```

### Report Contents

- Summary statistics table
- Astrometry offset scatter plot
- Flux ratio histogram
- Completeness curve (detected fraction vs flux)
- Source distribution map
- Flagged/rejected sources table

## Configuration

```python
from dsa110_contimg.common.unified_config import settings

# Validation settings
settings.validation.enabled = True
settings.validation.catalog = "nvss"
settings.validation.validation_types = ["astrometry", "flux", "completeness"]
settings.validation.min_snr = 5.0
settings.validation.generate_report = True
settings.validation.astrometry_rms_threshold = 2.0
settings.validation.flux_ratio_threshold = 0.1
settings.validation.completeness_threshold = 0.8
```

## CLI Commands

```bash
# Run full validation
dsa110 validate /data/image.fits --catalog nvss

# Astrometry only
dsa110 validate /data/image.fits --types astrometry

# Generate report
dsa110 validate /data/image.fits --report /data/qa/report.html

# Batch validation
dsa110 validate /data/images/*.fits --output-dir /data/qa/
```

## Image Requirements

Validation requires FITS format (prefers PB-corrected):

```python
# Priority order for image selection:
# 1. {name}.image.pbcor.fits  (PB-corrected)
# 2. {name}.image.fits        (Regular FITS)
# 3. {name}.fits              (Generic FITS)

# If only CASA .image format exists, skip validation
if not fits_image:
    logger.warning("Validation requires FITS format")
```

## Database Storage

```sql
-- Validation results stored in pipeline.sqlite3 
CREATE TABLE validation_results (
    id INTEGER PRIMARY KEY,
    image_id INTEGER REFERENCES images(id),
    validation_type TEXT,  -- 'astrometry', 'flux', 'completeness'
    status TEXT,           -- 'passed', 'warning', 'failed'
    metrics JSON,          -- Type-specific metrics
    report_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Low match count | Wrong catalog or threshold | Check catalog coverage |
| Large astrometry offset | Phase calibration error | Re-calibrate |
| Flux ratio ≠ 1 | Amplitude calibration | Apply flux correction |
| Low completeness | Shallow image or high noise | Check imaging depth |

## Integration with Pipeline

Validation results affect downstream stages:

```python
# In pipeline orchestrator
if validation_results["status"] == "failed":
    logger.error("Validation failed - flagging data products")
    flag_products_as_unreliable(image_id)
    
elif validation_results["status"] == "warning":
    logger.warning("Validation warnings - proceed with caution")
    add_qa_warning(image_id, validation_results)
```

## Related Resources

- Crossmatch skill: `.agent/skills/crossmatch/SKILL.md`
- Imaging skill: `.agent/skills/imaging/SKILL.md`
- Pre-imaging QA skill: `.agent/skills/pre-imaging-qa/SKILL.md`
