# Pre-Imaging QA Skill

Expert guidance for pre-imaging quality assessment in the DSA-110 pipeline.

## Overview

The pre-imaging QA stage runs between calibration and imaging to verify that the calibrated data has sufficient UV coverage for quality imaging. This catches problems before expensive imaging runs.

## ⚠️ CRITICAL: Phaseshift Validation - Two Different Phaseshifts!

**Before ANY imaging, you MUST verify the MS is phaseshifted to MEDIAN MERIDIAN!**

> **Warning**: The calibration pipeline creates `*_cal.ms` phaseshifted to the **calibrator** position.
> This is for **solving only** - do NOT image it!
> For imaging, you need `*_meridian.ms` phaseshifted to **median meridian**.

### The Two Phaseshifts (Don't Confuse!)

| MS | Phaseshifted To | RA Spread | Can Image? |
|----|-----------------|-----------|------------|
| `observation.ms` | (not phaseshifted) | ~4000 arcsec | Needs phaseshift |
| `*_cal.ms` | **Calibrator** position | ~0 arcsec | Valid (centered on calibrator) |
| `*_meridian.ms` | **Median meridian** | ~0 arcsec | ✅ RECOMMENDED |

DSA-110 drift-scan creates 24 fields with different phase centers (RA spread ~1.2°).
Phaseshift to a common center before imaging for best results.

### Quick Check Before Imaging

```python
import casacore.tables as ct
import numpy as np

def check_phaseshift_status(ms_path):
    """Check if MS has been phaseshifted for imaging."""
    with ct.table(str(ms_path) + "/FIELD", readonly=True) as field_table:
        ra_deg = np.rad2deg(field_table.getcol("PHASE_DIR")[:, 0, 0])
        n_fields = len(ra_deg)
        ra_spread = ra_deg.max() - ra_deg.min()
        
    # Calculate spread in arcsec
    ra_spread_arcsec = ra_spread * 3600
    
    print(f"Number of fields: {n_fields}")
    print(f"RA spread: {ra_spread_arcsec:.1f} arcsec")
    
    if ra_spread_arcsec > 10:  # More than 10 arcsec spread
        print("❌ NOT PHASESHIFTED - DO NOT IMAGE")
        print("   Run: phaseshift_ms(ms_path, mode='median_meridian')")
        return False
    else:
        print("✅ Phaseshifted - ready for imaging")
        # BUT ALSO CHECK: is this the MERIDIAN ms or the CAL ms?
        if "_cal.ms" in str(ms_path):
            print("⚠️  WARNING: This appears to be a calibration MS (*_cal.ms)")
            print("            For imaging, use *_meridian.ms instead!")
        return True

# RUN THIS BEFORE EVERY IMAGING OPERATION
is_ready = check_phaseshift_status("/path/to/observation_meridian.ms")
if not is_ready:
    raise RuntimeError("MS not phaseshifted - imaging will produce wrong flux")
```

### Why This Matters

| MS State | RA Spread | Coherent Addition | Expected Flux |
|----------|-----------|-------------------|---------------|
| NOT phaseshifted | ~4000+ arcsec | ~4% of data | **60× too low** |
| Phaseshifted | <1 arcsec | ~100% of data | Correct |

## Key Modules

| Module | Purpose |
|--------|---------|
| `workflow/pipeline/stages/pre_imaging_qa.py` | `PreImagingQAStage` pipeline stage |
| `core/visualization/uv_plots.py` | UV coverage plotting |
| `core/visualization/plot_manager.py` | Plot management |
| `core/qa/uv_coverage.py` | UV metrics computation |

## UV Coverage Assessment

Key metrics computed:

| Metric | Description | Good Threshold |
|--------|-------------|----------------|
| `coverage_score` | Fraction of UV plane covered | >0.3 |
| `baseline_count` | Number of valid baselines | >200 |
| `max_baseline_m` | Longest baseline | >10,000 m |
| `min_baseline_m` | Shortest baseline | <50 m |
| `uv_gap_fraction` | Fraction of UV plane with gaps | <0.4 |

## Pipeline Stage Usage

```python
from dsa110_contimg.workflow.pipeline.stages import PreImagingQAStage

stage = PreImagingQAStage(config)
context = PipelineContext(
    config=config,
    outputs={"ms_path": "/data/calibrated.ms"}
)

result = stage.execute(context)
uv_qa_passed = result.outputs["uv_qa_passed"]
coverage_score = result.outputs["uv_coverage_score"]
baseline_count = result.outputs["baseline_count"]
```

## UV Extraction

```python
from dsa110_contimg.core.visualization.uv_plots import extract_uv_from_ms

uv_data = extract_uv_from_ms(
    ms_path=Path("/data/calibrated.ms"),
    field_id=0,
    spw_id=0,
)
# Returns: {
#     "u_lambda": np.ndarray,  # U coordinates in wavelengths
#     "v_lambda": np.ndarray,  # V coordinates in wavelengths
#     "weights": np.ndarray,   # Data weights
#     "flags": np.ndarray,     # Flag status
# }
```

## Coverage Score Computation

```python
from dsa110_contimg.core.qa.uv_coverage import compute_coverage_score

score = compute_coverage_score(
    u_lambda=uv_data["u_lambda"],
    v_lambda=uv_data["v_lambda"],
    weights=uv_data["weights"],
    grid_size=512,  # Grid resolution
)
# Returns: float 0.0-1.0

# Score interpretation:
# >0.5: Excellent coverage
# 0.3-0.5: Good coverage  
# 0.2-0.3: Marginal coverage (imaging will have artifacts)
# <0.2: Poor coverage (imaging not recommended)
```

## UV Coverage Plots

```python
from dsa110_contimg.core.visualization.uv_plots import (
    plot_uv_coverage,
    plot_uv_density,
)

# Basic UV coverage plot
fig = plot_uv_coverage(
    u_lambda=uv_data["u_lambda"],
    v_lambda=uv_data["v_lambda"],
    title="UV Coverage",
)
fig.savefig("/data/qa/uv_coverage.png")

# UV density heatmap
fig = plot_uv_density(
    u_lambda=uv_data["u_lambda"],
    v_lambda=uv_data["v_lambda"],
    grid_size=256,
    colormap="viridis",
)
fig.savefig("/data/qa/uv_density.png")
```

## Baseline Analysis

```python
from dsa110_contimg.core.qa.uv_coverage import analyze_baselines

baseline_stats = analyze_baselines(
    u_lambda=uv_data["u_lambda"],
    v_lambda=uv_data["v_lambda"],
)
# Returns: {
#     "n_baselines": 253,
#     "max_baseline_m": 12500.0,
#     "min_baseline_m": 35.0,
#     "median_baseline_m": 2100.0,
#     "baseline_distribution": np.array,  # Histogram
# }
```

## QA Thresholds

From `config/health_checks.yaml`:

```yaml
pre_imaging_qa:
  coverage_score_min: 0.3
  baseline_count_min: 100
  max_flagged_fraction: 0.5
  generate_plots: true
  plots_dir: qa/pre_imaging
```

## Configuration

```python
from dsa110_contimg.common.unified_config import settings

# QA settings
settings.qa.generate_uv_coverage_plots = True
settings.qa.uv_coverage_threshold = 0.3
settings.qa.min_baselines = 100
```

## Stage Output Structure

```python
# Result context outputs
{
    "uv_qa_passed": True,           # Overall QA status
    "uv_coverage_score": 0.45,      # 0-1 coverage metric
    "baseline_count": 253,          # Valid baselines
    "uv_metrics": {                 # Detailed metrics
        "max_baseline_m": 12500.0,
        "min_baseline_m": 35.0,
        "flagged_fraction": 0.15,
        "n_fields": 24,
    },
    "uv_plot_path": "/data/qa/uv_coverage.png",  # If generated
}
```

## Warning Conditions

The stage warns but continues if:

```python
# Low coverage (flag but don't fail)
if coverage_score < 0.3:
    logger.warning(
        f"UV coverage score ({coverage_score:.2f}) below recommended "
        f"threshold (0.3). Imaging may produce poor quality results."
    )
```

## DSA-110 Specific Notes

- **Short baselines**: DSA-110 has minimum baseline ~35m due to antenna packing
- **Maximum baseline**: ~12.5 km East-West configuration
- **Transit mode**: UV coverage fills in as source transits (LST-dependent)
- **Declination effect**: Higher declinations have better hour-angle coverage

## Typical Coverage by Field Transit Time

| Transit Progress | Coverage Score | Quality |
|------------------|----------------|---------|
| 0-20% | <0.2 | Too early |
| 20-40% | 0.2-0.3 | Marginal |
| 40-60% | 0.3-0.5 | Good |
| 60-80% | 0.4-0.6 | Very good |
| 80-100% | 0.5-0.7 | Excellent |

## CLI Commands

```bash
# Run pre-imaging QA
dsa110 qa pre-imaging /data/calibrated.ms

# Generate UV plots only
dsa110 qa uv-plot /data/calibrated.ms --output /data/qa/

# Check coverage score
dsa110 qa uv-coverage /data/calibrated.ms
```

## Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| **Flux 40-100× too low** | **NOT PHASESHIFTED** | **Run `phaseshift_ms()` before imaging** |
| Low coverage | Early transit | Wait for more data |
| Many flagged | RFI or bad antenna | Check flagging |
| Missing baselines | Antenna offline | Check array status |
| UV gaps | Shadowing | Normal for compact config |
| RA spread >10 arcsec | Fields not combined | Phaseshift required |

## Integration with Imaging

**FIRST: Check phaseshift status before ANY imaging:**

```python
# MANDATORY pre-imaging checks
is_phaseshifted = check_phaseshift_status(ms_path)
if not is_phaseshifted:
    raise RuntimeError("STOP: MS not phaseshifted. Run phaseshift_ms() first.")
```

If `uv_qa_passed=False`, imaging stage should:

1. Log warning about expected quality
2. Consider using robust weighting (less sensitive to gaps)
3. Flag output images as "QA Warning"

```python
# In imaging stage
if not context.outputs.get("uv_qa_passed", True):
    logger.warning("Proceeding with imaging despite UV QA warning")
    # Use more robust settings
    imaging_params.robust = 0.0  # Natural weighting
```

## Related Resources

- Imaging skill: `.agent/skills/imaging/SKILL.md`
- Validation skill: `.agent/skills/validation/SKILL.md`
