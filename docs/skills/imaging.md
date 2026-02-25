# Imaging Skill

Expert guidance for radio continuum imaging in the DSA-110 pipeline.

## ℹ️ NOTE: TWO Different Phaseshifts - Understand the Difference

> **DSA-110 has TWO phaseshift operations - both produce valid MS files.**
>
> Use `*_meridian.ms` for science imaging (centered on your field).

### The Two Phaseshifts

| Purpose | Mode | Target | MS Name | Recommended for Imaging? |
|---------|------|--------|---------|--------|
| **Calibration** | `mode="calibrator"` | Calibrator position | `*_cal.ms` | Use *_meridian.ms |
| **Imaging** | `mode="median_meridian"` | Median field center | `*_meridian.ms` | ✅ YES |

### Why Phaseshift to Median Meridian is Recommended

DSA-110 drift-scan observations create 24 fields over ~5 minutes, each at a **different** meridian phase center (RA spread ~1.2°). Phaseshifting to median meridian:

- Aligns all 24 fields to a common phase center
- Centers the output image on your science field
- Both *_cal.ms and *_meridian.ms are valid for imaging

### ⚠️ Common Mistake: Imaging `*_cal.ms`

The calibration pipeline creates `*_cal.ms` phaseshifted to the **calibrator position**.
This is for **solving calibration tables ONLY** - do NOT image it!

For imaging, create a **separate** MS phaseshifted to the **median meridian**.

### Correct Workflow

```python
from dsa110_contimg.core.calibration.runner import phaseshift_ms
from dsa110_contimg.core.calibration.applycal import apply_to_target
from dsa110_contimg.core.imaging.cli_imaging import image_ms

# 1. Create IMAGING MS (phaseshift to MEDIAN MERIDIAN, not calibrator!)
meridian_ms, _ = phaseshift_ms(
    ms_path="/stage/dsa110-contimg/ms/observation.ms",  # Original MS
    mode="median_meridian",  # NOT "calibrator"!
)

# 2. Apply calibration to the meridian MS
apply_to_target(
    ms_target=meridian_ms,
    gaintables=[bp_table, g_table],
)

# 3. IMAGE the median-meridian-phaseshifted MS
image_ms(
    ms_path=meridian_ms,  # NOT original, NOT *_cal.ms!
    imagename="/stage/output",
    ...
)
```

### Verification Before Imaging

```python
from casacore.tables import table
import numpy as np

# Check phase centers are identical (spread should be ~0)
with table(f"{ms_path}::FIELD", readonly=True, ack=False) as f:
    ra_spread = np.ptp(np.degrees(f.getcol('PHASE_DIR')[:, 0, 0]))
    assert ra_spread < 0.001, f"Phase centers not aligned! Run phaseshift_ms() first. Spread={ra_spread*3600:.1f} arcsec"
```

### ✅ Validated Example: What Happens When You Get It Wrong vs Right

**Validated 2026-02-02** on 3C454.3 observation (~10 Jy calibrator):

| Metric | ❌ Wrong: Imaged `*_cal.ms` | ✅ Correct: Imaged `*_meridian.ms` |
|--------|------------------------------|-----------------------------------|
| **Peak flux** | 22.4 Jy/beam (inflated!) | **8.88 Jy/beam** (correct) |
| **Source offset** | 5.99" | **0.00"** (exactly centered) |
| **Dynamic range** | 891:1 (fake, from artifacts) | 258:1 (real) |
| **First sidelobe** | 37% (elevated) | **22.8%** (clean) |

**Why the wrong approach looked "better"**: The artificially high peak and dynamic range
came from sidelobe structure adding constructively at the phase center. This is a classic
signature of imaging the wrong MS — results look impressive but are scientifically useless.

**Validation script**: `/data/dsa110-contimg/tests/manual/01d_image_one_ms.py`

---

## EveryBeam DSA-110 Beam Correction

### Overview

WSClean now includes **EveryBeam 0.8.0** with DSA-110 support (Airy beam pattern, added in v0.7.2).
This enables proper primary beam correction for science imaging and mosaics.

### Beam Correction Modes

| Flag | Mode | Description | Use Case |
|------|------|-------------|----------|
| `-apply-primary-beam` | Post-imaging | Divide by beam pattern after deconvolution | Single pointings |
| `-grid-with-beam` | A-projection | Apply beam during gridding (direction-dependent) | **Mosaics, wide-field** |
| `-reuse-primary-beam` | Cached | Load beam from previous run | Self-cal iterations |

### When to Use Each Mode

| Scenario | Recommended Mode | Why |
|----------|------------------|-----|
| Quick single-field imaging | `-apply-primary-beam` | Fast, adequate for point sources |
| Science mosaics | `-grid-with-beam` | Proper DD correction, flux accuracy |
| Self-calibration iterations | `-reuse-primary-beam` | 60-80% faster after first run |
| Photometric accuracy needed | `-grid-with-beam` | Beam-corrected flux in model |

### Example Commands

**Standard science imaging with beam correction:**
```bash
wsclean -name science \
    -size 4096 4096 -scale 3asec \
    -niter 100000 -auto-threshold 3 \
    -apply-primary-beam \
    -use-wgridder \
    observation_meridian.ms
```

**Wide-field imaging with direction-dependent correction:**
```bash
wsclean -name widefield \
    -size 4096 4096 -scale 3asec \
    -niter 100000 -auto-threshold 3 \
    -grid-with-beam \
    -use-wgridder \
    observation_meridian.ms
```

**GPU imaging with beam correction:**
```bash
wsclean -name gpu_beam \
    -size 4096 4096 -scale 3asec \
    -niter 100000 -auto-threshold 3 \
    -grid-with-beam \
    -use-idg -idg-mode gpu \
    observation_meridian.ms
```

**Self-cal with cached beam:**
```bash
# First iteration (computes beam)
wsclean -name selfcal_iter1 -grid-with-beam ...

# Subsequent iterations (reuses beam)
wsclean -name selfcal_iter2 -reuse-primary-beam -grid-with-beam ...
```

### Beam Caching

The wrapper auto-injects beam caching settings:

| Setting | Value | Purpose |
|---------|-------|---------|
| `-temp-dir` | `/dev/shm/wsclean-cache/` | RAM disk for fast I/O |
| `-beam-aterm-update` | 1800 sec | Conservative default for static beam (any value ≥ obs duration works) |

**Override if needed:**
```bash
wsclean -beam-aterm-update 300 ...  # More frequent update
wsclean -temp-dir /stage/tmp ...    # Different temp dir
```

### Output Products

With `-apply-primary-beam`:
- `output-image.fits` - Uncorrected image
- `output-image-pb.fits` - Primary beam pattern
- `output-image-pbcor.fits` - Beam-corrected image ✅

With `-grid-with-beam`:
- `output-image.fits` - Beam-corrected image ✅
- `output-image-beam.fits` - Average beam pattern

---

## Overview

The imaging stage produces continuum images from calibrated Measurement Sets. DSA-110 uses WSClean (default) or CASA tclean for deconvolution, with parameters optimized for point source detection.

## Key Modules

| Module | Purpose |
|--------|---------|
| `core/imaging/params.py` | `ImagingParams` dataclass |
| `core/imaging/worker.py` | WSClean/tclean execution |
| `core/imaging/cli_imaging.py` | `image_ms()` main function |
| `core/imaging/masks.py` | Mask generation |
| `workflow/pipeline/stages/imaging.py` | Pipeline stage |

## ImagingParams Dataclass

All imaging parameters are bundled in `ImagingParams`:

```python
from dsa110_contimg.core.imaging.params import ImagingParams

params = ImagingParams(
    imagename="output",
    imsize=2400,           # Image size in pixels
    cell_arcsec=6.0,       # Pixel size (arcsec)
    weighting="briggs",
    robust=0.5,
    niter=1000,
    threshold="0.005Jy",   # Cleaning threshold
    auto_mask=5.0,         # Auto-mask at 5σ
    auto_threshold=1.0,    # Stop at 1σ
    gridder="wgridder",    # or "idg" for GPU
    backend="wsclean",     # or "tclean"
    apply_primary_beam=True,  # EveryBeam beam correction
)
```

## Survey Mode Defaults

**Optimized for ESE detection** (point sources, speed, photometric repeatability):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `imsize` | 2400 | Covers primary beam |
| `cell_arcsec` | 6.0 | ~2 pixels per beam (12" synthesized beam) |
| `weighting` | briggs | Balanced resolution/sensitivity |
| `robust` | 0.5 | Standard for continuum |
| `deconvolver` | hogbom | Point sources only |
| `nterms` | 1 | Speed over spectral detail |
| `niter` | 1000 | Adequate for >10 mJy sources |
| `auto_mask` | 5.0 | Conservative masking |
| `auto_threshold` | 1.0 | Clean to noise level |
| `gridder` | wgridder | W-term aware (or idg for GPU) |
| `apply_primary_beam` | True | Beam-corrected output |

## Factory Method

```python
# Get survey-optimized parameters
params = ImagingParams.for_survey(
    imagename="/stage/output",
    imsize=2400,
)

# Customize gridder for GPU
params = ImagingParams.for_survey(
    imagename="/stage/output",
    gridder="idg",  # Use GPU via IDG
)

# With beam correction mode
params = ImagingParams.for_survey(
    imagename="/stage/output",
    grid_with_beam=True,  # Direction-dependent beam
)
```

## Backend Selection

### WSClean (Default)
```python
params = ImagingParams(
    imagename="output",
    backend="wsclean",
    gridder="wgridder",     # CPU w-term gridder
    apply_primary_beam=True,
)
```

### WSClean with GPU + Beam
```python
params = ImagingParams(
    imagename="output",
    backend="wsclean",
    gridder="idg",          # GPU Image Domain Gridder
    grid_with_beam=True,    # Direction-dependent beam
)
```

### CASA tclean
```python
params = ImagingParams(
    imagename="output",
    backend="tclean",
    gridder="wproject",     # CASA w-projection
    wprojplanes=128,        # Number of w-planes
)
```

### Gridder Comparison

| Gridder | Backend | W-Term | GPU | Beam | Speed | Memory |
|---------|---------|--------|-----|------|-------|--------|
| `wgridder` | WSClean | ✅ | ❌ | ✅ | Medium | Low |
| `idg` (gpu) | WSClean | ✅ | ✅ | ✅ | Fastest | High |
| `idg` (hybrid) | WSClean | ✅ | ✅ | ✅ | Fast | Medium |
| `wproject` | CASA | ✅ | ❌ | ❌ | Slow | Medium |
| `awproject` | CASA | ✅ + A-term | ❌ | ✅ | Very Slow | High |

**Recommendation**: Use `idg -idg-mode gpu` with GPU. Use `wgridder` when no GPU.
Both support EveryBeam beam correction.

## Image Size Guidelines

| Field of View | imsize | cell_arcsec | Pixels/beam |
|---------------|--------|-------------|-------------|
| Primary beam (4°) | 2400 | 6.0 | 2 |
| High resolution | 4096 | 3.0 | 4 |
| Full synthesis | 8192 | 1.5 | 8 |

**Formula**: `imsize = FoV_deg × 3600 / cell_arcsec`

## Auto-Masking

Auto-mask dynamically creates clean masks:

```python
params = ImagingParams(
    auto_mask=5.0,       # Mask formation threshold (σ)
    auto_threshold=1.0,  # Stop cleaning at this level (σ)
    mgain=0.8,           # Major cycle gain
)
```

**How it works**:
1. Minor cycle: Clean down to `auto_threshold × noise`
2. Major cycle: Update mask at `auto_mask × noise`
3. Iterate until niter reached or threshold met

## Catalog Seeding

Pre-seed clean model from known source positions:

```python
params = ImagingParams(
    imagename="output",
    use_unicat_mask=True,        # Use unified catalog for mask
    unicat_min_mjy=10.0,         # Only sources >10 mJy
    mask_radius_arcsec=60.0,     # Mask radius per source
)
```

## Galvin Adaptive Clipping

Suppress artifacts during self-cal with adaptive clipping:

```python
params = ImagingParams(
    imagename="output",
    galvin_clip_mask="/path/to/previous.fits",  # Previous iteration
    galvin_box_size=100,                         # Smoothing scale
    galvin_adaptive_depth=3,                     # Recursion depth
)
```

## Quality Tiers

| Tier | niter | Threshold | Use Case |
|------|-------|-----------|----------|
| `development` | 500 | 10 mJy | Quick tests |
| `standard` | 1000 | 5 mJy | Production |
| `high_precision` | 5000 | 1 mJy | Deep imaging |

```python
params = ImagingParams(
    imagename="output",
    quality_tier="standard",  # Auto-sets niter, threshold
)
```

## Pipeline Stage Usage

```python
from dsa110_contimg.workflow.pipeline.stages import ImagingStage

stage = ImagingStage(config)
context = PipelineContext(
    config=config,
    outputs={"ms_path": "/data/calibrated.ms"}
)

result = stage.execute(context)
image_path = result.outputs["image_path"]
```

## CLI Commands

```bash
# Image a measurement set
dsa110 image /data/obs.ms --output /stage/output --gridder idg

# With beam correction
dsa110 image /data/obs.ms --apply-primary-beam

# With specific parameters
dsa110 image /data/obs.ms --imsize 4096 --niter 5000 --robust 0.0

# Use GPU
dsa110 image /data/obs.ms --gridder idg --backend wsclean
```

## Direct WSClean Commands

```bash
# Standard imaging with beam correction (CPU)
wsclean -name output \
    -size 2400 2400 -scale 6asec \
    -niter 1000 -auto-mask 5 -auto-threshold 1 \
    -apply-primary-beam \
    -use-wgridder \
    observation_meridian.ms

# GPU imaging with direction-dependent beam
wsclean -name output \
    -size 4096 4096 -scale 3asec \
    -niter 10000 -auto-mask 5 -auto-threshold 1 \
    -grid-with-beam \
    -use-idg -idg-mode gpu \
    observation_meridian.ms

# Self-cal with cached beam (faster)
wsclean -name selfcal2 \
    -reuse-primary-beam \
    -grid-with-beam \
    -use-idg -idg-mode gpu \
    observation_meridian.ms
```

## Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| **Image centered on calibrator** | Using *_cal.ms | **Use `*_meridian.ms` for science imaging** |
| **Low flux (~100x)** | Missing flux calibration | Run `setjy()` or apply SEFD scaling |
| Empty image | Empty OBSERVATION table | Run `configure_ms_for_imaging()` first |
| Memory OOM | Image too large for RAM | Reduce imsize or set idg-mode=hybrid |
| Stripes in image | W-term not corrected | Use wgridder or idg (not standard gridder) |
| Slow imaging | CPU gridder | Use GPU with idg |
| **Beam not applied** | Missing EveryBeam | Check `wsclean --version` shows EveryBeam |
| **Unknown telescope** | Old EveryBeam | Need EveryBeam ≥0.7.2 for DSA-110 |

## Related Resources

- GPU acceleration: `.agent/skills/gpu-acceleration/SKILL.md`
- Pipeline advisor: `.agent/skills/pipeline-advisor/SKILL.md`
- Self-calibration: `.agent/skills/calibration/SKILL.md`
- Mosaic: `.agent/skills/mosaic/SKILL.md`
