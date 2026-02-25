# Mosaic Skill

Expert guidance for mosaic construction in the DSA-110 pipeline.

## Overview

The mosaic stage combines multiple 5-minute field tiles into larger mosaics, typically spanning 60 minutes (12 tiles). This creates the **daily mosaics** used for ESE detection via photometry.

## Key Modules

| Module | Purpose |
|--------|---------|
| `core/mosaic/streaming_mosaic.py` | `StreamingMosaicManager` for production |
| `core/mosaic/wsclean_mosaic.py` | WSClean-based mosaicking |
| `core/mosaic/qa.py` | Quality assessment (astrometry, photometry) |
| `core/mosaic/tiers.py` | Processing tier definitions |
| `core/mosaic/orchestrator.py` | Multi-mosaic orchestration |
| `workflow/pipeline/stages/mosaic.py` | Pipeline stage |

## DSA-110 Mosaic Architecture

```
  Time →
  ├────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┤
  │ t0 │ t1 │ t2 │ t3 │ t4 │ t5 │ t6 │ t7 │ t8 │ t9 │t10 │t11 │  Mosaic 1
  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
                                  ├────┬────┤
                                  │t10 │t11 │  Overlap (2 tiles)
                                  └────┴────┘
                                  ├────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┤
                                  │t10 │t11 │t12 │t13 │t14 │t15 │t16 │t17 │t18 │t19 │t20 │t21 │  Mosaic 2
                                  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
```

- Each tile: ~5 minutes of transit data
- Standard mosaic: 12 tiles (60 minutes)
- Overlap: 2-3 tiles for continuity (accounts for DSA-110 transit pattern)
- Daily coverage: ~24 mosaics

## Configuration

```python
from dsa110_contimg.common.unified_config import settings

# Mosaic configuration
settings.mosaic.min_images = 6           # Minimum tiles for valid mosaic
settings.mosaic.max_images = 16          # Maximum tiles per mosaic
settings.mosaic.overlap_tiles = 2        # Overlap between consecutive mosaics
settings.mosaic.weighting = "natural"    # Mosaic weighting scheme
settings.mosaic.trim_radius_deg = 3.5    # Trim to this radius from center
```

### MosaicConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_images` | 6 | Minimum tiles for valid mosaic |
| `max_images` | 16 | Maximum tiles per mosaic |
| `overlap_tiles` | 2 | Overlap with previous mosaic |
| `weighting` | "natural" | Pixel weighting: natural, weighted |
| `trim_radius_deg` | 3.5 | Trim mosaic edges |
| `qa_enabled` | True | Run quality assessment |

## Streaming Mosaic Manager

Production mosaics use `StreamingMosaicManager`:

```python
from dsa110_contimg.core.mosaic.streaming_mosaic import StreamingMosaicManager

manager = StreamingMosaicManager(
    output_dir=Path("/stage/mosaics"),
    tiles_per_mosaic=12,
    overlap_tiles=2,
    db_path=Path("/data/dsa110-contimg/state/db/pipeline.sqlite3"),
)

# Add tiles as they arrive
for tile_path in tile_paths:
    manager.add_tile(tile_path)
    
    # Check if ready to create mosaic
    if manager.ready_for_mosaic():
        mosaic_path = manager.create_mosaic()
```

## Pipeline Stage Usage

```python
from dsa110_contimg.workflow.pipeline.stages import MosaicStage

stage = MosaicStage(config)
context = PipelineContext(
    config=config,
    outputs={
        "image_paths": [
            "/data/img_t0.fits",
            "/data/img_t1.fits",
            # ... 12 tiles
        ],
        "ms_paths": [
            "/data/obs_t0.ms",
            "/data/obs_t1.ms",
            # ... corresponding MS files
        ]
    }
)

result = stage.execute(context)
mosaic_path = result.outputs["mosaic_path"]
mosaic_id = result.outputs["mosaic_id"]
```

## Quality Assessment

Each mosaic undergoes QA checks:

```python
from dsa110_contimg.core.mosaic.qa import (
    AstrometryResult,
    PhotometryResult,
    QAResult,
)

# QA result structure
qa_result = QAResult(
    astrometry=AstrometryResult(
        mean_offset_arcsec=1.2,
        rms_offset_arcsec=0.8,
        n_matches=150,
    ),
    photometry=PhotometryResult(
        flux_ratio_median=1.02,
        flux_ratio_std=0.05,
        n_sources=200,
    ),
    passed=True,
    issues=[],
)
```

### QA Thresholds

| Metric | Pass | Warning | Fail |
|--------|------|---------|------|
| Astrometry RMS | <2" | 2-5" | >5" |
| Photometry scatter | <5% | 5-10% | >10% |
| N sources | >100 | 50-100 | <50 |
| Edge artifacts | None | Minor | Major |

## Mosaic Tiers

Processing tiers balance speed vs quality:

```python
from dsa110_contimg.core.mosaic.tiers import MosaicTier

# Available tiers
MosaicTier.QUICK      # Minimal QA, fastest
MosaicTier.STANDARD   # Full QA, production default
MosaicTier.DEEP       # Extended cleaning, deep imaging
```

### Tier Settings

| Tier | Cleaning | QA | Astrometry | Use Case |
|------|----------|-----|------------|----------|
| QUICK | 1000 iter | Basic | Skip | Real-time check |
| STANDARD | 3000 iter | Full | NVSS match | Production |
| DEEP | 10000 iter | Full | Multi-catalog | Science analysis |

## WSClean Mosaicking

For advanced control, use WSClean directly:

```python
from dsa110_contimg.core.mosaic.wsclean_mosaic import (
    WSCleanMosaicConfig,
    create_wsclean_mosaic,
)

config = WSCleanMosaicConfig(
    output_path=Path("/stage/mosaic"),
    tile_paths=tile_paths,
    imsize=4096,
    cell_arcsec=6.0,
    primary_beam_model="dsa110",
)

result = create_wsclean_mosaic(config)
```

## Database Storage

Mosaics are registered in `pipeline.sqlite3`:

```sql
-- Query mosaics for a date range
SELECT id, created_at, n_tiles, qa_passed, path
FROM mosaics
WHERE created_at BETWEEN '2025-01-15' AND '2025-01-30'
ORDER BY created_at DESC;

-- Get tiles for a mosaic
SELECT tile_path, mjd, field_center
FROM mosaic_tiles
WHERE mosaic_id = 123;
```

## Primary Beam Handling

DSA-110 primary beam characteristics:

| Property | Value |
|----------|-------|
| FWHM | ~4° at 1.4 GHz |
| Model | Gaussian approximation |
| Correction | Applied per-tile |

```python
# Primary beam correction is applied during mosaic creation
params.pbcor = True
params.pblimit = 0.2  # Mask below 20% of peak response
```

## Noise Improvement

Mosaicking combines tiles for improved sensitivity:

```python
from dsa110_contimg.core.mosaic.qa import NoiseImprovementResult

# Expected noise improvement
# σ_mosaic = σ_tile / √(N_tiles × overlap_factor)
# For 12 tiles: ~3.5× improvement in noise
```

| N Tiles | Overlap | Theoretical Improvement | Typical Achieved |
|---------|---------|------------------------|------------------|
| 6 | 2 | 2.4× | 2.2× |
| 12 | 2 | 3.5× | 3.2× |
| 16 | 3 | 4.0× | 3.6× |

## CLI Commands

```bash
# Create mosaic from tile list
dsa110 mosaic create --tiles /data/tiles/*.fits --output /stage/mosaic.fits

# Run QA on existing mosaic
dsa110 mosaic qa /stage/mosaic.fits --catalog nvss

# List recent mosaics
dsa110 mosaic list --days 7
```

## Orchestration (Multi-Mosaic)

For batch processing multiple mosaics:

```python
from dsa110_contimg.core.mosaic.orchestrator import (
    MosaicOrchestrator,
    OrchestratorConfig,
)

config = OrchestratorConfig(
    tiles_per_mosaic=12,
    overlap_tiles=2,
    parallel_mosaics=2,
    qa_enabled=True,
)

orchestrator = MosaicOrchestrator(config)
orchestrator.process_day("2025-01-15")
```

## Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Discontinuities | Missing tiles | Increase overlap or skip gap |
| Poor astrometry | Bad calibration | Re-run with quality check |
| Artifacts at edges | Primary beam model | Tighter trim_radius_deg |
| Memory OOM | Too many tiles | Reduce tiles_per_mosaic |

## Performance

| Mosaic Size | Tiles | Time (CPU) | Time (GPU) |
|-------------|-------|------------|------------|
| Standard | 12 | ~15 min | ~5 min |
| Extended | 16 | ~20 min | ~7 min |
| Daily full | 24 mosaics | ~6 hours | ~2 hours |

## Related Resources

- Imaging skill: `.agent/skills/imaging/SKILL.md`
- Photometry skill: `.agent/skills/photometry/SKILL.md`
- GPU acceleration: `.agent/skills/gpu-acceleration/SKILL.md`
