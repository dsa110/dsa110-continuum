# Variable Source Detection Skill

Expert guidance for detecting flux variability and ESE candidates in the DSA-110 pipeline.

## Overview

Variable source detection is the **final stage of ESE science**: comparing daily photometry against baseline catalogs to identify:
- **Variable sources** - Significant flux changes from baseline (ESE candidates)
- **New sources** - Sources not in baseline catalog (transients)
- **Fading sources** - Baseline sources below detection threshold

## Key Modules

| Module | Purpose |
|--------|---------|
| `core/catalog/variable_source_detection.py` | Detection algorithms |
| `workflow/pipeline/stages/variable_source_detection.py` | Pipeline stage |

## Detection Algorithm

```python
from dsa110_contimg.core.catalog.variable_source_detection import detect_variable_sources

new_sources, variable_sources, fading_sources = detect_variable_sources(
    observed_sources=detected_df,      # DataFrame from photometry
    baseline_sources=catalog_df,       # NVSS/VLASS baseline
    detection_threshold_sigma=5.0,     # For new source detection
    variability_threshold=3.0,         # σ for variable classification
    match_radius_arcsec=10.0,          # Cross-match radius
    baseline_catalog="unicat",         # Catalog name for logging
)
```

### Detection Categories

| Category | Criteria | ESE Relevance |
|----------|----------|---------------|
| **Variable** | Matched source with flux ratio > threshold | ⭐ Primary ESE candidates |
| **New** | Detected but no baseline match | Possible transient (rare) |
| **Fading** | Baseline source not detected | Possible ESE fading phase |

## Configuration

```python
from dsa110_contimg.common.unified_config import VariableSourceDetectionConfig

config = VariableSourceDetectionConfig(
    enabled=True,
    detection_threshold_sigma=5.0,      # For new source detection
    variability_threshold_sigma=3.0,    # For flux change detection
    match_radius_arcsec=10.0,           # Cross-match tolerance
    baseline_catalog="unicat",          # Baseline: nvss, first, racs, unicat
    alert_threshold_sigma=7.0,          # Trigger alert at this level
    store_lightcurves=True,             # Save to database
    min_baseline_flux_mjy=10.0,         # Min flux for fading detection
)
```

### Parameter Guidelines

| Parameter | Conservative | Moderate | Aggressive |
|-----------|-------------|----------|------------|
| `variability_threshold_sigma` | 5.0 | 3.0 | 2.0 |
| `detection_threshold_sigma` | 7.0 | 5.0 | 3.5 |
| `alert_threshold_sigma` | 10.0 | 7.0 | 5.0 |

**Production default**: Moderate settings balance sensitivity with false positive rate.

## Pipeline Stage Usage

```python
from dsa110_contimg.workflow.pipeline.stages import VariableSourceDetectionStage

stage = VariableSourceDetectionStage(config)
context = PipelineContext(
    config=config,
    outputs={"detected_sources": photometry_df}  # From AdaptivePhotometryStage
)

result = stage.execute(context)

# Access results
results = result.outputs["variable_source_results"]
print(f"New sources: {results['n_new']}")
print(f"Variable sources: {results['n_variable']}")
print(f"Fading sources: {results['n_fading']}")
print(f"Alerts generated: {len(results['alert_ids'])}")
```

## Database Tables

The detection system uses three database tables:

### `variable_source_candidates`
```sql
CREATE TABLE variable_source_candidates (
    id INTEGER PRIMARY KEY,
    source_name TEXT NOT NULL,
    ra_deg REAL NOT NULL,
    dec_deg REAL NOT NULL,
    detection_type TEXT NOT NULL,      -- 'new', 'variable', 'fading'
    flux_obs_mjy REAL NOT NULL,
    flux_baseline_mjy REAL,
    flux_ratio REAL,
    significance_sigma REAL NOT NULL,
    baseline_catalog TEXT,
    detected_at REAL NOT NULL,         -- MJD
    mosaic_id INTEGER,
    classification TEXT,               -- 'ese_candidate', 'agn', 'artifact'
    variability_index REAL
);
```

### `variable_source_alerts`
```sql
CREATE TABLE variable_source_alerts (
    id INTEGER PRIMARY KEY,
    candidate_id INTEGER NOT NULL,
    alert_level TEXT NOT NULL,         -- 'info', 'warning', 'critical'
    alert_message TEXT NOT NULL,
    created_at REAL NOT NULL,
    acknowledged BOOLEAN DEFAULT 0,
    follow_up_status TEXT              -- 'pending', 'vla_triggered', 'false_positive'
);
```

### `variable_source_lightcurves`
```sql
CREATE TABLE variable_source_lightcurves (
    id INTEGER PRIMARY KEY,
    candidate_id INTEGER NOT NULL,
    mjd REAL NOT NULL,
    flux_mjy REAL NOT NULL,
    flux_err_mjy REAL,
    frequency_ghz REAL NOT NULL
);
```

## ESE Identification Criteria

An ESE candidate should meet ALL of:

1. **Flux drop ≥2×** on ≲10 day timescale
2. **Coincident spectral-index evolution** (more pronounced at lower frequencies)
3. **Point source** (unresolved in NVSS/VLASS)
4. **Baseline flux >10 mJy** (sufficient SNR for monitoring)

### ESE Light Curve Pattern

```
Flux
  |    ___
  |   /   \
  | _/     \___  ← ESE (2× drop over ~week)
  |             \_____
  |_________________________ Time
       10 days
```

## Alert System

Alerts are generated when significance exceeds threshold:

```python
from dsa110_contimg.core.catalog.variable_source_detection import (
    generate_variable_source_alerts,
)

alert_ids = generate_variable_source_alerts(
    candidate_ids=[1, 2, 3],
    alert_threshold_sigma=7.0,
    db_path="/data/dsa110-contimg/state/db/pipeline.sqlite3",
)
```

### Alert Levels

| Level | Threshold | Action |
|-------|-----------|--------|
| `info` | 5-7σ | Log for review |
| `warning` | 7-10σ | Notify duty astronomer |
| `critical` | >10σ | Trigger VLA follow-up |

## Query Functions

```python
from dsa110_contimg.core.catalog.variable_source_detection import (
    get_variable_source_candidates,
    get_variable_source_alerts,
)

# Get recent candidates
candidates = get_variable_source_candidates(
    db_path=db_path,
    detection_type="variable",
    min_significance=5.0,
    limit=100,
)

# Get unacknowledged alerts
alerts = get_variable_source_alerts(
    db_path=db_path,
    acknowledged=False,
    alert_level="critical",
)
```

## Workflow Integration

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────────┐
│   Mosaic    │────▶│ AdaptivePhotom.  │────▶│ VariableSourceDetection │
└─────────────┘     └──────────────────┘     └─────────────────────────┘
                            │                           │
                            ▼                           ▼
                    [detected_sources]         [alerts, candidates]
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ VLA Follow-Up   │
                                               │ (if critical)   │
                                               └─────────────────┘
```

## Event System

Variable source detection emits events for downstream consumption:

```python
from dsa110_contimg.workflow.pipeline.events import EventType

# Events emitted:
# - VARIABLE_SOURCE_DETECTED: When any variable source is found
# - ESE_DETECTED: When ESE criteria are met (subset of variable)
```

## Related Resources

- Photometry skill: `.agent/skills/photometry/SKILL.md`
- Science context: `.agent/skills/pipeline-advisor/SCIENCE_CONTEXT.md`
- Light curve stage: `backend/src/dsa110_contimg/workflow/pipeline/stages/light_curve.py`
