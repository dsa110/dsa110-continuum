# Crossmatch Skill

Expert guidance for source cross-matching in the DSA-110 pipeline.

## Overview

The crossmatch stage matches detected sources from images with reference catalogs (NVSS, FIRST, RACS). This enables source identification, astrometry correction, and flux scale validation.

## Key Modules

| Module | Purpose |
|--------|---------|
| `workflow/pipeline/stages/crossmatch.py` | `CrossMatchStage` pipeline stage |
| `core/catalog/crossmatch.py` | Core matching algorithms |
| `core/qa/catalog_validation.py` | Source extraction + validation |
| `core/catalog/coverage.py` | Catalog availability checks |

## Matching Methods

### 1. Nearest Neighbor (Default)

Finds the single closest catalog match within a search radius:

```python
from dsa110_contimg.core.catalog.crossmatch import nearest_neighbor_match

matches = nearest_neighbor_match(
    detected_sources=sources_df,
    catalog_sources=catalog_df,
    max_separation_arcsec=5.0,  # Default search radius
)
# Returns: DataFrame with matched pairs + separation
```

### 2. All Matches

Returns all catalog sources within search radius (for crowded fields):

```python
from dsa110_contimg.core.catalog.crossmatch import all_matches

matches = all_matches(
    detected_sources=sources_df,
    catalog_sources=catalog_df,
    max_separation_arcsec=10.0,
)
# Returns: DataFrame with all match pairs (may have duplicates)
```

### 3. Multi-Catalog Match

Matches against multiple catalogs simultaneously:

```python
from dsa110_contimg.core.catalog.crossmatch import multi_catalog_match

results = multi_catalog_match(
    detected_sources=sources_df,
    catalogs=["nvss", "first", "vlass"],
    catalog_dir=Path("/data/dsa110-contimg/state/catalogs"),
    max_separation_arcsec=5.0,
)
# Returns: Dict[catalog_name -> matches_df]
```

## Pipeline Stage Usage

```python
from dsa110_contimg.workflow.pipeline.stages import CrossMatchStage

stage = CrossMatchStage(config)
context = PipelineContext(
    config=config,
    outputs={
        "image_path": "/data/image.fits",
        "detected_sources": sources_df,  # From photometry
    }
)

result = stage.execute(context)
crossmatch_results = result.outputs["crossmatch_results"]
```

## Cross-Match Results Structure

```python
# crossmatch_results dict structure
{
    "matches": DataFrame,           # Matched source pairs
    "unmatched_detected": DataFrame,  # No catalog counterpart (new sources!)
    "unmatched_catalog": DataFrame,   # Not detected (faded sources!)
    "metrics": {
        "n_detected": 150,
        "n_matched": 130,
        "match_fraction": 0.87,
        "median_separation_arcsec": 1.2,
    },
    "astrometry": {
        "ra_offset_arcsec": 0.3,
        "dec_offset_arcsec": -0.2,
        "rms_offset_arcsec": 0.8,
    },
    "flux_scale": {
        "ratio_median": 1.02,
        "ratio_std": 0.05,
        "outlier_fraction": 0.03,
    },
}
```

## Astrometry Correction

```python
from dsa110_contimg.core.catalog.crossmatch import calculate_positional_offsets

offsets = calculate_positional_offsets(matches)
# Returns: {
#     "ra_offset_arcsec": 0.3,
#     "dec_offset_arcsec": -0.2,
#     "rms_arcsec": 0.8,
#     "n_sources": 130,
# }

# Apply correction to WCS
from dsa110_contimg.core.imaging.wcs_utils import apply_astrometry_correction

apply_astrometry_correction(
    fits_path=image_path,
    ra_offset_arcsec=offsets["ra_offset_arcsec"],
    dec_offset_arcsec=offsets["dec_offset_arcsec"],
)
```

## Flux Scale Correction

```python
from dsa110_contimg.core.catalog.crossmatch import calculate_flux_scale

flux_scale = calculate_flux_scale(
    matches,
    detected_flux_col="flux_jy",
    catalog_flux_col="nvss_flux_jy",
)
# Returns: {
#     "ratio_median": 1.02,  # Detected / Catalog
#     "ratio_std": 0.05,
#     "recommended_correction": 0.98,
# }
```

## Duplicate Detection

Identify sources matched to multiple catalog entries:

```python
from dsa110_contimg.core.catalog.crossmatch import identify_duplicate_catalog_sources

duplicates = identify_duplicate_catalog_sources(
    matches,
    separation_threshold_arcsec=3.0,
)
# Returns: DataFrame of potential duplicate sources
```

## ESE Science: New/Faded Sources

**Critical for ESE detection**: Unmatched sources indicate variable objects.

```python
# New sources (detected but no catalog match)
new_sources = crossmatch_results["unmatched_detected"]
for _, source in new_sources.iterrows():
    # Could be:
    # - Genuine new transient
    # - Variable source in high state
    # - Detection artifact
    if source["snr"] > 10:
        flag_for_followup(source)

# Faded sources (catalog but not detected)
faded_sources = crossmatch_results["unmatched_catalog"]
for _, source in faded_sources.iterrows():
    # Could be:
    # - Variable source in low state
    # - Scintillating source
    # - Catalog artifact
    if source["nvss_flux_jy"] > 0.01:  # >10 mJy expected
        flag_for_investigation(source)
```

## Configuration

```python
from dsa110_contimg.common.unified_config import settings

# Crossmatch settings
settings.crossmatch.enabled = True
settings.crossmatch.catalogs = ["nvss", "first"]
settings.crossmatch.max_separation_arcsec = 5.0
settings.crossmatch.min_snr = 5.0
settings.crossmatch.flux_scale_correction = True
settings.crossmatch.astrometry_correction = True
```

## CLI Commands

```bash
# Run crossmatch on image
dsa110 crossmatch /data/image.fits --catalog nvss

# Apply flux correction
dsa110 crossmatch /data/image.fits --apply-flux-scale

# Generate crossmatch report
dsa110 crossmatch /data/image.fits --report /data/reports/crossmatch.html
```

## Quality Thresholds

| Metric | Good | Warning | Fail |
|--------|------|---------|------|
| Match fraction | >80% | 60-80% | <60% |
| RMS offset | <2" | 2-5" | >5" |
| Flux ratio std | <10% | 10-20% | >20% |
| Outlier fraction | <5% | 5-10% | >10% |

## Database Storage

```sql
-- Crossmatch results stored in pipeline.sqlite3
CREATE TABLE crossmatch_results (
    id INTEGER PRIMARY KEY,
    image_id INTEGER REFERENCES images(id),
    catalog TEXT NOT NULL,  -- 'nvss', 'first', etc.
    n_detected INTEGER,
    n_matched INTEGER,
    match_fraction REAL,
    ra_offset_arcsec REAL,
    dec_offset_arcsec REAL,
    rms_offset_arcsec REAL,
    flux_ratio_median REAL,
    flux_ratio_std REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Low match rate | Wrong catalog | Check `validate_catalog_choice()` |
| Large offsets | Calibration error | Re-run calibration |
| Flux ratio ≠ 1 | Flux scale error | Apply flux correction |
| Many duplicates | Crowded field | Use smaller search radius |

## Related Resources

- Catalog setup skill: `.agent/skills/catalog-setup/SKILL.md`
- Validation skill: `.agent/skills/validation/SKILL.md`
- Variable source detection: `.agent/skills/variable-source-detection/SKILL.md`
