# Catalog Setup Skill

Expert guidance for catalog database setup in the DSA-110 pipeline.

## Overview

The catalog setup stage ensures reference catalog databases (NVSS, FIRST, RACS) exist for the observation declination before processing begins. Since DSA-110 only slews in elevation and changes declination rarely, catalogs need updating when declination changes.

## Key Modules

| Module | Purpose |
|--------|---------|
| `workflow/pipeline/stages/catalog_setup.py` | `CatalogSetupStage` pipeline stage |
| `core/catalog/builder.py` | Catalog database builder |
| `core/catalog/coverage.py` | Declination coverage validation |
| `infrastructure/database/unified.py` | Pipeline database access |

## Declination-Based Architecture

DSA-110 observes a fixed declination strip at any given time:

```
   DSA-110 Transit Observations
   ─────────────────────────────────
   Declination strip: +42.0° ± 2°
   
   Catalogs needed:
   ├── nvss_dec+42.0.sqlite3     (NVSS 1.4 GHz)
   ├── first_dec+42.0.sqlite3    (FIRST 1.4 GHz)
   └── racs_dec+42.0.sqlite3     (RACS 888 MHz)
```

## Declination Extraction Priority

The stage extracts declination from three sources (in order):

1. **Database query** (fastest): `start_time/end_time` → query `hdf5_files` table
2. **HDF5 file** (moderate): `input_path` → read phase center from metadata
3. **MS file** (fallback): `ms_path` → read FIELD table

```python
# Priority 1: Database query (milliseconds)
from dsa110_contimg.infrastructure.database.unified import get_pipeline_db_path

db_path = get_pipeline_db_path()
cursor.execute("""
    SELECT dec_deg FROM hdf5_files
    WHERE timestamp_iso BETWEEN ? AND ?
    LIMIT 1
""", (start_time, end_time))

# Priority 2: HDF5 metadata (~100ms)
from dsa110_contimg.common.utils.fast_meta import FastMeta
import numpy as np

with FastMeta(hdf5_path) as meta:
    dec_deg = np.degrees(meta.phase_center_app_dec)  # radians to degrees

# Priority 3: MS FIELD table (~1s)
import casacore.tables as ct

with ct.table(str(ms_path) + "/FIELD") as field_table:
    phase_dir = field_table.getcol("PHASE_DIR")
    dec_rad = phase_dir[0, 0, 1]
    dec_deg = np.degrees(dec_rad)
```

## Catalog Types

| Catalog | Frequency | Coverage | Source Density |
|---------|-----------|----------|----------------|
| NVSS | 1.4 GHz | δ > -40° | ~2 sources/sq arcmin |
| FIRST | 1.4 GHz | RA limited | ~7 sources/sq arcmin |
| RACS | 888 MHz | δ < +41° | ~3 sources/sq arcmin |
| VLASS | 3 GHz | δ > -40° | ~5 sources/sq arcmin |

## Catalog Database Location

All catalogs stored in `/data/dsa110-contimg/state/catalogs/`:

```bash
state/catalogs/
├── nvss_dec+42.0.sqlite3
├── first_dec+42.0.sqlite3
├── racs_dec+42.0.sqlite3
├── vla_calibrators.sqlite3
└── sources/              # Raw source files
    └── vlacalibrators.txt
```

## Pipeline Stage Usage

```python
from dsa110_contimg.workflow.pipeline.stages import CatalogSetupStage

stage = CatalogSetupStage(config)
context = PipelineContext(
    config=config,
    inputs={
        "start_time": "2025-01-15T00:00:00",
        "end_time": "2025-01-15T23:59:59",
    }
)

result = stage.execute(context)
status = result.outputs["catalog_setup_status"]
# Status: "completed", "skipped_no_dec", "skipped_error"
```

## Building Catalogs

```python
from dsa110_contimg.core.catalog.builder import (
    build_nvss_catalog,
    build_first_catalog,
    build_racs_catalog,
)

# Build NVSS for declination +42°
build_nvss_catalog(
    dec_center=42.0,
    dec_width=4.0,  # ±2° strip
    output_path=Path("/data/dsa110-contimg/state/catalogs/nvss_dec+42.0.sqlite3"),
)
```

## Coverage Validation

```python
from dsa110_contimg.core.catalog.coverage import (
    validate_catalog_choice,
    get_available_catalogs,
)

# Check which catalogs cover a position
available = get_available_catalogs(
    ra_deg=180.0,
    dec_deg=42.0,
    catalog_dir=Path("/data/dsa110-contimg/state/catalogs/"),
)
# Returns: ["nvss", "first", "vlass"]

# Validate catalog for observation
is_valid, reason = validate_catalog_choice(
    catalog="nvss",
    dec_deg=42.0,
    catalog_dir=catalog_dir,
)
```

## Database Schema

```sql
-- Catalog source table (same schema for all catalogs)
CREATE TABLE sources (
    id INTEGER PRIMARY KEY,
    source_name TEXT NOT NULL,
    ra_deg REAL NOT NULL,
    dec_deg REAL NOT NULL,
    flux_jy REAL NOT NULL,
    flux_err_jy REAL,
    major_arcsec REAL,
    minor_arcsec REAL,
    pa_deg REAL,
    
    -- Spatial index for cone searches
    CONSTRAINT valid_coords CHECK (
        ra_deg >= 0 AND ra_deg < 360 AND
        dec_deg >= -90 AND dec_deg <= 90
    )
);

-- R*tree spatial index
CREATE VIRTUAL TABLE sources_rtree USING rtree(
    id,
    min_ra, max_ra,
    min_dec, max_dec
);
```

## Cone Search

```python
from dsa110_contimg.core.catalog.query import cone_search

# Find sources within 1 degree of position
sources = cone_search(
    catalog_path=Path("/data/dsa110-contimg/state/catalogs/nvss_dec+42.0.sqlite3"),
    ra_deg=180.0,
    dec_deg=42.0,
    radius_deg=1.0,
    min_flux_jy=0.001,  # 1 mJy threshold
)
# Returns: DataFrame with source_name, ra_deg, dec_deg, flux_jy
```

## CLI Commands

```bash
# Check catalog status for a declination
dsa110 catalog status --dec 42.0

# Build missing catalogs
dsa110 catalog build --dec 42.0 --catalogs nvss,first

# Verify catalog integrity
dsa110 catalog verify --dec 42.0
```

## Configuration

```python
from dsa110_contimg.common.unified_config import settings

# Catalog settings (in config/pipeline.yaml)
settings.catalogs.catalog_dir = "/data/dsa110-contimg/state/catalogs"
settings.catalogs.declination_width_deg = 4.0
settings.catalogs.preferred_catalogs = ["nvss", "first", "vlass"]
settings.catalogs.auto_build = True  # Auto-build missing catalogs
```

## Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| No catalog for dec | First observation at new declination | Run `dsa110 catalog build` |
| Slow cone search | Large catalog | Add spatial index |
| Missing sources | Flux threshold too high | Lower `min_flux_jy` |
| Empty catalog | Download failed | Re-download source file |

## Related Resources

- Crossmatch skill: `.agent/skills/crossmatch/SKILL.md`
- Validation skill: `.agent/skills/validation/SKILL.md`
