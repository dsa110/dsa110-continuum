---
name: ms-generation
description: Generate CASA Measurement Sets from DSA-110 HDF5 subband visibility data. Covers the complete conversion workflow from raw telescope data to calibration-ready MS files.
license: MIT
---

# Measurement Set Generation from HDF5 Data

Convert DSA-110 UVH5 (HDF5) visibility data to CASA Measurement Sets for calibration and imaging.

> **⚠️ BEFORE RUNNING CONVERSION: CHECK FOR EXISTING MS FILES!**
>
> ```bash
> # Check what MS files already exist for your date
> ls /data/stage/dsa110-contimg/ms/YYYY-MM-DDT*.ms 2>/dev/null | wc -l
> ```
>
> **If MS files already exist, SKIP to calibration!** The pipeline is idempotent - you can start from any step. Do NOT regenerate MS files that already exist.

## Overview

The DSA-110 telescope produces **16 subband files per observation** (`*_sb00.hdf5` through `*_sb15.hdf5`). These must be grouped by timestamp and combined into a single Measurement Set.

```
16 UVH5 files (*_sb00.hdf5 ... *_sb15.hdf5)
    ↓
Group by timestamp (120s tolerance)
    ↓
Combine subbands (pyuvdata +=)
    ↓
Write Measurement Set (24 fields, each at own meridian RA)
    ↓
Configure for imaging (antenna positions, field names)
    ↓
⚠️ PHASESHIFT REQUIRED before imaging!
```

## ⚠️ CRITICAL: Output MS Has Per-Field Phase Centers

> **The converted MS has 24 fields, each at a DIFFERENT phase center!**
>
> You MUST run `phaseshift_ms()` before imaging to align all fields.

DSA-110 drift-scan observations create one field per ~12.88 seconds over ~5 minutes:

- **Field 0**: RA at observation start (e.g., 343.16°)
- **Field 23**: RA at observation end (e.g., 344.40°)
- **RA spread**: ~1.2° = 74 arcmin

### Why This Matters

Raw MS has 24 fields with different phase centers:

- Each field's visibilities have different phase gradients
- Phaseshift aligns all fields to a common center for coherent imaging
- \*\_meridian.ms is recommended for science imaging

### Next Step After Conversion - TWO SEPARATE PHASESHIFTS

The pipeline uses **two different phaseshifts** - don't confuse them!

| Step        | Mode                     | Target Position            | Output          | Purpose          |
| ----------- | ------------------------ | -------------------------- | --------------- | ---------------- |
| Calibration | `mode="calibrator"`      | Calibrator (e.g., 3C454.3) | `*_cal.ms`      | Solving BP/gains |
| **Imaging** | `mode="median_meridian"` | Median field center        | `*_meridian.ms` | **For imaging**  |

```python
from dsa110_contimg.core.calibration.runner import phaseshift_ms
from dsa110_contimg.core.calibration.applycal import apply_to_target

# [Already done by calibration] Phaseshift to calibrator → solve cal tables

# [For imaging] Create median-meridian phaseshifted MS
meridian_ms, _ = phaseshift_ms(
    ms_path="/stage/dsa110-contimg/ms/observation.ms",  # Original MS
    mode="median_meridian",  # NOT "calibrator"!
)

# Apply calibration to meridian MS
apply_to_target(meridian_ms, gaintables=[bp_table, g_table])

# Image the meridian MS (NOT *_cal.ms!)
```

> **⚠️ Do NOT image `*_cal.ms`** - it's phaseshifted to the calibrator position (for solving only).
> Image **`*_meridian.ms`** - phaseshifted to median meridian (for imaging).

## Critical Constraint: 16-Subband Architecture

> All 16 subbands share a bit-identical `time_array[0]` (Julian Date).
>
> Grouping uses **exact match** — no tolerance window needed. Always use `query_subband_groups()`.

## Prerequisites

1. **Activate the CASA environment**:

   ```bash
   conda activate /opt/miniforge/envs/casa6
   ```

2. **HDF5 files indexed in database**:
   - Pipeline database: `/data/dsa110-contimg/state/db/pipeline.sqlite3`
   - Table: `hdf5_files` with columns `path`, `timestamp_iso`, etc.

3. **Input data location**: `/data/incoming/` (raw HDF5 subband files)

4. **Output location**: `/stage/dsa110-contimg/ms/` (NVMe SSD for fast I/O)

---

## Method 1: CLI Command (Recommended for Users)

The `dsa110 convert` command handles grouping, conversion, and diagnostics:

```bash
conda activate /opt/miniforge/envs/casa6

dsa110 convert \
    --input-dir /data/incoming \
    --output-dir /stage/dsa110-contimg/ms \
    --start-time "2025-01-15T00:00:00" \
    --end-time "2025-01-15T12:00:00"
```

### CLI Options

| Option              | Description                          | Default          |
| ------------------- | ------------------------------------ | ---------------- |
| `--input-dir`       | Directory containing HDF5 files      | Required         |
| `--output-dir`      | Directory for output MS files        | Required         |
| `--start-time`      | Start of time range (ISO format)     | Required         |
| `--end-time`        | End of time range (ISO format)       | Required         |
| `--execution-mode`  | `auto`, `inprocess`, or `subprocess` | `auto`           |
| `--scratch-dir`     | Fast temp storage (NVMe/tmpfs)       | System default   |
| `--writer`          | MS writer type                       | `direct-subband` |
| `--dry-run`         | Preview without executing            | False            |
| `--diagnostics`     | Generate diagnostic HTML report      | True             |
| `--remap-input-dir` | Alternative HDF5 location            | None             |
| `--ms-suffix`       | Suffix for output MS name            | None             |
| `--max-workers`     | Parallel I/O workers                 | 8                |
| `--json`            | Output result as JSON                | False            |

### Example: Dry Run

```bash
dsa110 convert \
    --input-dir /data/incoming \
    --output-dir /stage/dsa110-contimg/ms \
    --start-time "2025-01-15T00:00:00" \
    --end-time "2025-01-15T06:00:00" \
    --dry-run
```

### Example: With Downsampled Data

```bash
dsa110 convert \
    --input-dir /data/incoming \
    --output-dir /stage/dsa110-contimg/ms \
    --start-time "2025-01-15T00:00:00" \
    --end-time "2025-01-15T12:00:00" \
    --remap-input-dir /data/golden-datasets/2025-01-15/raw \
    --ms-suffix "_12x"
```

---

## Method 2: Python Public API (Recommended for Scripts)

Use the public API for integration in Python scripts:

```python
from dsa110_contimg.interfaces.public_api import (
    ConversionRequest,
    ConversionSettings,
    convert_uvh5_to_ms,
)
from pathlib import Path

# Configure conversion settings
config = ConversionSettings(
    execution_mode="auto",        # auto, inprocess, or subprocess
    writer="direct-subband",      # Only supported writer
    max_workers=8,                # Parallel I/O workers
    scratch_dir=Path("/dev/shm/dsa110-contimg"), # Fast temp storage
)

# Build request
request = ConversionRequest(
    input_dir=Path("/data/incoming"),
    output_dir=Path("/stage/dsa110-contimg/ms"),
    start_time="2025-01-15T00:00:00",
    end_time="2025-01-15T12:00:00",
    config=config,
)

# Execute conversion
result = convert_uvh5_to_ms(request)

if result.success:
    print(f"✓ Created: {result.ms_path}")
else:
    print(f"✗ Failed: {result.error_message}")
```

### ConversionResult Fields

| Field           | Type   | Description                  |
| --------------- | ------ | ---------------------------- | ----------------------- |
| `success`       | `bool` | Whether conversion succeeded |
| `ms_path`       | `str   | None`                        | Path to output MS       |
| `group_id`      | `str`  | Identifier for the group     |
| `error_message` | `str   | None`                        | Error details if failed |
| `metrics`       | `dict` | Performance metrics          |

---

## Method 3: Dagster Orchestration (Recommended for Production)

Dagster provides scheduled, partitioned, and monitored execution with retry policies.

### Start the Dagster Webserver

```bash
conda activate /opt/miniforge/envs/casa6

# Development mode (auto-reloads on code changes)
dagster dev -m dsa110_contimg.workflow.dagster

# Production mode
dagster-webserver -m dsa110_contimg.workflow.dagster -h 0.0.0.0 -p 3050
```

Access the UI at: http://localhost:3050

### Available Jobs

| Job                | Description                                            |
| ------------------ | ------------------------------------------------------ |
| `conversion_only`  | Run UVH5 → MS conversion for a day                     |
| `full_pipeline`    | Complete pipeline (conversion → calibration → imaging) |
| `calibration_only` | Run calibration on existing MS files                   |
| `imaging_only`     | Create images from calibrated MS                       |

### Launch a Conversion Run via UI

1. Navigate to **Jobs** → `conversion_only`
2. Click **Launchpad**
3. Select partition (date) or configure time range
4. Adjust config parameters:
   ```yaml
   ops:
     measurement_sets:
       config:
         max_workers: 8
         skip_incomplete: true
         cluster_tolerance_s: 120.0
         overwrite_existing: false
   ```
5. Click **Launch Run**

### Launch via Python API

```python
from dagster import DagsterInstance, execute_job, reconstructable
from dsa110_contimg.workflow.dagster import defs

# Connect to Dagster instance
instance = DagsterInstance.get()

# Get the conversion job
conversion_job = defs.get_job_def("conversion_only")

# Execute for a specific partition (date)
result = conversion_job.execute_in_process(
    instance=instance,
    partition_key="2025-01-15",
    run_config={
        "ops": {
            "measurement_sets": {
                "config": {
                    "max_workers": 8,
                    "skip_incomplete": True,
                    "overwrite_existing": False,
                }
            }
        }
    },
)

if result.success:
    # Get output from asset
    ms_list = result.output_for_node("measurement_sets")
    print(f"Converted {len(ms_list)} measurement sets")
else:
    print("Conversion failed")
```

### Launch via CLI

```bash
# Run for a specific date partition
dagster job execute \
    -m dsa110_contimg.workflow.dagster \
    -j conversion_only \
    --partition "2025-01-15"

# Run with custom config
dagster job execute \
    -m dsa110_contimg.workflow.dagster \
    -j conversion_only \
    --partition "2025-01-15" \
    --config-json '{"ops": {"measurement_sets": {"config": {"max_workers": 12}}}}'
```

### ConversionRunConfig Parameters

| Parameter              | Type    | Default | Description                        |
| ---------------------- | ------- | ------- | ---------------------------------- | ------------------------------ |
| `start_time`           | `str    | None`   | None                               | Override start time (ISO 8601) |
| `end_time`             | `str    | None`   | None                               | Override end time (ISO 8601)   |
| `max_workers`          | `int`   | 8       | Parallel conversion workers (1-32) |
| `omp_threads`          | `int`   | 4       | OpenMP threads per worker (1-16)   |
| `skip_incomplete`      | `bool`  | True    | Skip groups with <16 subbands      |
| `cluster_tolerance_s`  | `float` | 120.0   | Time window for grouping (30-300s) |
| `overwrite_existing`   | `bool`  | False   | Overwrite existing MS files        |
| `cleanup_intermediate` | `bool`  | True    | Remove temp files after success    |

### Sensors for Automated Triggering

The pipeline includes sensors that automatically trigger conversion when new data arrives:

```python
# Sensor watches /data/incoming/ for new HDF5 files
# Triggers conversion_only job when complete subband groups are detected
```

Check sensor status:

```bash
dagster sensor list -m dsa110_contimg.workflow.dagster
dagster sensor start -m dsa110_contimg.workflow.dagster new_data_sensor
```

### Partition Backfill

To convert historical data for a date range:

1. Go to **Jobs** → `conversion_only` → **Partitions**
2. Select date range in the calendar view
3. Click **Launch Backfill**

Or via CLI:

```bash
dagster job backfill \
    -m dsa110_contimg.workflow.dagster \
    -j conversion_only \
    --from "2025-01-01" \
    --to "2025-01-31"
```

---

## Method 4: Low-Level API (Advanced)

For full control over the conversion process:

### Step 1: Query Subband Groups

```python
from dsa110_contimg.infrastructure.database.hdf5_index import query_subband_groups

db_path = "/data/dsa110-contimg/state/db/pipeline.sqlite3"

# Query groups with DP-based optimal segmentation
result = query_subband_groups(
    db_path=db_path,
    start_time="2025-01-15T00:00:00",
    end_time="2025-01-15T12:00:00",
    m_min=14,  # Minimum files per group
    m_max=18,  # Maximum files per group
)

# Filter for complete groups (16 subbands)
complete_groups = [g for g in result if g.is_complete]

for group in complete_groups:
    print(f"Group at {group.representative_time}: {len(group.files)} files")
    print(f"  Complete: {group.is_complete}")
    print(f"  Missing subbands: {group.missing_subbands}")
```

### Step 2: Convert with Auto-Discovery

```python
from dsa110_contimg.core.conversion import convert_subband_groups_to_ms

result = convert_subband_groups_to_ms(
    input_dir="/data/incoming",
    output_dir="/stage/dsa110-contimg/ms",
    start_time="2025-01-15T00:00:00",
    end_time="2025-01-15T12:00:00",
    skip_incomplete=True,   # Skip groups with <16 subbands
    skip_existing=True,     # Skip already-converted
    stage_to_tmpfs=True,    # Use /dev/shm for fast I/O
)

print(f"Converted: {result['converted_count']} groups")
print(f"Skipped: {result['skipped_count']} groups")
```

### Step 3: Convert Explicit File List

For converting a known set of files (bypasses database discovery):

```python
from dsa110_contimg.core.conversion.writers import get_writer
from pathlib import Path
import pyuvdata

# Explicit list of 16 subband files
file_list = [
    "/data/incoming/2025-01-15T12:00:00_sb00.hdf5",
    "/data/incoming/2025-01-15T12:00:00_sb01.hdf5",
    # ... all 16 subbands
    "/data/incoming/2025-01-15T12:00:00_sb15.hdf5",
]

output_path = Path("/stage/dsa110-contimg/ms/2025-01-15T12:00:00.ms")

# Use DirectSubbandWriter via the writer registry
writer_cls = get_writer("direct-subband")
uvdata = pyuvdata.UVData()  # Empty - DirectSubbandWriter reads files directly
writer = writer_cls(uvdata, str(output_path), file_list=file_list, max_workers=4)
writer.write()
```

### Step 4: Using the DirectSubbandWriter Directly

```python
from pyuvdata import UVData
from dsa110_contimg.core.conversion.writers import DirectSubbandWriter

# Load first file to get UVData template
uv = UVData()
uv.read(file_list[0], file_type="uvh5", read_data=False)

# Create writer
writer = DirectSubbandWriter(
    uv=uv,
    ms_path=str(output_path),
    file_list=file_list,
    scratch_dir="/dev/shm/dsa110-contimg",
    max_workers=8,
    stage_to_tmpfs=True,
    merge_spws=False,  # Keep multi-SPW for calibration compatibility
)

# Write the MS
writer_type = writer.write()
print(f"Written with: {writer_type}")
```

---

## Method 5: Bandpass Calibrator Transit Selection (Science Calibration)

Generate MS files specifically targeting the transit of a bandpass calibrator for science-quality calibration. This method:

1. Determines telescope pointing from the latest observation
2. Selects the best calibrator from the VLA catalog for that pointing
3. Calculates when the calibrator transits through the primary beam
4. Selects HDF5 groups centered on the transit time
5. Converts the selected groups to produce a calibrator-centered MS

### When to Use This Method

- **Science-grade calibration**: Bandpass calibration requires high SNR observations
- **Automated test runs**: Daily pipeline health checks use this approach
- **Mosaic photometry**: Creating well-calibrated mosaics around calibrator transits

### Quick Start: Using CalibratorMSGenerator (Recommended)

The `CalibratorMSGenerator` class provides a unified high-level interface:

```python
from dsa110_contimg.interfaces.public_api import CalibratorMSGenerator
from pathlib import Path
from astropy.time import Time

generator = CalibratorMSGenerator(
    input_dir=Path("/data/incoming"),
    output_dir=Path("/stage/dsa110-contimg/ms"),
)

# Option 1: Generate from a known calibrator transit
result = generator.generate_from_transit(
    calibrator_name="0834+555",
    transit_time=Time("2025-01-15T14:30:00"),
    window_minutes=30,
)

print(f"MS: {result.ms_path}")
print(f"Calibrator in MS: {result.calibrator_in_ms}")

# Option 2: Auto-detect best calibrator for current pointing
result = generator.generate_for_pointing(
    dec_deg=43.5,           # Telescope declination
    lookback_days=7,        # Search last 7 days for transits
    min_flux_jy=1.0,        # Minimum calibrator flux
)

print(f"Selected: {result.calibrator.name} ({result.calibrator.flux_jy:.1f} Jy)")
print(f"Transit: {result.transit.transit_time_iso}")
print(f"MS: {result.ms_path}")

# Option 3: Generate multiple MS for mosaic creation
result = generator.generate_multiple(
    calibrator_name="3C286",
    transit_time=Time("2025-01-15T12:00:00"),
    n_groups=12,  # 12 × 5 min = 1 hour of data
)

print(f"Created {len(result.ms_paths)} MS files")
```

### Key Functions

| Function                              | Module                               | Purpose                                    |
| ------------------------------------- | ------------------------------------ | ------------------------------------------ |
| `CalibratorMSGenerator`               | `interfaces.public_api`              | **Unified high-level interface**           |
| `get_best_vla_calibrator()`           | `interfaces.public_api`              | Select optimal calibrator from VLA catalog |
| `transit_times()`                     | `core.calibration.transit`           | Calculate meridian transit times           |
| `select_hdf5_groups_around_transit()` | `infrastructure.database.hdf5_index` | Query groups before/after transit          |
| `select_bandpass_from_catalog()`      | `core.calibration.selection`         | Find calibrator within MS fields           |

### Step-by-Step: Manual Workflow

For more control, you can use the individual functions:

#### Step 1: Find the Best Calibrator for Current Pointing

```python
from dsa110_contimg.interfaces.public_api import get_best_vla_calibrator

# Get calibrator matching current telescope declination
# Uses the authoritative VLA calibrator catalog at:
# /data/dsa110-contimg/state/catalogs/vla_calibrators.sqlite3

calibrator = get_best_vla_calibrator(
    dec_deg=43.5,           # Telescope pointing declination
    dec_tolerance=5.0,      # Search ±5° in declination
    min_flux_jy=1.0,        # Minimum flux threshold at L-band
)

if calibrator:
    print(f"Selected: {calibrator['name']}")
    print(f"  RA:   {calibrator['ra_deg']:.4f}°")
    print(f"  Dec:  {calibrator['dec_deg']:.4f}°")
    print(f"  Flux: {calibrator['flux_jy']:.2f} Jy")
else:
    raise RuntimeError(f"No calibrator found for Dec {dec_deg}°")
```

### Calibrator Selection Criteria

Calibrators are ranked by:

- **Flux density** (higher is better for SNR)
- **Position code** (A = best astrometry, B/C = acceptable)
- **Proximity** to target declination

#### Step 2: Calculate Transit Times

```python
from dsa110_contimg.core.calibration.transit import transit_times
from astropy.time import Time
import astropy.units as u

# Define time window (e.g., last 7 days)
end_time = Time.now()
start_time = end_time - 7 * u.day

# Get all transits in the window
transits = transit_times(
    ra_deg=calibrator['ra_deg'],
    start_time=start_time,
    end_time=end_time,
)

print(f"Found {len(transits)} transits in the last 7 days:")
for t in transits:
    print(f"  {t.iso}")

# Use the most recent transit
last_transit = transits[-1]
print(f"\nMost recent transit: {last_transit.iso}")
```

### Understanding Transit Times

The DSA-110 is a drift-scan telescope. Calibrators transit the meridian once per sidereal day (~23h 56m). The `transit_times()` function:

- Computes when RA = Local Sidereal Time (LST)
- Uses DSA-110 location (OVRO site)
- Returns astropy `Time` objects in UTC

#### Step 3: Select HDF5 Groups Around Transit

```python
from dsa110_contimg.infrastructure.database.hdf5_index import select_hdf5_groups_around_transit

db_path = "/data/dsa110-contimg/state/db/pipeline.sqlite3"

# Select groups centered on calibrator transit
# Each group is ~5 minutes, so 6 before + 6 after = ~1 hour window
selected_groups = select_hdf5_groups_around_transit(
    db_path=db_path,
    transit_time=last_transit,
    n_groups_before=6,    # 6 × 5 min = 30 min before transit
    n_groups_after=6,     # 6 × 5 min = 30 min after transit
)

print(f"Selected {len(selected_groups)} groups for conversion")

# Each group is a list of 16 file paths (one per subband)
for i, group in enumerate(selected_groups):
    print(f"  Group {i}: {len(group)} files")
```

### Step 4: Convert Selected Groups

```python
from dsa110_contimg.core.conversion import convert_subband_groups_to_ms
from pathlib import Path

output_dir = Path("/stage/dsa110-contimg/ms/calibrator_transit")
output_dir.mkdir(parents=True, exist_ok=True)

# Convert each group
ms_paths = []
for group in selected_groups:
    # Extract representative timestamp from first file
    first_file = Path(group[0]).name
    # Expected format: 2025-01-15T12:00:00_sb00.hdf5
    timestamp = first_file.rsplit("_sb", 1)[0]

    ms_path = output_dir / f"{timestamp}.ms"

    writer_cls = get_writer("direct-subband")
    uvdata = pyuvdata.UVData()
    writer = writer_cls(uvdata, str(ms_path), file_list=group, max_workers=4)
    writer.write()
    ms_paths.append(ms_path)

print(f"Created {len(ms_paths)} Measurement Sets around {calibrator['name']} transit")
```

### Complete Workflow Example

```python
"""Generate MS files around the last transit of the best calibrator."""
from astropy.time import Time
import astropy.units as u
from pathlib import Path

from dsa110_contimg.core.calibration.transit import transit_times
from dsa110_contimg.core.catalog.calibrator_registry import get_best_calibrator
from dsa110_contimg.infrastructure.database.hdf5_index import select_hdf5_groups_around_transit
from dsa110_contimg.core.conversion.writers import get_writer
import pyuvdata

# Configuration
TELESCOPE_DEC_DEG = 43.5  # Current telescope pointing
LOOKBACK_DAYS = 7
N_GROUPS_BEFORE = 6
N_GROUPS_AFTER = 6
DB_PATH = "/data/dsa110-contimg/state/db/pipeline.sqlite3"
OUTPUT_DIR = Path("/stage/dsa110-contimg/ms/calibrator_transit")

# Step 1: Get best calibrator
calibrator = get_best_calibrator(dec_deg=TELESCOPE_DEC_DEG)
if not calibrator:
    raise RuntimeError(f"No calibrator found for Dec {TELESCOPE_DEC_DEG}°")

print(f"Selected calibrator: {calibrator['name']} ({calibrator['flux_1400mhz_jy']:.1f} Jy)")

# Step 2: Find recent transits
end_time = Time.now()
start_time = end_time - LOOKBACK_DAYS * u.day

transits = transit_times(
    ra_deg=calibrator['ra_deg'],
    start_time=start_time,
    end_time=end_time,
)

if not transits:
    raise RuntimeError(f"No transits found in the last {LOOKBACK_DAYS} days")

last_transit = transits[-1]
print(f"Last transit: {last_transit.iso}")

# Step 3: Select HDF5 groups around transit
groups = select_hdf5_groups_around_transit(
    db_path=DB_PATH,
    transit_time=last_transit,
    n_groups_before=N_GROUPS_BEFORE,
    n_groups_after=N_GROUPS_AFTER,
)

if len(groups) < (N_GROUPS_BEFORE + N_GROUPS_AFTER):
    raise RuntimeError(
        f"Insufficient data: found {len(groups)} groups, "
        f"need {N_GROUPS_BEFORE + N_GROUPS_AFTER}"
    )

print(f"Selected {len(groups)} groups")

# Step 4: Convert to MS
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ms_paths = []

for group in groups:
    timestamp = Path(group[0]).stem.rsplit("_sb", 1)[0]
    ms_path = OUTPUT_DIR / f"{timestamp}.ms"

    writer_cls = get_writer("direct-subband")
    uvdata = pyuvdata.UVData()
    writer = writer_cls(uvdata, str(ms_path), file_list=group, max_workers=4)
    writer.write()
    ms_paths.append(ms_path)

print(f"✓ Created {len(ms_paths)} MS files centered on {calibrator['name']} transit")
```

### Using the Test Run Framework

The pipeline includes a complete test run mechanism that automates this workflow:

```python
from dsa110_contimg.workflow.pipeline.test_run import run_mosaic_photometry_test
from pathlib import Path

# Run complete test pipeline (conversion → calibration → imaging → photometry)
result = run_mosaic_photometry_test(
    db_path="/data/dsa110-contimg/state/db/pipeline.sqlite3",
    output_dir=Path("/stage/dsa110-contimg/test_runs/2025-01-15"),
    lookback_days=7,
    n_groups_before=6,
    n_groups_after=6,
    dry_run=False,  # Set True to validate without processing
)

if result["success"]:
    print(f"✓ Test run complete")
    print(f"  Calibrator: {result['calibrator_info']['calibrator_name']}")
    print(f"  Transit: {result['calibrator_info']['transit_time_iso']}")
    print(f"  Groups processed: {len(result['selected_groups'])}")
    print(f"  Mosaic: {result['mosaic_path']}")
else:
    print(f"✗ Test run failed: {result['error_message']}")
```

### Post-Conversion: Verify Calibrator in MS

After conversion, verify the calibrator is present in the MS fields:

```python
from dsa110_contimg.core.calibration.selection import select_bandpass_from_catalog

ms_path = "/stage/dsa110-contimg/ms/2025-01-15T12:00:00.ms"

# Find calibrator within MS fields (uses primary beam weighting)
field_sel, field_indices, weighted_flux, cal_info, peak_field = select_bandpass_from_catalog(
    ms_path=ms_path,
    calibrator_name=calibrator['name'],  # Optional: restrict to specific calibrator
    search_radius_deg=1.0,
    freq_GHz=1.4,
)

name, ra_deg, dec_deg, flux_jy = cal_info

print(f"Calibrator {name} found in MS:")
print(f"  Peak field: {peak_field} (field selection: {field_sel})")
print(f"  Catalog position: ({ra_deg:.4f}°, {dec_deg:.4f}°)")
print(f"  Catalog flux: {flux_jy:.2f} Jy")
print(f"  Peak weighted flux: {weighted_flux[field_indices.index(peak_field)]:.2f} Jy")
```

### Primary Beam Considerations

The DSA-110 primary beam FWHM is ~3° at 1.4 GHz. When selecting calibrators:

- **Peak response**: Calibrator transits at the beam center (field closest to meridian)
- **±1.5° window**: Usable data within ~±6 fields around transit
- **Beam-weighted flux**: `flux × beam_response(θ)` determines effective SNR

The `select_bandpass_from_catalog()` function handles beam weighting when `use_beam_weighting=True`:

```python
field_sel, indices, weighted_flux, cal_info, peak = select_bandpass_from_catalog(
    ms_path=ms_path,
    use_beam_weighting=True,  # Weight fields by primary beam response
    window=3,                 # Use ±3 fields around peak
    min_pb=0.1,               # Minimum 10% beam response
)
```

### VLA Calibrator Catalog

The catalog at `/data/dsa110-contimg/state/catalogs/vla_calibrators.sqlite3` contains:

| Column            | Description                          |
| ----------------- | ------------------------------------ |
| `source_name`     | VLA Standard name (e.g., `0834+555`) |
| `ra_deg`          | Right Ascension (J2000)              |
| `dec_deg`         | Declination (J2000)                  |
| `flux_1400mhz_jy` | Flux at 1.4 GHz                      |
| `spectral_index`  | Spectral index (S ∝ ν^α)             |
| `quality_score`   | Combined quality metric              |

To query directly:

```python
import sqlite3

db_path = "/data/dsa110-contimg/state/catalogs/vla_calibrators.sqlite3"

with sqlite3.connect(db_path) as conn:
    cur = conn.cursor()
    cur.execute("""
        SELECT source_name, ra_deg, dec_deg, flux_1400mhz_jy
        FROM calibrator_sources
        WHERE dec_deg BETWEEN 38 AND 48
          AND flux_1400mhz_jy > 2.0
        ORDER BY flux_1400mhz_jy DESC
        LIMIT 5
    """)
    for row in cur.fetchall():
        print(f"{row[0]}: RA={row[1]:.2f}°, Dec={row[2]:.2f}°, Flux={row[3]:.1f} Jy")
```

---

## Post-Conversion: Configure MS for Imaging

After writing, apply DSA-110-specific configuration:

```python
from dsa110_contimg.core.conversion.ms_utils import configure_ms_for_imaging

# Apply DSA-110 configuration
configure_ms_for_imaging(
    ms_path="/stage/dsa110-contimg/ms/2025-01-15T12:00:00.ms",
    rename_calibrator_fields=True,  # Auto-detect and rename calibrator field
)
```

This function:

1. Updates antenna positions from DSA-110 station coordinates
2. Auto-detects calibrator in field list and renames the field
3. Sets proper telescope identity and metadata

---

## Storage Architecture

| Location                         | Type     | Size  | Use For                           |
| -------------------------------- | -------- | ----- | --------------------------------- |
| `/data/incoming/`                | HDD      | 13 TB | Raw HDF5 files (read-only)        |
| `/stage/dsa110-contimg/ms/`      | NVMe SSD | 1 TB  | Processing outputs (fast I/O)     |
| `/data/stage/dsa110-contimg/ms/` | HDD      | 13 TB | MS archive storage (long-term)    |
| `/dev/shm/`                      | tmpfs    | RAM   | Temporary files during conversion |

**Processing Workflow**:

1. Convert to `/stage/dsa110-contimg/ms/` (SSD) for fast processing
2. Calibration + imaging runs on SSD for fast I/O
3. After successful validation, auto-archive to `/data/stage/dsa110-contimg/ms/` (HDD)

**Auto-Archive**: Enabled by default (`validation.archive_to_hdd=True`). Disable with:

```python
config.validation.archive_to_hdd = False
```

**CRITICAL**: Avoid I/O-intensive operations directly on `/data/` (HDD). Write intermediates to `/stage/` or `/dev/shm/`.

---

## Troubleshooting

### No Files Found

```python
# Check if files are indexed
from dsa110_contimg.infrastructure.database.hdf5_index import count_files_in_range

count = count_files_in_range(
    db_path="/data/dsa110-contimg/state/db/pipeline.sqlite3",
    start_time="2025-01-15T00:00:00",
    end_time="2025-01-15T12:00:00",
)
print(f"Files in range: {count}")
```

### Re-index HDF5 Files

```bash
dsa110 index add --start "2025-01-15" --end "2025-01-16"
```

### Incomplete Groups

Groups with fewer than 16 subbands are incomplete. Common causes:

- Correlator glitches
- Network transfer failures
- Clock synchronization issues

To process anyway (with missing subbands):

```python
result = convert_subband_groups_to_ms(
    ...,
    skip_incomplete=False,  # Process incomplete groups
)
```

### Memory Issues

For memory-constrained systems:

```bash
dsa110 convert \
    --memory-mb 8000 \
    --max-workers 4 \
    --omp-threads 4 \
    ...
```

### CASA Log Conflicts

Enable process isolation for concurrent CASA operations:

```bash
export DSA110_CASA_PROCESS_ISOLATION=true
```

---

## Output Structure

Successful conversion produces:

```
/stage/dsa110-contimg/ms/
├── 2025-01-15T12:00:00.staged.ms/    # Multi-SPW Measurement Set
│   ├── ANTENNA/
│   ├── DATA_DESCRIPTION/
│   ├── FEED/
│   ├── FIELD/
│   ├── OBSERVATION/
│   ├── POLARIZATION/
│   ├── SOURCE/
│   ├── SPECTRAL_WINDOW/
│   ├── STATE/
│   └── table.dat
└── diagnostics/                       # If --diagnostics enabled
    └── 2025-01-15T12:00:00/
        ├── amp_vs_time.png
        ├── phase_vs_time.png
        └── summary.html
```

### MS Structure Details

- **24 fields**: `meridian_icrs_t0` through `meridian_icrs_t23` (one per 12.88s timestamp)
- **16 spectral windows**: 48 channels each = 768 total channels
- **~5 minutes**: Each MS covers approximately 309 seconds of observation

---

## Validation: Verifying MS Was Correctly Generated

The pipeline automatically validates MS files after conversion. The agent can also perform manual validation to confirm correctness.

### Quick Validation Check (Required Columns)

```python
from dsa110_contimg.core.conversion.helpers import table

ms_path = "/stage/dsa110-contimg/ms/2025-01-15T12:00:00.staged.ms"

with table(ms_path, readonly=True) as tb:
    # Check required columns exist
    required_cols = ["DATA", "ANTENNA1", "ANTENNA2", "TIME", "UVW"]
    missing = [c for c in required_cols if c not in tb.colnames()]
    if missing:
        raise RuntimeError(f"MS missing required columns: {missing}")

    # Check data rows exist
    nrows = tb.nrows()
    if nrows == 0:
        raise RuntimeError("MS has no data rows")

    print(f"✓ MS has {nrows:,} rows with all required columns")
```

### Comprehensive Validation Suite

```python
from dsa110_contimg.core.conversion.helpers_validation import (
    validate_ms_frequency_order,
    validate_phase_center_coherence,
    validate_uvw_precision,
    validate_antenna_positions,
)

ms_path = "/stage/dsa110-contimg/ms/2025-01-15T12:00:00.staged.ms"

# 1. Frequency order (CRITICAL - imaging will fail if wrong)
validate_ms_frequency_order(ms_path)
print("✓ Frequencies in ascending order (required for MFS imaging)")

# 2. Phase center coherence
validate_phase_center_coherence(ms_path, tolerance_arcsec=1.0)
print("✓ Phase centers coherent across subbands")

# 3. UVW coordinate precision
validate_uvw_precision(ms_path, tolerance_lambda=0.1)
print("✓ UVW coordinates within tolerance")

# 4. Antenna positions match DSA-110
validate_antenna_positions(ms_path, position_tolerance_m=0.05)
print("✓ Antenna positions accurate (within 5cm)")
```

### Validation Criteria Summary

| Check                 | Expected Value                      | Failure Impact                           |
| --------------------- | ----------------------------------- | ---------------------------------------- |
| **Row count**         | >0                                  | No data to process                       |
| **Required columns**  | DATA, ANTENNA1, ANTENNA2, TIME, UVW | Cannot calibrate/image                   |
| **Frequency order**   | Ascending (sb15 → sb00)             | MFS imaging artifacts, bandpass failures |
| **Antenna count**     | 110 antennas                        | Missing baselines                        |
| **SPW count**         | 16 spectral windows                 | Missing frequency coverage               |
| **Total channels**    | 768 (16 × 48)                       | Reduced bandwidth                        |
| **Field count**       | 24 fields                           | Incomplete time coverage                 |
| **Phase coherence**   | <1 arcsec separation per field      | Imaging artifacts                        |
| **UVW precision**     | <0.1λ error                         | Decorrelation, flagged solutions         |
| **Antenna positions** | <5cm error vs reference             | Poor calibration                         |

### Automated Validation in Pipeline

The `ConversionStage.validate_outputs()` method runs automatically after conversion:

```python
# This is called automatically by the pipeline after conversion
from dsa110_contimg.workflow.pipeline.stages.conversion import ConversionStage

stage = ConversionStage()
is_valid, error_msg = stage.validate_outputs(context)

if not is_valid:
    print(f"✗ Validation failed: {error_msg}")
else:
    print("✓ MS passed automatic validation")
```

### CLI Validation

```bash
conda activate /opt/miniforge/envs/casa6

# Validate MS structure
python -m dsa110_contimg.core.conversion.validate_ms /stage/dsa110-contimg/ms/2025-01-15T12:00:00.staged.ms
```

### Expected Validation Output

For a correctly generated MS:

```
✓ Frequency order validation passed: 16 SPW(s), range 1280.5-1530.0 MHz
✓ Phase center coherence validated: 24 field(s), max separation 0.01 arcsec
✓ UVW coordinate validation passed: median baseline 1245.3m, max 3100.8m
✓ Antenna position validation passed: 110 antennas, max error 2.1cm, RMS 0.8cm
```

### Validation Failure Recovery

| Failure                 | Cause                      | Recovery                       |
| ----------------------- | -------------------------- | ------------------------------ |
| Missing columns         | Incomplete conversion      | Re-run conversion              |
| Zero rows               | Empty input or failed read | Check input HDF5 files         |
| Wrong frequency order   | Subband ordering bug       | Report to maintainer           |
| Phase incoherence >1"   | Subbands misaligned        | Re-convert with fresh grouping |
| Large UVW errors        | Coordinate system issue    | Check antenna position file    |
| Antenna position errors | Outdated antpos file       | Update `antpos_local.py`       |

### DSA-110 Specific Expectations

```python
# Constants for DSA-110 MS validation
NUM_ANTENNAS = 110
NUM_SUBBANDS = 16
CHANNELS_PER_SUBBAND = 48
TOTAL_CHANNELS = NUM_SUBBANDS * CHANNELS_PER_SUBBAND  # 768
NUM_FIELDS = 24  # One per 12.88s timestamp
OBS_DURATION_S = 309  # ~5 minutes per MS
FREQ_RANGE_MHZ = (1280, 1530)  # L-band
MAX_BASELINE_M = 3210  # Longest DSA-110 baseline
```

### Programmatic Validation Example

```python
from dsa110_contimg.core.conversion.helpers import table
import numpy as np

def validate_ms_complete(ms_path: str) -> dict:
    """Comprehensive MS validation returning a result dict."""
    results = {"valid": True, "checks": {}, "errors": []}

    # 1. Check structure
    with table(ms_path, readonly=True) as tb:
        results["checks"]["nrows"] = tb.nrows()
        results["checks"]["columns"] = tb.colnames()
        if tb.nrows() == 0:
            results["valid"] = False
            results["errors"].append("No data rows")

    # 2. Antenna count
    with table(f"{ms_path}::ANTENNA", readonly=True) as ant:
        nant = ant.nrows()
        results["checks"]["antennas"] = nant
        if nant < 100:
            results["errors"].append(f"Only {nant} antennas (expected ~110)")

    # 3. SPW count
    with table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True) as spw:
        nspw = spw.nrows()
        results["checks"]["spectral_windows"] = nspw
        if nspw != 16:
            results["errors"].append(f"{nspw} SPWs (expected 16)")

        # Total channels
        chan_freq = spw.getcol("CHAN_FREQ")
        total_chan = chan_freq.size
        results["checks"]["total_channels"] = total_chan
        if total_chan != 768:
            results["errors"].append(f"{total_chan} channels (expected 768)")

    # 4. Field count
    with table(f"{ms_path}::FIELD", readonly=True) as field:
        nfield = field.nrows()
        results["checks"]["fields"] = nfield
        if nfield != 24:
            results["errors"].append(f"{nfield} fields (expected 24)")

    if results["errors"]:
        results["valid"] = False

    return results


# Usage
result = validate_ms_complete("/stage/dsa110-contimg/ms/2025-01-15T12:00:00.staged.ms")
if result["valid"]:
    print("✓ MS is correctly generated")
else:
    print("✗ MS has issues:")
    for err in result["errors"]:
        print(f"  - {err}")
```

---

## Related Skills

- `calibration/SKILL.md` - Calibrate the generated MS
- `imaging/SKILL.md` - Create images from calibrated MS
- `dagster-workflows/SKILL.md` - Automated pipeline execution

---

## Key Code References

| File                                         | Purpose                    |
| -------------------------------------------- | -------------------------- |
| `interfaces/cli/commands/convert.py`         | CLI implementation         |
| `interfaces/public_api.py`                   | Public API functions       |
| `core/conversion/conversion_orchestrator.py` | Batch conversion           |
| `core/conversion/direct_subband.py`          | DirectSubbandWriter class  |
| `core/conversion/writers.py`                 | Writer factory             |
| `core/conversion/ms_utils.py`                | MS configuration utilities |
| `infrastructure/database/hdf5_index.py`      | Subband grouping queries   |
