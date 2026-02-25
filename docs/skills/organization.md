# Organization Skill

Expert guidance for MS file organization in the DSA-110 pipeline.

## Overview

The organization stage moves MS files into a structured date-based directory hierarchy, separating calibrator observations from science observations and tracking failed processing attempts.

## Key Modules

| Module | Purpose |
|--------|---------|
| `workflow/pipeline/stages/organization.py` | `OrganizationStage` pipeline stage |
| `common/utils/ms_organization.py` | MS file organization utilities |
| `common/utils/paths.py` | Path determination and MS type detection |

## Directory Structure

Organized directory layout:

```
output_dir/
├── ms/
│   ├── cal/                    # Calibrator observations
│   │   ├── 2025-01-15/
│   │   │   ├── 3C286_2025-01-15T10:30:00.ms
│   │   │   └── 0834+555_2025-01-15T14:45:00.ms
│   │   └── 2025-01-16/
│   │       └── ...
│   ├── sci/                    # Science observations
│   │   ├── 2025-01-15/
│   │   │   ├── survey_2025-01-15T00:00:00.ms
│   │   │   └── survey_2025-01-15T01:00:00.ms
│   │   └── 2025-01-16/
│   │       └── ...
│   └── failed/                 # Failed processing
│       └── 2025-01-15/
│           └── obs_2025-01-15T12:00:00.ms
```

## MS Type Detection

```python
from dsa110_contimg.common.utils.paths import determine_ms_type

is_calibrator, is_failed = determine_ms_type(ms_path)
# Returns: (bool, bool)

# Classification logic:
# is_calibrator=True: Name matches known calibrator patterns
# is_failed=True: Processing failed (flagged in database or by filename)
```

### Calibrator Detection

Known calibrator patterns:

```python
CALIBRATOR_PATTERNS = [
    r"^3C\d+",           # 3C286, 3C48, etc.
    r"^\d{4}[+-]\d{3}",  # J2000 format: 0834+555
    r"^J\d{4}[+-]\d{4}", # Full J2000: J1331+3030
    r"^CYG[_ ]?A",       # Cygnus A
    r"^CAS[_ ]?A",       # Cassiopeia A
    r"^VIR[_ ]?A",       # Virgo A
]
```

## Pipeline Stage Usage

```python
from dsa110_contimg.workflow.pipeline.stages import OrganizationStage

stage = OrganizationStage(config)
context = PipelineContext(
    config=config,
    outputs={
        "ms_path": "/data/raw/observation.ms",
        # Or multiple:
        "ms_paths": ["/data/raw/obs1.ms", "/data/raw/obs2.ms"],
    }
)

result = stage.execute(context)
organized_path = result.outputs["ms_path"]  # Now in organized location
```

## Organization Function

```python
from dsa110_contimg.common.utils.ms_organization import organize_ms_file

# Organize MS during processing (SSD - fast I/O)
organized_path = organize_ms_file(
    ms_source=Path("/data/raw/observation.ms"),
    ms_base_dir=Path("/stage/dsa110-contimg"),  # SSD for processing
    products_db_path=Path("/data/dsa110-contimg/state/db/pipeline.sqlite3"),
    is_calibrator=True,
    is_failed=False,
    update_database=True,  # Update path in database
)
# Returns: Path("/stage/dsa110-contimg/ms/cal/2025-01-15/3C286_2025-01-15T10:30:00.ms")

# For archive storage after processing, move to HDD:
# Archive location: /data/stage/dsa110-contimg/ms/
```

## Organization Logic

```python
def get_organized_path(
    ms_path: Path,
    base_dir: Path,
    is_calibrator: bool,
    is_failed: bool,
) -> Path:
    """Determine organized path for MS file."""
    
    # Extract date from MS name or metadata
    obs_date = extract_date_from_ms(ms_path)
    date_str = obs_date.strftime("%Y-%m-%d")
    
    # Determine subdirectory
    if is_failed:
        subdir = "failed"
    elif is_calibrator:
        subdir = "cal"
    else:
        subdir = "sci"
    
    return base_dir / "ms" / subdir / date_str / ms_path.name
```

## Database Path Updates

When `update_database=True`, the stage updates all database references:

```python
# Tables with MS paths that get updated:
# - ms_index
# - caltables (MS reference)
# - images (MS reference)
# - photometry (MS reference)

from dsa110_contimg.common.utils.ms_organization import update_database_paths

update_database_paths(
    db_path=products_db_path,
    old_path=old_ms_path,
    new_path=organized_path,
)
```

## Symbolic Links (Optional)

For backward compatibility, create symlinks at old locations:

```python
from dsa110_contimg.common.utils.ms_organization import create_backward_link

create_backward_link(
    original_path=old_path,
    organized_path=new_path,
)
# Creates: /data/raw/observation.ms -> /stage/ms/sci/2025-01-15/observation.ms
```

## Configuration

```python
from dsa110_contimg.common.unified_config import settings

# Organization settings
settings.organization.enabled = True
settings.organization.create_symlinks = False
settings.organization.update_database = True
settings.organization.date_format = "%Y-%m-%d"
```

## CLI Commands

```bash
# Organize single MS
dsa110 organize /data/raw/observation.ms --output-dir /stage/dsa110-contimg

# Batch organize
dsa110 organize /data/raw/*.ms --output-dir /stage/dsa110-contimg

# Dry run (show what would happen)
dsa110 organize /data/raw/*.ms --dry-run

# Force re-organization
dsa110 organize /data/raw/observation.ms --force
```

## Context Flow

```python
# Input context
context.outputs = {
    "ms_path": "/data/raw/observation.ms"
}

# After organization
context.outputs = {
    "ms_path": "/stage/ms/sci/2025-01-15/observation.ms",
    "ms_original_path": "/data/raw/observation.ms",
    "ms_organized": True,
}
```

## Failed MS Handling

MS files marked as failed are moved to `failed/` subdirectory:

```python
# Reasons for marking as failed:
# - Calibration failed
# - Validation failed with "failed" status
# - Imaging produced empty output
# - Manual flag in database

# Check if MS is failed
from dsa110_contimg.common.utils.paths import is_ms_failed

if is_ms_failed(ms_path, db_path):
    organize_ms_file(..., is_failed=True)
```

## Space Management

Organization helps with disk space management:

```python
# Query organized MS by category
dsa110 organize list --type cal --date 2025-01-15
dsa110 organize list --type failed --older-than 30d

# Clean old failed MS
dsa110 organize clean --type failed --older-than 90d --dry-run
```

## Database Schema

```sql
-- MS tracking in pipeline.sqlite3
CREATE TABLE ms_index (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    original_path TEXT,
    ms_type TEXT,  -- 'calibrator', 'science', 'failed'
    observation_date DATE,
    organized_at TIMESTAMP,
    file_size_bytes INTEGER,
    -- ... other metadata
);
```

## Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Path not updated | Missing update_database | Set `update_database=True` |
| Duplicate names | Same timestamp | Add unique suffix |
| Permission denied | Wrong ownership | Check file permissions |
| Broken symlinks | Original moved | Re-run organization |

## Integration Notes

Organization stage typically runs:
1. **After conversion**: Organize newly created MS
2. **After calibration**: Re-classify if calibrator identified
3. **After validation**: Move to failed if validation fails

```python
# Example pipeline sequence
stages = [
    ConversionStage(config),
    CalibrationStage(config),
    ValidationStage(config),
    OrganizationStage(config),  # Runs last to classify correctly
]
```

## Related Resources

- MS generation skill: `.agent/skills/ms-generation/SKILL.md`
- Calibration skill: `.agent/skills/calibration/SKILL.md`
- Validation skill: `.agent/skills/validation/SKILL.md`
