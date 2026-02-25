# External Package Documentation Skill

This skill provides access to documentation for key radio astronomy and data processing packages used in the DSA-110 pipeline.

## Available Packages

### Python Libraries
| Package | Version | Entries | Description |
|---------|---------|---------|-------------|
| **astropy** | 7.1.0 | 5,495 | Core astronomy utilities (coordinates, time, units, FITS, tables) |
| **pyuvdata** | 3.2.5 | 953 | UV visibility data handling for radio interferometry |
| **pyuvsim** | 1.4.2 | 707 | UV simulation for radio interferometry |
| **numpy** | 2.3.5 | 509 | Numerical computing with arrays |
| **casacore** | 3.7.1 | 162 | CASA table interface for Measurement Sets |
| **h5py** | 3.14.0 | 106 | HDF5 file handling |

### Command-Line Tools
| Package | Entries | Description |
|---------|---------|-------------|
| **casa** | 20 | CASA tasks (tclean, gaincal, bandpass, applycal, flagdata, etc.) |
| **wsclean** | 9 | WSClean imaging (mgain, multiscale, auto-mask, weighting, predict) |
| **aoflagger** | 7 | AOFlagger RFI detection (Lua strategies, Python interface, sumthreshold) |

**Total: 7,968 indexed documentation entries**

## How to Search Documentation

### Command Line

```bash
# Search all packages
python scripts/agents/index_external_docs.py --search "UVData read"

# Search for CLI tool documentation
python scripts/agents/index_external_docs.py --search "wsclean multiscale"

# Search for CASA tasks
python scripts/agents/index_external_docs.py --search "tclean gridder"
```

### Python API

```python
from pathlib import Path
import sqlite3

DB_PATH = Path("/data/dsa110-contimg/state/db/external_docs.sqlite3")

def search_package_docs(query: str, packages: list[str] = None, limit: int = 5) -> list[dict]:
    """Search indexed package documentation.
    
    Parameters
    ----------
    query : str
        Full-text search query (FTS5 syntax supported)
    packages : list of str, optional
        Filter to specific packages (e.g., ['casa', 'wsclean', 'pyuvdata'])
    limit : int
        Maximum results to return
        
    Returns
    -------
    list of dict
        Search results with 'name', 'content', 'package', 'type'
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        
        if packages:
            placeholders = ",".join("?" * len(packages))
            results = conn.execute(f'''
                SELECT d.name, d.content, d.package, d.type
                FROM doc_fts f
                JOIN documentation d ON f.rowid = d.id
                WHERE doc_fts MATCH ?
                  AND d.package IN ({placeholders})
                ORDER BY bm25(doc_fts)
                LIMIT ?
            ''', (query, *packages, limit)).fetchall()
        else:
            results = conn.execute('''
                SELECT d.name, d.content, d.package, d.type
                FROM doc_fts f
                JOIN documentation d ON f.rowid = d.id
                WHERE doc_fts MATCH ?
                ORDER BY bm25(doc_fts)
                LIMIT ?
            ''', (query, limit)).fetchall()
        
        return [dict(r) for r in results]

# Example: Search for imaging documentation across all tools
results = search_package_docs("multiscale deconvolution")
for r in results:
    print(f"[{r['package']}] {r['name']}")
    print(r['content'][:300])
    print()

# Example: Search CASA tasks specifically  
results = search_package_docs("calibration", packages=["casa"])
```

## Common Documentation Queries

### CASA Tasks

```python
# Imaging with tclean
search_package_docs("tclean deconvolver", ["casa"])

# Calibration (gaincal, bandpass, applycal)
search_package_docs("gaincal solint", ["casa"])

# Self-calibration workflow
search_package_docs("self-calibration", ["casa"])

# Data weighting options
search_package_docs("weighting briggs", ["casa"])
```

### WSClean

```python
# Multi-scale cleaning
search_package_docs("multiscale scales", ["wsclean"])

# Auto-masking for deep cleaning
search_package_docs("auto-mask threshold", ["wsclean"])

# Image weighting
search_package_docs("weight briggs", ["wsclean"])

# Self-calibration with MODEL_DATA
search_package_docs("predict model", ["wsclean"])
```

### AOFlagger

```python
# SumThreshold algorithm
search_package_docs("sumthreshold", ["aoflagger"])

# Lua strategy scripting
search_package_docs("execute lua strategy", ["aoflagger"])

# Python interface
search_package_docs("python aoflagger", ["aoflagger"])
```

### pyuvdata

```python
# Reading UVH5 files
search_package_docs("read_uvh5", ["pyuvdata"])

# Writing to Measurement Set
search_package_docs("write_ms", ["pyuvdata"])

# UVData object structure
search_package_docs("UVData class", ["pyuvdata"])

# Antenna information
search_package_docs("antenna_positions", ["pyuvdata"])
```

### casacore

```python
# Opening CASA tables
search_package_docs("table open", ["casacore"])

# Writing table data
search_package_docs("putcol", ["casacore"])

# Table queries
search_package_docs("query table", ["casacore"])
```

### h5py

```python
# Creating HDF5 files
search_package_docs("create_dataset", ["h5py"])

# Reading datasets
search_package_docs("read file", ["h5py"])

# Groups and attributes
search_package_docs("attrs group", ["h5py"])
```

### astropy

```python
# Coordinate transformations
search_package_docs("SkyCoord transform", ["astropy"])

# Time handling
search_package_docs("Time mjd", ["astropy"])

# Units
search_package_docs("Quantity unit", ["astropy"])
```

## Re-indexing Documentation

If package versions change or you need to update the documentation:

```bash
# Re-index all Python packages
python scripts/agents/index_external_docs.py --all

# Re-index specific Python package
python scripts/agents/index_external_docs.py --packages pyuvdata

# Re-index CLI tools (CASA, WSClean, AOFlagger)
python scripts/agents/index_cli_docs.py --package all

# Check status
python scripts/agents/index_external_docs.py --status
```

## Documentation Sources

### Python Libraries (auto-indexed)
Documentation is extracted using Python's `inspect` module from installed packages.

### CLI Tools (manually curated)
Documentation is sourced from official ReadTheDocs sites:
- **CASA**: https://casadocs.readthedocs.io/en/v6.6.0/
- **WSClean**: https://wsclean.readthedocs.io/en/latest/
- **AOFlagger**: https://aoflagger.readthedocs.io/en/latest/

## Database Location

The documentation is stored in:
```
/data/dsa110-contimg/state/db/external_docs.sqlite3
```

Schema:
- `packages` - Package metadata (name, version, doc_count, indexed_at)
- `documentation` - Full documentation entries (package, type, name, content, source)
- `doc_fts` - FTS5 full-text search index

## Integration with Agent Workflows

When the agent needs documentation for external packages:

1. **Before writing code**: Search for relevant API documentation
2. **When debugging**: Look up method signatures and parameters
3. **For best practices**: Search for usage examples in docstrings

Example agent workflow:
```
User: "How do I image with WSClean using multi-scale cleaning?"

Agent:
1. Search: search_package_docs("multiscale", ["wsclean"])
2. Find: -multiscale flag with scale-bias parameter
3. Generate correct command: wsclean -multiscale -mgain 0.8 ...
```
