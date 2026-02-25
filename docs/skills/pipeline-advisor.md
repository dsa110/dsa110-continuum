# Pipeline Expert Advisor Skill

This skill enables an AI agent to analyze the DSA-110 pipeline's calibration, imaging, and mosaicking code and suggest informed improvements based on:

1. **Indexed documentation** from CASA, WSClean, AOFlagger, and Python libraries
2. **DSA-110 science goals** (transient detection vs. continuum survey)
3. **Instrument properties** (synthesized beam, bandwidth, sensitivity)

## Science-Informed Analysis

The analyzer considers DSA-110's specific properties:

| Property | Value | Impact |
|----------|-------|--------|
| Synthesized beam | ~12" | Pixel scale, source detection |
| Primary beam | 4° | Wide-field gridding requirements |
| Frequency | 1.31-1.50 GHz | RFI environment, spectral modeling |
| Fractional bandwidth | 14% | nterms for wideband imaging |
| Detection threshold | 5σ | Noise vs artifact trade-offs |

### Science Mode

DSA-110 operates in **survey mode**: daily mosaics with photometry on ~10⁴ compact sources to detect ESEs (Extreme Scattering Events) via flux variability.

**Survey Mode Priorities**:
- Speed (daily cadence)
- Photometric repeatability (consistent fluxes)
- Point-source sensitivity (compact sources only)

**Deprioritized for ESE Detection**:
- Multiscale (no extended sources)
- High nterms (spectral detail less critical than repeatability)
- Deep cleaning (10 mJy sources don't need it)

### GPU Acceleration (h17)

h17 has 2× RTX 2080 Ti GPUs (11 GB each). GPU imaging provides **4× speedup**:

| Image Size | Gridder | Mode |
|------------|---------|------|
| ≤4096 | IDG | `gpu` |
| ≤8192 | IDG | `hybrid` |
| >8192 | wgridder | CPU |

**Default production command**:
```bash
docker run --rm --gpus all dsa110-contimg:gpu \
    wsclean -gridder idg -idg-mode hybrid ...
```

For detailed GPU guidance, see: `GPU_CONTEXT.md`

## Purpose

Combine knowledge of:
1. **Current pipeline implementation** in `backend/src/dsa110_contimg/core/`
2. **Best practices from external tools** indexed in `state/db/external_docs.sqlite3`
3. **Radio astronomy domain expertise** to identify improvement opportunities

## How to Use This Skill

### 1. Analyze a Specific Pipeline Component

When asked to improve calibration, imaging, or mosaicking:

```python
from pathlib import Path
import sqlite3

DB_PATH = Path("/data/dsa110-contimg/state/db/external_docs.sqlite3")

def get_best_practices(topic: str, packages: list[str] = None) -> list[dict]:
    """Retrieve best practices from indexed documentation.
    
    Parameters
    ----------
    topic : str
        Topic to search (e.g., "self-calibration", "multiscale", "bandpass")
    packages : list of str
        Limit to specific packages (e.g., ["casa", "wsclean"])
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        
        if packages:
            placeholders = ",".join("?" * len(packages))
            results = conn.execute(f'''
                SELECT d.package, d.name, d.content, d.type
                FROM doc_fts f
                JOIN documentation d ON f.rowid = d.id
                WHERE doc_fts MATCH ?
                  AND d.package IN ({placeholders})
                ORDER BY bm25(doc_fts)
                LIMIT 10
            ''', (topic, *packages)).fetchall()
        else:
            results = conn.execute('''
                SELECT d.package, d.name, d.content, d.type
                FROM doc_fts f
                JOIN documentation d ON f.rowid = d.id
                WHERE doc_fts MATCH ?
                ORDER BY bm25(doc_fts)
                LIMIT 10
            ''', (topic,)).fetchall()
        
        return [dict(r) for r in results]
```

### 2. Compare Current Implementation to Documentation

For any pipeline module, follow this pattern:

1. **Read the current code** to understand what parameters/techniques are used
2. **Query the documentation** for those specific features
3. **Identify gaps** between current usage and documented best practices
4. **Suggest improvements** with specific parameter recommendations

## Improvement Analysis Framework

### Calibration (`core/calibration/`)

**Key files to analyze:**
- `calibration.py` - Main calibration workflow
- `presets.py` - Calibration presets (standard, fast, selfcal, etc.)
- `bandpass_diagnostics.py` - Bandpass analysis
- `selfcal.py` - Self-calibration logic
- `flagging_adaptive.py` - RFI flagging

**Documentation queries:**
```python
# CASA calibration tasks
get_best_practices("gaincal solint combine", ["casa"])
get_best_practices("bandpass minsnr", ["casa"])
get_best_practices("self-calibration workflow", ["casa"])
get_best_practices("polcal leakage", ["casa"])

# Flagging
get_best_practices("rflag tfcrop", ["casa"])
get_best_practices("sumthreshold high_pass_filter", ["aoflagger"])
```

**Common improvement areas:**
- Solution intervals (`solint`) optimization
- SNR thresholds (`minsnr`) tuning
- Reference antenna selection strategies
- combine parameter usage (spw, scan, field)
- Pre-bandpass phase correction
- Polarization calibration if supported

### Imaging (`core/imaging/`)

**Key files to analyze:**
- `params.py` - ImagingParams dataclass
- `worker.py` - WSClean/tclean execution
- `masks.py` - Mask generation
- `catalog_tools.py` - Catalog-based seeding

**Documentation queries:**
```python
# WSClean parameters
get_best_practices("multiscale scale-bias", ["wsclean"])
get_best_practices("auto-mask auto-threshold", ["wsclean"])
get_best_practices("mgain major cycle", ["wsclean"])
get_best_practices("gridder wstacking idg", ["wsclean"])
get_best_practices("channels-out join-channels", ["wsclean"])

# CASA tclean
get_best_practices("tclean deconvolver mtmfs", ["casa"])
get_best_practices("weighting briggs robust", ["casa", "wsclean"])
get_best_practices("wide-field wproject awproject", ["casa"])
```

**Common improvement areas:**
- Multi-scale cleaning for extended emission
- Auto-masking for deep cleaning
- Spectral index imaging (nterms > 1, channels-out)
- Gridder selection for wide-field
- Weighting optimization (robust parameter)
- UV taper for sensitivity to extended emission

### Mosaicking (`core/mosaic/`)

**Key files to analyze:**
- `tiers.py` - Tier configurations (Quicklook, Science, Deep)
- `builder.py` - Core mosaic building
- `wsclean_mosaic.py` - WSClean joint imaging
- `jobs_wsclean.py` - Mosaic jobs

**Documentation queries:**
```python
# Joint imaging
get_best_practices("mosaic joint imaging", ["casa", "wsclean"])
get_best_practices("primary beam pblimit", ["casa", "wsclean"])

# Spectral imaging
get_best_practices("wideband spectral", ["casa", "wsclean"])
```

**Common improvement areas:**
- Primary beam correction strategies
- Joint deconvolution vs linear mosaicking
- Weighting across pointings
- Phase center handling (chgcentre)

## ℹ️ CRITICAL: DSA-110 Pipeline Workflow - TWO PHASESHIFTS

> **There are TWO separate phaseshift operations in the DSA-110 workflow!**
> Both produce valid MS files - choose based on your imaging needs.

### The Two Phaseshifts

| Phaseshift | Target | Purpose | Output |
|------------|--------|---------|--------|
| **For Calibration** | Calibrator position (e.g., 3C454.3) | Solve BP/gains with calibrator at phase center | `*_cal.ms` (temporary) |
| **For Imaging** | Median meridian | Align all 24 fields for coherent imaging | `*_meridian.ms` |

### Complete Workflow Diagram

```
Original MS (24 fields at different meridian RAs)
        │
        ├──▶ [1] CALIBRATION PATH
        │         ↓
        │    Split calibrator transit fields
        │         ↓
        │    Phaseshift to CALIBRATOR position
        │         ↓
        │    Solve BP + gains → cal tables
        │         ↓
        │    (cal_staging/*_cal.ms - for SOLVING ONLY, do NOT image!)
        │
        └──▶ [2] IMAGING PATH
                  ↓
             Phaseshift ALL fields to MEDIAN MERIDIAN
                  ↓
             Apply cal tables (from step 1)
                  ↓
             *_meridian.ms (for IMAGING)
                  ↓
             WSClean imaging
```

### Common Mistake: Imaging the Wrong MS

| MS | RA Spread | What It's For | Can Image? |
|----|-----------|---------------|------------|
| `observation.ms` | ~4000 arcsec | Original data | ❌ NO |
| `*_cal.ms` | ~0 arcsec | Calibration solving | ❌ NO (wrong center) |
| `*_meridian.ms` | ~0 arcsec | **Imaging** | ✅ YES |

### Correct Code Sequence

```python
from dsa110_contimg.core.calibration.runner import phaseshift_ms
from dsa110_contimg.core.calibration.applycal import apply_to_target
from dsa110_contimg.core.imaging.cli_imaging import image_ms

# [CALIBRATION PATH - already done by calibrator pipeline]
# cal_ms = phaseshift_ms(ms, mode="calibrator")  # To calibrator position
# Solve BP/gains on cal_ms → produces bp_table, g_table

# [IMAGING PATH - what you must do before imaging]
# Step 1: Create median-meridian phaseshifted MS
meridian_ms, _ = phaseshift_ms(
    ms_path="observation.ms",  # Original, NOT cal_ms!
    mode="median_meridian",    # To median meridian, NOT calibrator!
)

# Step 2: Apply existing cal tables to meridian MS
apply_to_target(
    ms_target=meridian_ms,
    gaintables=[bp_table, g_table],
)

# Step 3: Image the correctly-phaseshifted MS
image_ms(
    ms_path=meridian_ms,  # NOT original, NOT cal_ms!
    imagename="/stage/output",
    ...
)
```

### Common Mistakes to Avoid

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Image original MS (no phaseshift) | Phase incoherence | Run `phaseshift_ms(mode="median_meridian")` first |
| Image `*_cal.ms` instead of `*_meridian.ms` | Image centered on calibrator | Use `*_meridian.ms` for science imaging |
| Skip flux model | ~100x flux scale error | Run `populate_model_from_catalog()` |
| Apply cal to wrong MS | Wrong data calibrated | Apply cal to meridian MS |
| CORRECTED/RAW ratio ≈ 1.0 | No flux scaling | Check model was populated |

### ℹ️ Calibration Order (DSA-110)

DSA-110 has stable delays — K-calibration is NOT required:

```
1. populate_model_from_catalog() - Flux model from VLA catalog
2. gaincal(calmode='p')          - Pre-BP phase (optional)
3. bandpass(solnorm=True)        - Frequency-dependent gains
4. gaincal(calmode='ap')         - Amp+phase gains
5. applycal()                    - Apply all corrections
```

**Empirical Validation** (2026-02-02):
- Phase slopes <3° across band (K-cal threshold: 30°)
- Overall phase std ~27° — acceptable for DSA-110

**Diagnostic: Check phase coherence after calibration:**
```python
import numpy as np
from casacore.tables import table

with table(ms_path, readonly=True, ack=False) as tb:
    corr = tb.getcol('CORRECTED_DATA', nrow=10000)
    flag = tb.getcol('FLAG', nrow=10000)

good = ~flag
phase_std = np.std(np.angle(corr[good], deg=True))
print(f"Phase std: {phase_std:.1f}°")
# DSA-110 typical: 25-30° (acceptable)
# Needs investigation: >50°
```

## Example Analysis Workflow

### Task: "Improve self-calibration in the pipeline"

**Step 1: Query documentation**
```python
results = get_best_practices("self-calibration", ["casa", "wsclean"])
for r in results:
    print(f"[{r['package']}] {r['name']}")
    print(r['content'][:500])
```

**Step 2: Read current implementation**
```python
# Read selfcal.py
with open("backend/src/dsa110_contimg/core/calibration/selfcal.py") as f:
    code = f.read()
```

**Step 3: Compare and identify gaps**

Documentation says:
- Start with long solint, shorten gradually
- Start with phase-only ('p'), then amplitude+phase ('ap')
- Use combine='spw' if SNR is low
- Monitor solution stability

Current code does:
- [Check current solint progression]
- [Check calmode progression]
- [Check combine settings]

**Step 4: Generate recommendations**
- If solint progression is missing → suggest iterative shortening
- If starting with 'ap' → suggest starting with 'p'
- If no solution monitoring → suggest adding convergence checks

## Quick Reference: Key Parameters

### Calibration Parameters to Check

| Parameter | CASA Name | Current Default | Documentation Guidance |
|-----------|-----------|-----------------|------------------------|
| Solution interval | `solint` | "inf" | Start long, shorten iteratively |
| Calibration mode | `calmode` | "ap" | Start "p", add "a" later |
| Combine axes | `combine` | "" | Use "spw" for low SNR |
| Min SNR | `minsnr` | 3.0 | 3.0 typical, 5.0 for BP |
| Reference antenna | `refant` | "103" | Choose most stable |

### Imaging Parameters to Check

| Parameter | WSClean Flag | Current Default | Documentation Guidance |
|-----------|--------------|-----------------|------------------------|
| Major gain | `-mgain` | 0.8 | 0.8 typical for Cotton-Schwab |
| Multi-scale | `-multiscale` | off | Enable for extended emission |
| Scale bias | `-multiscale-scale-bias` | 0.6 | 0.6 balanced, 0.4-0.5 for extended |
| Auto-mask | `-auto-mask` | off | 5σ for mask creation |
| Auto-threshold | `-auto-threshold` | off | 1-3σ for stopping |
| Weighting | `-weight briggs` | 0.5 | -0.5 to 0.5 typical |

### RFI Flagging Parameters

| Parameter | AOFlagger Function | Common Values |
|-----------|-------------------|---------------|
| x_threshold | `sumthreshold()` | 1.0 (lower = more aggressive) |
| y_threshold | `sumthreshold()` | 1.0 |
| Filter size | `high_pass_filter()` | 21, 31 pixels |
| Extension | `scale_invariant_rank_operator()` | 0.2 |

## Generating Improvement Reports

For comprehensive analysis, use this template:

```markdown
## [Component] Improvement Analysis

### Current Implementation
- File: `backend/src/dsa110_contimg/core/[component]/[file].py`
- Key parameters: [list current settings]

### Documentation Comparison
| Feature | Current | Best Practice | Gap |
|---------|---------|---------------|-----|
| [param] | [value] | [recommended] | [description] |

### Recommendations
1. **[Priority: High/Medium/Low]** - [Specific change]
   - Rationale: [Why from documentation]
   - Implementation: [How to change]

### Code Changes
```python
# Suggested modification
```
```

## Related Files

- **Documentation index**: `/data/dsa110-contimg/state/db/external_docs.sqlite3`
- **Calibration code**: `backend/src/dsa110_contimg/core/calibration/`
- **Imaging code**: `backend/src/dsa110_contimg/core/imaging/`
- **Mosaic code**: `backend/src/dsa110_contimg/core/mosaic/`
- **Self-cal code**: `backend/src/dsa110_contimg/core/selfcal/`
- **Science context**: `.agent/skills/pipeline-advisor/SCIENCE_CONTEXT.md`
- **GPU context**: `.agent/skills/pipeline-advisor/GPU_CONTEXT.md`
- **Full GPU skill**: `.agent/skills/gpu-acceleration/SKILL.md`

## Automated Analysis Tool

Use the analyzer CLI to generate science-informed recommendations:

```bash
# ESE survey mode (default - daily mosaics for variability monitoring)
python scripts/agents/analyze_pipeline.py --science-mode survey

# Calibrator mode (for calibrator observations)
python scripts/agents/analyze_pipeline.py --science-mode calibrator

# Specific component analysis
python scripts/agents/analyze_pipeline.py --component imaging --science-mode survey

# Generate detailed report
python scripts/agents/analyze_pipeline.py --science-mode survey --output report.md
```

### ESE Survey Mode Settings

DSA-110 operates in **survey mode** exclusively, using the transit nature of the instrument
to build daily mosaics from ~5-minute field tiles. ESE (Extreme Scattering Events) are
detected via photometry against catalog positions in these mosaics.

| Recommendation | Priority | Rationale |
|---------------|----------|-----------|
| Deconvolver | 🟢 hogbom | Point sources only (compact sources ~10⁴) |
| Auto-mask | 🔴 High | Speeds cleaning, minimizes RFI artifacts |
| nterms | 🟢 1 | Speed over spectral fidelity |
| Multiscale | 🟢 Off | Point source detections only |
| UV taper | 🟢 Off | Maximum resolution for position accuracy |
