## DSA-110 Continuum Imaging Pipeline — Overview

This is a **radio astronomy continuum imaging pipeline** for the DSA-110 (Deep Synoptic Array, 110 antennas) telescope at OVRO (Owens Valley Radio Observatory). Its science goal is to detect and monitor **variable (and possibly transient) compact radio sources** — in search of Extreme Scattering Events (ESEs) and other flux-variable phenomena — by producing daily sky mosaics and measuring per-source flux with daily cadence for weeks/months at a time.

It was ported and rewritten from the older `dsa110-contimg` codebase.

---

### The Instrument

DSA-110 is a **meridian drift-scan transit array** — it doesn't track sources, it lets the sky drift through its fixed beam. Key numbers:
- **117 antennas**, 4.65 m dishes, L-band (1.31–1.50 GHz, 187.5 MHz bandwidth)
- **16 subbands × 48 channels**, 12.885 s integrations
- Each "tile" is a ~5-minute transit of a field through the meridian
- Raw data comes as **HDF5 files** (one per subband per timestamp) stored at `/data/incoming/`

---

### Data Flow (End to End)

```
HDF5 files (16 subbands × N timestamps)
        │
        ▼
  [conversion/]  ─ UVH5 → Measurement Set (MS)
        │           PyUVData ingest, UVW reconstruction, phase centre assignment
        ▼
  [calibration/] ─ Flagging (AOFlagger + MAD clip) → Bandpass (B) + Gain (G) solve → applycal
        │           Calibrator: 3C454.3 (blazar); self-cal via selfcal.py
        ▼
  [imaging/]     ─ Phaseshift → WSClean (wgridder/IDG) → FITS tile image
        │           4800×4800 px, 3 arcsec/px, primary beam correction (EveryBeam)
        ▼
  [mosaic/]      ─ 12 tiles → hourly-epoch mosaic
        │           QUICKLOOK: image-domain Airy-disk coadd
        │           SCIENCE/DEEP: visibility-domain joint WSClean (IDG)
        ▼
  [photometry/]  ─ Forced photometry at NVSS/RACS catalog positions (Condon matched-filter)
        │           Variability metrics: η (reduced χ²), Vs (significance), m (modulation index)
        ▼
  [products/]    ─ mosaics/{date}/*_mosaic.fits, *_forced_phot.csv → light curves
```

---

### Package Structure (11 submodules)

| Module | Role |
|---|---|
| `conversion/` | HDF5→MS via PyUVData; UVW reconstruction; merge SPWs for IDG |
| `calibration/` | Bandpass/gain solve, applycal, flagging, self-cal, phaseshift, presets, QA |
| `imaging/` | WSClean/CASA tclean interface; `ImagingParams`; sky model seeding |
| `mosaic/` | QUICKLOOK + SCIENCE/DEEP mosaicking tiers; tile scheduling |
| `photometry/` | Forced photometry, Condon errors, ESE detection, variability metrics |
| `catalog/` | Source catalog management (NVSS, RACS, FIRST, VLA cal list); SQLite backend |
| `qa/` | Delay validation, image quality metrics, pipeline QA hooks |
| `simulation/` | Synthetic UVH5 generation for testing |
| `visualization/` | Diagnostic plots (bandpass, UV coverage, calibration, mosaics, light curves) |
| `validation/` | MS/image validators, storage checks |
| `evaluation/` | Pipeline stage evaluation harness |

---

### Key Scripts

| Script | Purpose |
|---|---|
| `run_pipeline.py` | Single-tile reference run: phaseshift → applycal → WSClean → check flux |
| `mosaic_day.py` | Process all tiles for one date → produce a full-day mosaic |
| `batch_pipeline.py` | Full orchestration: all tiles → hourly-epoch mosaics → forced photometry |
| `source_finding.py` | BANE + Aegean on mosaics → blind source catalog |
| `forced_photometry.py` | Standalone forced photometry against reference catalog |
| `inventory.py` | HDF5 data inventory with conversion status |

---

### The Science Target

The science target is **multi-day light curves** for all detectable sources in the DSA-110 sky strip — enabling detection of ESEs, flares, and slowly-varying AGN. The pipeline exists to go from raw HDF5 → `products/lightcurves/lightcurves_*.csv` with Mooley variability metrics per source.

---