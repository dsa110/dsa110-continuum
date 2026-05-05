# DSA-110 Instrument Reference

Comprehensive instrument specification for the Deep Synoptic Array 110 (DSA-110),
compiled from primary literature and the official `dsa110/dsa110-antpos` repository.
This document is authoritative for all pipeline code in this repository.

**Last verified:** 2026-04-13  
**Primary sources:**

| Source | Reference |
|---|---|
| Connor et al. 2025 (NSFRB pipeline paper) | arxiv:2510.18136, DOI:10.1088/1538-3873/ae390e |
| Sherman et al. 2024 (Polarimetry of 25 FRBs) | DOI:10.3847/1538-4357/ad275e |
| Law et al. 2024 (First FRB & Host Galaxy Catalog) | DOI:10.3847/1538-4357/ad3736 |
| Ravi et al. 2023 (50 Mpc FRB / CGM constraints) | DOI:10.3847/1538-3881/adc725 |
| dsa110/dsa110-antpos (antenna positions) | https://github.com/dsa110/dsa110-antpos |
| dsa110-continuum/docs/pipeline-specs.md | This repository |

---

## 1. Overview

The DSA-110 is a 96-antenna (96-element active) drift-scan radio interferometer
operated by Caltech at the Owens Valley Radio Observatory (OVRO) in Big Pine,
California. It was designed to detect and interferometrically localize fast radio
bursts (FRBs) in real time at 1.4 GHz. Its defining characteristic is the
**meridian transit mode**: the array does not slew in azimuth; instead the sky
drifts through a fixed primary beam.

---

## 2. Site and Reference Frame

| Parameter | Value | Source |
|---|---|---|
| Site | Owens Valley Radio Observatory (OVRO), Big Pine, CA | pipeline-specs.md |
| Geodetic latitude | 37.2339° N | pipeline-specs.md |
| Geodetic longitude | 118.2825° W | pipeline-specs.md |
| Geodetic altitude | 1222 m | pipeline-specs.md |
| T-center latitude | 0.6498455107238486 rad (37.2334°N) | antpos/utils.py |
| T-center longitude | −2.064427799136453 rad (−118.2834°W) | antpos/utils.py |
| T-center height | 1188.0519 m | antpos/utils.py |

The T-center (intersection of E-W and N-S arms) is at approximately
lat = 37.2334°N, lon = −118.2834°W.

---

## 3. Array Geometry

### 3.1 Array Shape

The DSA-110 is a **T-shaped array** consisting of three components:

| Component | Count (total slots) | Active | Notes |
|---|---|---|---|
| East-West (E-W) arm | 51 slots (DSA001–DSA051) | **47** | Verified from H17 HDF5 metadata (2026-05-05). Inactive: DSA-{10, 21, 22, 23}. |
| North-South (N-S) arm | 51 slots (DSA052–DSA102) | **35** | Verified from H17 HDF5 metadata. Active range: DSA068–DSA102 (DSA052–DSA067 inactive). |
| Outriggers | 15 slots (DSA103–DSA117) | **14** | Verified from H17 HDF5 metadata. Inactive: DSA-117. |
| **Total (CSV slots)** | **117** | — | CSV has 117 rows (headerline=5) |
| **Active antennas** | — | **96** | `Nants_data=96` in real H17 HDF5 metadata (`/data/incoming/2026-01-25T*_sb00.hdf5`). Verified 2026-05-05; full active list in `outputs/ground_truth_audit_2026-05-04/active_antennas_2026-01-25.md`. |

> **Important:** The CSV in this repository
> (`dsa110_continuum/simulation/pyuvsim/antennas.csv`) contains all 117 allocated
> position slots. Pipeline code that models thermal noise uses 96 active antennas
> (47 E-W + 35 N-S + 14 outriggers, verified 2026-05-05 against H17 HDF5 metadata).
> The `ant_ids_mid.csv` in the `dsa110-antpos` repo is a **66-antenna commissioning
> configuration** (48 EW + 3 near the T-junction + 15 outriggers) used during an
> earlier science run — it does NOT represent the full operational array.

### 3.2 E-W Arm

- **Station numbers:** DSA001–DSA051 (51 slots)
- **Latitude:** 37.233375179° (constant, from RevF design)
- **Spacing:** 5.750 m (uniform nominal design spacing)
- **Arm extent:** ±198.375 m from T-center (total baseline 396.75 m)
- **Geometry:** Antennas are nearly collinear East-West; there is a very slight
  North tilt (<21 m over 396 m, ~3°) due to terrain.
- **Design gaps in CSV:** Four positions have larger-than-nominal spacings
  (at DSA022→DSA023: 63.1 m gap implying ~10 nominal slots missing; and three
  22.96 m gaps around DSA029, DSA030, DSA035). These represent unbuilt pads
  at the designstage — they do not represent offline antennas.

### 3.3 N-S Arm

- **Station numbers:** DSA052–DSA102 (51 slots)
- **Longitude:** −118.283405115° (constant, from RevF design)
- **Spacing:** 8.653 m (uniform nominal design spacing)
- **Arm extent:** 7.343–440.000 m north of the T-junction (i.e. north-only)
- **Geometry:** Strictly north-south; near-uniform spacing confirmed by the
  consecutive-separation analysis (8.630–8.641 m range).

### 3.4 Outriggers

- **Station numbers:** DSA103–DSA117 (15 slots)
- **Distribution:** Scattered around the T-core at baselines from ~537 m to ~2043 m
- **Active (per paper):** 14 of 15 in real-time pipeline; all 15 are built
- **Purpose:** Long-baseline localization of FRBs; not used in real-time continuum
  imaging or the NSFRB pipeline (included only for detailed manual candidate inspection)

### 3.5 Baseline Summary

| Subset | Max baseline | Notes |
|---|---|---|
| E-W arm only | ~396 m | Collinear; limited N-S resolution |
| N-S arm only | ~440 m from T-junction | Collinear |
| 82-antenna dense core (b < 485 m) | ~485 m | Used in real-time NSFRB imaging |
| Full array (incl. outriggers) | ~2.6 km | Used for FRB localization |
| Minimum baseline (gridded) | 20 m | b_min cut in NSFRB UV gridding |

### 3.6 Position Data Files

| File | Format | Notes |
|---|---|---|
| `antpos/data/DSA110_Station_Coordinates.csv` | lat/lon/elevation WGS84 | 117 stations; canonical; updated 2022-02-15 |
| `antpos/data/DSA110_positions_RevF.csv` | lat/lon by arm with design spacings | 51 EW + 51 NS + 2 reference markers |
| `dsa110_continuum/simulation/pyuvsim/antennas.csv` | east_m, north_m, up_m (local ENU projected) | 117 rows; bundled projected-coordinate alternate; the default harness path uses `DSA110_Station_Coordinates.csv` via `load_geodetic_enu()`. |

**Reading `antennas.csv`:** The columns `east_m`, `north_m`, `up_m` in the
simulation CSV are not raw ECEF; they are projected local coordinates with
large baseline values (~411k, −375k, −24k). To compute relative ENU positions
for simulation, subtract the reference antenna (DSA001) or the T-center.

---

## 4. Receiver and Feed

| Parameter | Value | Source |
|---|---|---|
| Dish diameter | **4.65 m** (measured) | pipeline-specs.md (dsa110_measured_parameters.yaml) |
| Feed type | Dual-linear polarization | — |
| Polarizations | XX, YY (linear) | pipeline-specs.md |
| Wavelength at 1.4 GHz | ~21 cm | — |

> The correct dish diameter is **4.65 m**. The value 4.7 m sometimes appearing
> in older code is incorrect; all primary beam models, aperture calculations,
> and seeding radii must use 4.65 m.

---

## 5. Frequency Configuration

| Parameter | Value | Source |
|---|---|---|
| Observing band | L-band (~1.28–1.53 GHz nominal) | pipeline-specs.md |
| **Science frequency range** | **1311.25–1498.75 MHz** | pipeline-specs.md (confirmed) |
| **Total science bandwidth** | **187.5 MHz** | pipeline-specs.md |
| Correlator window | 250 MHz nominal | pipeline-specs.md |
| **Number of subbands (SPWs)** | **16** | pipeline-specs.md |
| **Subband bandwidth** | **11.71875 MHz** | 187.5 MHz / 16 |
| **Channels per subband** | **48** | pipeline-specs.md |
| **Total channels** | **768** (16 × 48) | pipeline-specs.md |
| **Channel width** | **244.14 kHz** | 11.71875 MHz / 48 |
| Raw spectral resolution | 384 channels × 30.5 kHz | pipeline-specs.md |
| Subband naming | `{timestamp}_sb{00..15}.hdf5` | pipeline-specs.md |

**NSFRB search uses:** 8 sub-bands × 1.46 MHz = 11.68 MHz effective bandwidth
(downsampled for the real-time transient search; Connor et al. 2025).

> **Do not use 250 MHz in any pipeline calculation.** The science-grade band is
> 187.5 MHz (1311.25–1498.75 MHz). The 250 MHz figure is the raw correlator
> window; only the 187.5 MHz sub-window is usable for science.

---

## 6. Timing and Observing Mode

| Parameter | Value | Source |
|---|---|---|
| Observing mode | **Meridian drift-scan** (transit instrument; no azimuthal slew) | Both |
| **Integration time (pipeline dump)** | **12.885 s** | pipeline-specs.md |
| Fringe-stopping interval | **~3.35 s** | Connor et al. 2025 |
| Fringe timescale formula | t_F ≈ 1.05 s / cos(δ) | Connor et al. 2025 |
| Fringe timescale at δ = 71.6° | t_F ≈ 3.36 s | Connor et al. 2025 |
| **Fields (timesteps) per tile** | **24** | pipeline-specs.md |
| **Tile duration** | **~5.15 min** (24 × 12.885 s) | pipeline-specs.md |
| Tiles per mosaic | 12 | pipeline-specs.md (PIPELINE CHOICE) |
| Mosaic stride | 6 tiles (50% overlap) | pipeline-specs.md (PIPELINE CHOICE) |
| Dec tolerance between tiles | 1.0° | pipeline-specs.md (PIPELINE CHOICE) |
| Survey declination | **Variable; read from HDF5 metadata per observation** | pipeline-specs.md |

> The survey declination is **not fixed**. The telescope can slew in elevation.
> No pipeline code should assume a fixed declination. Read the declination from
> each tile's HDF5 metadata.

---

## 7. Sensitivity and Noise

### 7.1 Single-Antenna SEFD

From `dsa110-antpos/antpos/utils.py`:

```python
sefd = 6500.0 / (1.*nant)  # Jy  — written as total_SEFD / N_ant
# Equivalently, single-antenna SEFD = 6500 Jy
```

| Parameter | Value | Source |
|---|---|---|
| **Single-antenna SEFD** | **6500 Jy** | antpos/utils.py |
| Dish diameter | 4.65 m | pipeline-specs.md |
| Aperture efficiency (η) | ~0.70 (implied) | Derived |
| Effective collecting area | ~11.9 m² (η = 0.70) | Derived |
| Implied Tsys | **~28 K** | Derived: SEFD = 2kTsys/Aeff |
| Published Tsys | ~25 K | pipeline-specs.md ADS abstract note |

The implied Tsys ≈ 28 K is consistent with the ~25 K stated in the literature;
the small difference reflects uncertainty in the aperture efficiency.

### 7.2 Image Sensitivity

From Connor et al. 2025 (NSFRB pipeline, 82-core antennas, R = −2 weighting):

| Context | Value |
|---|---|
| Theoretical σ_th (R = −2) | **~40 mJy** |
| Measured σ_srch (median) | **~50 mJy** |
| Range during DN-GPS test | 30–100 mJy |
| 90% completeness limit (25σ) | ~1200 mJy (~160 Jy ms fluence) |

These figures apply to the NSFRB search configuration (11.68 MHz bandwidth, 3.35 s
integration, 82-core antennas). For the continuum imaging pipeline (187.5 MHz, 12.885 s
dump rate, all 96 antennas, Briggs R = 0.5), the noise per image is lower.

---

## 8. Beams and Imaging Parameters

### 8.1 Primary Beam

| Parameter | Value | Source |
|---|---|---|
| Primary beam model | Airy disk: (2·J₁(x)/x)², D = 4.65 m | pipeline-specs.md |
| **FWHM (at 1.4 GHz)** | **~4.1° circular** | pipeline-specs.md |
| FWHM (NSFRB paper) | **~1.5°/cos(δ) × 1.5°** | Connor et al. 2025 |
| PB cutoff (quicklook) | 0.1 | pipeline-specs.md |
| PB cutoff (imaging) | pblimit = 0.2 | pipeline-specs.md |

> The NSFRB paper's "1.5°" refers to the real-time search region, not the full
> primary beam FWHM. The full FWHM for a 4.65 m dish at 1.4 GHz is ~4.1°.
> These are consistent: the search is conducted within the inner 1.5° where
> sensitivity is highest.

### 8.2 Synthesized Beam (Core Array)

| Parameter | Value | Source |
|---|---|---|
| Synthesized beam (82-core, R = −2) | **~31 arcsec** | Connor et al. 2025 |
| Pixel scale (NSFRB UV grid) | 10.3″/cos(δ) × 10.3″ | Connor et al. 2025 |
| Pixels per synthesized beam (NSFRB) | ~3 | Connor et al. 2025 |
| Synthesized beam (continuum, R = 0.5) | **~10–15 arcsec** (estimated) | pipeline-specs.md |
| Pixel scale (continuum imaging) | 3.0 arcsec/px | pipeline-specs.md |

### 8.3 Real-Time NSFRB Search Image Parameters

| Parameter | Value | Source |
|---|---|---|
| UV grid size | 175 × 175 pixels | Connor et al. 2025 |
| Image FoV (search region) | 0.5°/cos(δ) × 0.5° | Connor et al. 2025 |
| Baselines gridded | b > 20 m, b < 485 m (82-core) | Connor et al. 2025 |
| UV weighting | Approximately uniform (Briggs R = −2) | Connor et al. 2025 |

### 8.4 Continuum Imaging Parameters (Pipeline)

| Parameter | Value | Source |
|---|---|---|
| Image size | 4800 × 4800 px | pipeline-specs.md |
| Cell size | 3.0 arcsec/px | pipeline-specs.md |
| FoV (4800 × 3 arcsec) | 4.0° | pipeline-specs.md |
| Weighting | Briggs R = 0.5 (standard), R = 0.0 (survey) | pipeline-specs.md |
| UV range cutoff | >1 klambda (min baseline for gridding) | pipeline-specs.md |
| Backend | WSClean (IDG gridder) | pipeline-specs.md |

---

## 9. Calibration Reference

### 9.1 Strategy

Each 5.15-minute transit tile receives its own calibration:
one bandpass table (B) and one complex gain table (G), solved from
the calibrator transit and applied to the science tile.

| Parameter | Value | Notes |
|---|---|---|
| Calibrator | 3C454.3 (commissioning) | Flat-spectrum variable blazar |
| Flux model | Static SQLite catalog lookup | Not a Perley-Butler primary standard |
| Pre-BP phase solve | True | 60 s, phase only, >1 klambda |
| Delay solve (K) | False | Phase slopes < 3° empirically |
| BP channels/SPW | 48 per SPW | Per-SPW solutions (16 SPWs) |
| bp_combine_spw | **False** | Each subband has distinct bandpass shape |
| fillgaps | 3 channels | ~730 kHz interpolation for RFI gaps |
| minblperant | 4 | Conservative guard |
| Reference antenna | 103 (default) | Eastern outrigger first |
| Gain solint | "inf" | One solution per tile |

### 9.2 Known RFI Features

Narrowband RFI at ~1350, ~1395, ~1405, ~1422, ~1475 MHz leaves flagged channels
that require the `fillgaps=3` interpolation in bandpass calibration.

---

## 10. Data Format

| Parameter | Value | Notes |
|---|---|---|
| Raw data format | HDF5 | One file per subband per timestamp |
| Subband naming | `{timestamp}_sb{00..15}.hdf5` | |
| UVW reconstruction | chgcentre (WSClean); CASA phaseshift as fallback | |
| pyuvdata dtype | float64 (explicit cast from float32 required) | pyuvdata HDF5 bug |
| run_check | False + strict_uvw_antpos_check=False | DSA-110 files fail default validation |

---

## 11. Active Antenna Configuration

### 11.1 Operational Array (as of 2025; Connor et al. 2025)

The repo treats DSA-110 as a **96 active antenna** array for operational noise
and QA calculations. The per-arm breakdown was resolved on 2026-05-05 from real
H17 HDF5 metadata (`/data/incoming/2026-01-25T00:00:10_sb00.hdf5`):

| Component | Active count | Station number range |
|---|---|---|
| E-W core arm | **47** | DSA001–DSA051 (inactive: 10, 21, 22, 23) |
| N-S arm | **35** | DSA068–DSA102 (DSA052–DSA067 inactive) |
| Outriggers | **14** | DSA103–DSA116 (DSA117 inactive) |
| **Total** | **96** | — |

The full active station-number list is in
`outputs/ground_truth_audit_2026-05-04/active_antennas_2026-01-25.md`.

### 11.2 Earlier Commissioning Configurations

- **64-antenna (pre-N-S arm):** Historical E-W-only commissioning phase associated
  with early DSA-110 science runs; per-arm active counts for that era are not
  reconstructible from the present H17 metadata snapshot.
- **66-antenna (`ant_ids_mid.csv`):** 48 EW + 3 near T-junction + 15 outriggers;
  a specific correlator configuration used for a particular observing campaign.
  This file does NOT represent the full operational 96-antenna array.

### 11.3 Using Antenna Positions in Code

The simulation harness should load real positions from the CSV:

```python
import csv, math

def load_antenna_enu(csv_path, active_count=None):
    """
    Load DSA-110 antenna ENU positions from the simulation CSV.
    The CSV contains 117 rows (allocated slots); only 96 are active.
    Returns arrays of (east_m, north_m, up_m) relative to DSA001.
    
    If active_count is None, returns all 117 rows.
    If active_count=96, returns the first 96 rows (covers E-W, N-S core, and some outriggers;
    not the precise 96-active selection — use with caution for science).
    """
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((
                float(row['east_m']),
                float(row['north_m']),
                float(row['up_m']),
            ))
    # Reference: DSA001
    e0, n0, u0 = rows[0]
    enu = [(e - e0, n - n0, u - u0) for (e, n, u) in rows]
    if active_count is not None:
        enu = enu[:active_count]
    return enu
```

For the most physically correct simulation, use all 117 entries (which includes
all built positions) and let the UV gridder/imager naturally handle the geometry.
The harness does not currently implement inactive-slot zero weighting; use a
real active-antenna list from HDF5/MS metadata if a 96-active simulation is
required.

---

## 12. Known Discrepancies and Corrections

| Location | Was | Should be | Status |
|---|---|---|---|
| `pipeline-specs.md` §1 "Total antennas" | 117 | 96 active (117 allocated) | **Fixed in this update** |
| `pipeline-specs.md` §1 "Core antennas" | 102 | 82 dense core (b < 485m) | **Fixed in this update** |
| `pipeline-specs.md` §1 "Outrigger antennas" | 15 (IDs 103–117) | 14 active outriggers; 15 allocated slots | **Fixed in this update** |
| `harness.py` `_make_antenna_enu()` | Generates random 1D E-W positions | Must load from `antennas.csv` | **Pending fix** |
| Various docstrings | "117 antennas" | "96 active antennas" | **Pending** |
| Sensitivity estimate in code | None documented | SEFD = 6500 Jy/antenna, Tsys ~ 28 K | **Documented here** |

---

## 13. Quick Reference Card

```
DSA-110 at a glance
===================
Active antennas:   96 total (47 E-W + 35 N-S + 14 outriggers, verified 2026-05-05)
Dense core:        82 (47 E-W + 35 N-S, b < 485 m)
Max baseline:      2.6 km (outriggers)
Dish diameter:     4.65 m
Frequency:         1311.25–1498.75 MHz (187.5 MHz bandwidth)
Subbands:          16 × 11.72 MHz
Channels/subband:  48 (244 kHz each)
Integration time:  12.885 s (pipeline dump)
Fringe interval:   ~3.35 s
Observing mode:    Meridian drift-scan (transit)
Site:              OVRO, 37.2339°N 118.2825°W, 1222 m
SEFD (per ant):    6500 Jy
Implied Tsys:      ~28 K (η ≈ 0.70)
Polarizations:     XX, YY (linear)
Synth beam (core): ~31 arcsec (R = −2) | ~10–15 arcsec (R = 0.5)
Primary beam FWHM: ~4.1° (Airy disk, 4.65 m dish at 1.4 GHz)
Search FoV:        0.5°/cos(δ) × 0.5° (NSFRB)
Image noise:       ~40 mJy (NSFRB, 11.68 MHz, 3.35 s)
                   ~5 mJy (continuum, 187.5 MHz, 12.885 s, est.)
```
