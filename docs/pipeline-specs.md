# DSA-110 Continuum Imaging Pipeline — Specifications

Compiled from direct source analysis of:
- `dsa110-continuum` (active pipeline): `/data/dsa110-continuum/`
- `dsa110-contimg` (reference pipeline): `/data/dsa110-contimg/`

Last updated: 2026-03-03

---

## How to read this document

Parameters are grouped into two categories:

**HARDWARE** — Fixed by the physical instrument. These cannot be changed by the
pipeline. If a pipeline value disagrees with the hardware specification, the
hardware specification is authoritative.

**PIPELINE CHOICE** — Values we have decided on. These can be changed. Where
the reasoning behind a choice is non-obvious, it is stated explicitly.

Items marked **OPEN** require further investigation or a decision has not yet
been made.

---

## 1. Instrument (HARDWARE)

| Parameter | Value | Source |
|---|---|---|
| Total allocated antenna slots (CSV) | 117 | dsa110-antpos position CSV |
| **Active antennas** | **96** | Connor et al. 2025 (arxiv:2510.18136) |
| Dense core active breakdown (b < 485 m) | external_pending | Repo docs have disagreed between 47 and 51 active E-W antennas; verify against DSA-specific evidence before using a breakdown. |
| Outrigger active breakdown (to ~2.6 km, IDs 103–117) | external_pending | Often cited as 14 active of 15 allocated, but keep external_pending until reconciled with the active-ID source. |}
| **Dish diameter** | **4.65 m** ± 0.01 m | Measured (dsa110_measured_parameters.yaml) |
| Observing band | L-band (1.28–1.53 GHz nominal) | Instrument design |
| **Science frequency range** | **1311.25–1498.75 MHz** | Observed band (confirmed) |
| **Total science bandwidth** | **187.5 MHz** | 16 × 11.71875 MHz subbands |
| **Subbands (SPWs)** | **16** | Correlator |
| **Channels per subband** | **48** | Correlator |
| **Total channels** | **768** (16 × 48) | Correlator |
| **Channel width** | **244.14 kHz** | 11.71875 MHz / 48 channels |
| **Integration time** | **12.885 s** | Correlator dump rate |
| Polarizations | XX, YY | Correlator |
| Observing mode | **Meridian drift-scan (transit instrument)** | Instrument design |
| Site | OVRO, 37.2339°N 118.2825°W, 1222 m | Instrument design |

Notes:
- The correlator nominal window is 250 MHz, but the science-grade band is
  187.5 MHz (1311.25–1498.75 MHz). The 250 MHz figure should not appear in
  pipeline code.
- The dish diameter of 4.65 m is the measured value. All pipeline code
  (primary beam models, seeding radius, pb_cutoff) must use 4.65 m. The
  previous incorrect value of 4.7 m has been corrected throughout.
- Survey declination is NOT fixed. The telescope can slew in elevation.
  Declination is read from each incoming observation's HDF5 metadata.
  No pipeline code should assume a fixed survey declination.

---

## 2. Observing Strategy (HARDWARE + PIPELINE CHOICE)

| Parameter | Value | Category | Notes |
|---|---|---|---|
| Fields (timesteps) per tile | 24 | HARDWARE | 24 × 12.885 s = ~5.15 min |
| Duration per tile | ~5.15 min | HARDWARE | One meridian transit |
| **Tiles per mosaic** | **12** | PIPELINE CHOICE | 12 × 5 min = ~60 min of RA coverage |
| Mosaic stride | 6 tiles | PIPELINE CHOICE | 50% overlap between adjacent mosaics |
| Dec tolerance between consecutive tiles | 1.0° | PIPELINE CHOICE | |
| Survey declination | Determined from data | HARDWARE | Read from HDF5 metadata per observation |

---

## 3. Data Ingest and Conversion (PIPELINE CHOICE)

| Parameter | Value | Notes |
|---|---|---|
| Expected subbands per group | 16 | |
| Timestamp cluster tolerance | 120 s | Groups subbands with slightly different write timestamps |
| Subband naming | `{timestamp}_sb{00..15}.hdf5` | |
| UVW reconstruction | chgcentre (WSClean), CASA phaseshift as fallback | |
| pyuvdata uvw_array dtype | Explicit float32 → float64 cast required | pyuvdata bug with HDF5 input |
| run_check on UVData read | False + strict_uvw_antpos_check=False | DSA-110 files fail default validation |

---

## 4. Calibration (PIPELINE CHOICE)

### Strategy overview

Each transit tile has its own calibration: one bandpass table (B) and one gain
table (G), solved from the calibrator transit and applied to the science tile.
Tables are not shared or interpolated across days.

### Calibrator

The calibrator must be specified at runtime. 3C454.3 has been used in
commissioning observations. 3C454.3 is a flat-spectrum variable blazar, not a
Perley-Butler primary flux standard. The flux model used is a static catalog
lookup — not a per-channel Perley-Butler polynomial. True primary calibrators
(3C286, 3C48, 3C147) are supported by the `fluxscale.py` module but are not
used in the current reference pipeline.

| Parameter | Value | Notes |
|---|---|---|
| Calibrator | Runtime input (e.g. 3C454.3) | Must be chosen per observation |
| Flux model | Static SQLite catalog lookup | No CASA setjy / Perley-Butler polynomial |
| Solve delay (K) | **False** | Empirically validated: phase slopes < 3° |
| Pre-BP phase solve | **True** | Improves BP SNR; applied only during BP solve |
| Pre-BP solint | "60s" | Phase only |
| **Pre-BP uvrange** | **">1klambda"** | Short baselines excluded |
| Pre-BP minsnr | 3.0 | |
| Pre-BP combine fields+SPW | True | |
| **bp_minsnr** | **5.0** | NOTE: function default is 3.0 — must pass explicitly |
| **bp_combine_spw** | **False** | Per-SPW solutions; each of 16 subbands has distinct bandpass shape |
| bp_combine_field | True | |
| fillgaps | **3** | Interpolate flagged channels up to 3 wide (~730 kHz at 244 kHz/ch) |
| minblperant | **4** | Minimum baselines per antenna required for a valid solution |
| gain_solint | "inf" | One complex gain per antenna per ~5-min scan |
| gain_calmode | "ap" | Amplitude + phase |
| gain_minsnr | 3.0 | |
| Reference antenna | "103" (default); "auto" for health-based selection | |
| Refant priority order | 104–108 (E), 109–113 (N), 114–116, 103, 117 | Eastern outriggers first |
| Flagging before calibration | AOFlagger + 7σ MAD clip + extend | |
| Dead antenna detection before BP | threshold = 0.95 (>95% flagged = dead) | |

### OPEN: bp_combine_spw

**Physical argument for False (recommended):**
Each of the 16 subbands has its own bandpass shape — distinct filter response,
cable length, and RFI environment. With `combine_spw=True`, CASA fits a single
bandpass solution applied uniformly across all 16 subbands, which forces an
incorrect shared shape. With `combine_spw=False`, each subband gets its own
per-channel solution, which is physically correct. Combining SPWs does not
increase per-channel SNR in bandpass calibration (unlike gain calibration where
it does). The old pipeline reference documentation explicitly recommends False.
The DEFAULT_PRESET currently sets True — this is likely a bug in the preset.

**Recommendation: set bp_combine_spw=False.** Pending user confirmation.

### OPEN: fillgaps and minblperant

**fillgaps=3 (recommended):**
Interpolates across flagged channel gaps up to 3 channels wide (~730 kHz).
DSA-110 has known narrowband RFI features at 1350, 1395, 1405, 1422, 1475 MHz
that leave flagged channels in the bandpass solution. Without fillgaps, these
appear as holes. Interpolating up to 3 consecutive channels is conservative
and physically appropriate. **Recommendation: add fillgaps=3 to new pipeline.**

**minblperant=4 (recommended):**
Minimum baselines per antenna for a valid solution. With 117 antennas,
essentially all antennas exceed this threshold except in severe flagging
scenarios. Setting 4 is the CASA default and a conservative guard against
solutions from antennas with too little data. **Recommendation: confirm and add
minblperant=4 to new pipeline.**

---

## 5. Imaging — Single Tile (PIPELINE CHOICE)

| Parameter | Value | Notes |
|---|---|---|
| Backend | WSClean | CASA tclean available as fallback |
| **Image size** | **4800 × 4800 px** | Updated from 2400 |
| **Cell size** | **3.0 arcsec/px** | Updated from 6.0 arcsec |
| **FoV (4800 × 3 arcsec)** | **4.0°** | Primary beam FWHM at 1.4 GHz with 4.65 m dish ≈ 4.1° |
| Synthesized beam | ~10–15 arcsec (estimated) | ~3–5 px per beam; not hardcoded; read from FITS |
| Gridder | wgridder (production script); idg (module default) | |
| IDG mode | "hybrid" | GPU validated: RTX 2080 Ti sm_75, cuda-nvcc-11-1 installed |
| Weighting | Briggs, robust = 0.5 | |
| Weighting (survey) | Briggs, robust = 0.0 | |
| niter (standard) | 1000 | |
| niter (high_precision) | ≥2000 | |
| niter (survey) | 10000 | |
| niter (development) | 300 | Non-science quality |
| Threshold (standard) | 5 mJy | |
| Threshold (high_precision) | 0.05 mJy | |
| auto-mask | 5σ | Hardcoded in WSClean call |
| auto-threshold | 1.0σ | Hardcoded in WSClean call |
| mgain | 0.8 | Hardcoded |
| Deconvolver | hogbom (standard); multiscale (survey) | |
| Multiscale scales | 0, 5, 15, 45 px | |
| nterms | 1 (standard); 2 (survey, ~18% fractional BW) | |
| specmode | mfs | |
| uvrange cutoff | >1 klambda (-minuv-l 1000) | Excludes very short baselines |
| pblimit | 0.2 | |
| Primary beam correction | -apply-primary-beam (EveryBeam, DSA_110 model) | |
| Polarization | Stokes I only | |
| WSClean memory (4800 px) | 64 GB default | env WSCLEAN_ABS_MEM to override |
| WSClean timeout | 1800 s | env WSCLEAN_DOCKER_TIMEOUT to override |
| Sky model seeding threshold | 2.0 mJy | Catalog: FIRST > RACS > NVSS |
| Catalog mask aperture | 60 arcsec radius | |

---

## 6. Mosaicking (PIPELINE CHOICE)

### Tier selection

| Tier | Time span | Method |
|---|---|---|
| QUICKLOOK | < 1 h | Image-domain linear mosaic |
| SCIENCE | 1–48 h | Visibility-domain joint WSClean deconvolution |
| DEEP | > 48 h | Visibility-domain joint WSClean deconvolution |

For the standard 12-tile (~1 h) science mosaic, the SCIENCE tier applies.

### SCIENCE/DEEP tier (WSClean visibility-domain)

| Parameter | Value | Notes |
|---|---|---|
| WSClean -size | 4096 × 4096 px | |
| WSClean -scale | 1 arcsec/px | |
| FoV (4096 × 1 arcsec) | 1.14° | |
| WSClean -niter | 50000 | |
| WSClean -mgain | 0.6 | |
| WSClean -auto-threshold | 3.0σ | |
| Gridder | IDG (-use-idg) | |
| IDG mode | "cpu" | GPU UNAVAILABLE (see Section 8) |
| -grid-with-beam | yes (EveryBeam, direction-dependent) | Different from single-tile -apply-primary-beam |
| -local-rms | yes | |
| -parallel-deconvolution | 2000 | |
| Scratch directory | /dev/shm/mosaic | RAM disk for speed |
| Tiles per mosaic | 12 | |
| Mosaic stride | 6 tiles | 50% overlap |

### QUICKLOOK tier

| Parameter | Value |
|---|---|
| Primary beam model | Airy disk: (2·J1(x)/x)², D = 4.65 m |
| PB floor (pb_cutoff) | 0.1 |
| Max grid | 8192 × 8192 px |
| Weighting | inverse-variance × PB² |
| Frequency for PB weight | 1.4 GHz (default) |

---

## 7. Photometry and Variability (PIPELINE CHOICE)

### Forced photometry

| Parameter | Value | Notes |
|---|---|---|
| Method | Condon 1997 matched-filter (2D Gaussian kernel) | PSF-matched to restoring beam |
| box_size_pix | 5 | |
| Annulus | 30–50 px | For background estimation |

### Differential normalization

| Parameter | Value | Notes |
|---|---|---|
| n_baseline_epochs | 10 | First 10 mosaic epochs (~10 days at daily cadence) |
| fov_radius_deg | 1.5° | Reference source search radius |
| min_snr (NVSS) | 50.0 | Reference source quality gate |
| max_sources | 20 | |
| Sigma clipping | 3.0σ MAD | |
| Minimum valid references | 3 | Below this, no correction applied |
| Stability check window | 30 days | |
| Stability max chi²_r | 2.0 | Above this, source flagged as variable and excluded |

**Known limitation:** The baseline is anchored to the first 10 DSA-110 mosaic
epochs, not to an external catalog. The NVSS catalog flux is stored but not
used in the correction computation. The absolute flux scale depends on how
well-calibrated those first 10 epochs were (3C454.3 calibrator, static flux).
NVSS-anchoring is available as a sanity check but is not used for science.

**OPEN: Robust relative flux methodology for daily cadence science.**
See Section 9 for the design requirements.

### Variability detection

| Parameter | Value | Notes |
|---|---|---|
| Metrics | η, Vs, m (Mooley et al. 2016, ApJ 818, 105) | |
| ESE detection threshold | 5.0σ (conservative preset) | Note: this threshold is not physically motivated; it is a pragmatic choice subject to revision. |
| ESE moderate preset | 3.5σ | |
| ESE sensitive preset | 2.5σ | |

---

## 8. GPU Acceleration Status

### vLLM inference (separate service)
- Running on 2× RTX 2080 Ti with tensor parallelism
- Port 8080, model Qwen/Qwen2.5-7B-Instruct
- NCCL_P2P_DISABLE=1 and TRITON_ATTN backend required on Turing arch

### WSClean IDG GPU mode
**STATUS: VALIDATED — `idg_mode="hybrid"` is operational**

Root cause of prior failure: IDG 0.8 JIT-compiles CUDA kernels at runtime using
`/usr/local/cuda-11.1/bin/nvcc`. The CUDA 11.1 toolkit was partially installed
(runtime libs only). Fixed by installing `cuda-nvcc-11-1` from the local CUDA 11.1
repo at `/var/cuda-repo-ubuntu1804-11-1-local/`.

Validation run output:
```
Device memory : 951 Mb / 11004 Mb (free / total)
Compiler flags: -use_fast_math -lineinfo -arch=sm_75
DONE (w=[0.000938419:3312.73] lambdas, maxuvw=14381.8 lambda)
```
GPU detected: RTX 2080 Ti, sm_75 (Turing), CUDA 11.1, nvcc at `/usr/local/cuda-11.1/bin/nvcc`.
The test MS produced "IDG does not support irregular data" — this is an IDG algorithmic
constraint on the test calibrator scan, not a GPU failure. Mosaic joint-imaging data
(regularly gridded multi-tile visibilities) satisfies IDG's regularity requirement.

Current default in mosaic code: `idg_mode="hybrid"` (CPU fallback for irregular data portions).

---

## 9. Open Design Items

### 9.1 bp_combine_spw — RESOLVED

`bp_combine_spw=False` implemented in `CalibrationPreset` dataclass default,
`DEFAULT_PRESET`, and explicitly passed in `runner.py`'s `solve_bandpass` call.

### 9.2 fillgaps and minblperant — RESOLVED

`fillgaps=3` and `minblperant=4` added to `CalibrationPreset` as `bp_fillgaps`
and `bp_minblperant`, and explicitly passed in `runner.py`'s `solve_bandpass` call.
(Both were already the function defaults in `solve_bandpass` but were not wired
through from the preset, and were not guaranteed against future refactors.)

### 9.3 Robust relative flux photometry for daily cadence

The current differential normalization scheme has two structural weaknesses
for day-to-day variability science:

1. The baseline is frozen to the first 10 epochs and never updates. Any
   systematic gain error in the commissioning phase propagates permanently.

2. The correction factor is a single scalar per mosaic. It cannot account for
   direction-dependent gain residuals (ionospheric gradients, beam shape
   changes) that affect sources differently depending on their position in
   the field.

Design requirements for a robust replacement:
- The absolute flux reference must be anchored to an external catalog
  (NVSS or FIRST) via Huber regression, not to an internal floating baseline.
  This is the approach used by the VAST pipeline (vast-post-processing/corrections.py).
- Per-epoch correction factor: `S_measured = gradient × S_catalog + offset`,
  solved via Huber regression across all reference sources per mosaic.
  This is robust to variable source contamination in the reference ensemble.
- The uncertainty on the correction factor (propagated from reference scatter)
  must be stored as a per-epoch systematic floor on all variability metrics.
- Upper limits for non-detections must be stored explicitly (3σ), not silently
  dropped. VAST silently drops non-detections — DSA-110 should not.

### 9.4 IDG GPU remediation

Install CUDA 11.1 toolkit or rebuild WSClean. Once remediated, set
`idg_mode="gpu"` (or `"hybrid"` which WSClean recommends) in
`WSCleanMosaicConfig` and re-run the mosaic imaging test to validate.
