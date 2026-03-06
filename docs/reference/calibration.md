# Reference: Calibration

Source: `/data/dsa110-contimg/backend/src/dsa110_contimg/core/calibration/`
Files: `presets.py`, `calibration.py`, `selfcal.py`, `refant_selection.py`,
       `bandpass_diagnostics.py`, `flux_validation.py`

---

## DEFAULT_PRESET — all validated parameters

Validated 2026-02-02 on 3C454.3 data. From `presets.py` lines 167–186.

```python
field            = "0~23"       # All 24 fields
refant           = "103"        # Default; use "auto" for MS-health-based selection
solve_delay      = False        # K-cal NOT needed for DSA-110 (see below)
solve_bandpass   = True
solve_gains      = True
gain_calmode     = "ap"         # Amplitude + phase
gain_solint      = "inf"        # One solution per scan
prebp_phase      = True         # Pre-BP phase solve (improves BP quality)
prebp_minsnr     = 3.0
bp_minsnr        = 5.0          # NOTE: solve_bandpass() default is 3.0; must pass 5.0 explicitly
gain_minsnr      = 3.0
bp_combine_field = True
bp_combine_spw   = True         # NOTE: see warning below about combine_spw for bandpass
do_flagging      = True
flag_autocorr    = True
use_adaptive_flagging = True
```

---

## Calibration sequence

```
1. setjy() / populate_model_from_catalog()  — Perley-Butler 2017 flux model
2. flag_rfi()                               — flagging BEFORE calibration solves
3. detect_and_flag_dead_antennas()          — before calibration (0.95 threshold)
4. solve_prebandpass_phase()                — phase-only, NOT applied to target data
5. solve_bandpass()                         — bp_minsnr=5.0, fillgaps=3, uvrange=">1klambda"
6. solve_gains()                            — two timescales: inf + 60s
7. applycal()                               — apply B + G + 2G to CORRECTED_DATA
```

---

## K-calibration (delay)

DSA-110 does NOT need delay calibration in routine operation.
- Empirical validation 2026-02-02: phase slopes **< 3°** across band.
- Post-calibration phase std: **~27°** (typical).
- Geometric delay bounds: DSA-110 max baseline ~2707 m → max geometric delay ~9 µs.
- Outlier threshold in QA: ±500 ns (flags as hardware errors).
- Fraction threshold: < 10% of antennas outside geometric bounds = pass (non-strict mode).

The K-table `solve_delay()` is implemented and produces diagnostics, but the parameter
`"K-calibration not required for DSA-110"` is explicitly stated at `calibration.py`
line 2628 — it is NOT applied in the gain chain.

---

## Bandpass solve — critical parameter warnings

### bp_minsnr must be 5.0, not the function default 3.0

`DEFAULT_PRESET` sets `bp_minsnr=5.0`.
`solve_bandpass()` function signature default is `minsnr=3.0`.
Always pass `minsnr=5.0` explicitly when calling `solve_bandpass()`.

### combine_spw=False for per-SPW solutions

`calibration.py` lines 2097–2101: combining SPWs does NOT increase per-channel SNR
(unlike gain calibration) and actually produces more flagging because each SPW has a
different bandpass shape. Use `combine_spw=False`.

### Other validated bandpass parameters

```python
solve_bandpass(
    ms, cal_field, refant, ktable=None,
    minsnr=5.0,                   # NOT the function default of 3.0
    combine_spw=False,            # Per-SPW solutions
    uvrange=">1klambda",          # Exclude short baselines
    fillgaps=3,                   # Interpolate flagged channels up to width 3
    minblperant=4,
    model_standard="Perley-Butler 2017",
    max_flag_fraction=0.05,       # Triggers diagnostic analysis above 5%
)
```

Target bandpass flag fraction: **< 3%**. Above 5% → automatic diagnostic analysis.
QA flag fraction grading: < 3% PRISTINE, 3–5% GOOD, 5–10% MODERATE, 10–20% HIGH,
> 20% CRITICAL.

---

## Gain solve — two timescales

```python
solve_gains(
    ms, cal_field, refant, ktable=None, bptables=[bp_table],
    solint="inf",       # Long timescale (.g): instrumental effects
    t_short="60s",      # Short timescale (.2g): atmospheric phase variations
    calmode="ap",       # Amplitude + phase
    minsnr=3.0,
)
```

Application order: `B → G → 2G` (calibration.py line 2665).
K-table is accepted as parameter but explicitly not used.
Gain solve QA: < 5% flagging ideal, < 10% acceptable.

---

## Pre-bandpass phase solve

`solve_prebandpass_phase()`:
- Phase-only (`calmode="p"`), `solint="inf"`, `uvrange=">1klambda"`.
- Applied ONLY during bandpass solve as a gaintable (NOT to target data).
- Purpose: removes phase decorrelation from integrating 24 timesteps, improves BP SNR.

---

## Reference antenna

Default: `refant="103"` (outrigger antenna).

Outrigger priority order for fallback chain (`refant_selection.py`):
```
Eastern first (best baseline coverage):
104, 105, 106, 107, 108
Northern (good azimuth):
109, 110, 111, 112, 113
Western/peripheral:
114, 115, 116, 103, 117
```

Core antennas (1–102) are within 500 m of centre. Outriggers (103–117) reach ~2.6 km.
Outriggers provide long-baseline leverage for calibration quality. For production,
use `select_best_outrigger_refant(ms_path)` instead of hardcoding.

---

## Flux scale validation

`flux_validation.py` `check_model_corrected_ratio()`:
- Samples 50% of central channels, up to 500 rows from mid-MS.
- Computes `mean(|MODEL|) / mean(|CORRECTED|)`.

| Threshold | Value | Status |
|---|---|---|
| `ratio_warn` | 1.5× | warn |
| `ratio_fail` | 5.0× | fail |

This is a lightweight sanity check only. Absolute flux scale is set via
`model_standard="Perley-Butler 2017"` in `solve_bandpass()` via CASA `setjy`.
No hardcoded DSA/NVSS ratio exists in this file.

---

## Self-calibration — SelfCalConfig validated defaults

From `selfcal.py` lines 104–176. **Not yet in reference pipeline script (`run_pipeline.py`).**

| Parameter | Value |
|---|---|
| `max_iterations` | 5 |
| `phase_solints` | `["300s", "120s", "60s"]` (progressive: 5 min → 2 min → 1 min) |
| `phase_minsnr` | 3.0 |
| `amp_minsnr` | 3.0 |
| `amp_antenna_snr` | **10.0** (stricter than phase at 3.0) |
| `amp_solint` | `"inf"` |
| `amp_combine` | `"scan"` |
| `max_phase_scatter_deg` | 30.0° |
| `max_amp_scatter_frac` | 0.3 |
| `min_beam_response` | 0.5 (50% primary beam) |
| `min_snr_improvement` | 1.05 (5% required per iteration) |
| `min_chi_squared_improvement` | 0.95 |
| `backend` | `"wsclean"` |
| `niter` | 10000 |
| `threshold` | `"0.1mJy"` |
| `robust` | 0.0 |
| `use_galvin_clip` | True |
| `galvin_box_size` | 100 |
| `galvin_adaptive_depth` | 3 |

Beam check: amplitude self-cal verifies primary beam response ≥ 0.5 before accepting
solutions. DSA-110 FWHM ~2.5° at 1.4 GHz (selfcal.py line 647).

Model prediction: WSClean `-predict` (CASA `ft()` is available but unused).

Chi-squared: `Σ(w|DATA−MODEL|²)/Σ(w)` using `WEIGHT_SPECTRUM` (falls back to `WEIGHT`).

---

## General QA thresholds (calibration.py module level)

```python
QA_SNR_MIN_THRESHOLD     = 3.0    # Minimum acceptable mean SNR
QA_SNR_WARN_THRESHOLD    = 10.0   # Below this → warning
QA_FLAGGED_MAX_THRESHOLD = 0.5    # Maximum acceptable flagged fraction
QA_FLAGGED_WARN_THRESHOLD = 0.2   # Above this → warning
QA_MIN_ANTENNAS          = 10     # Minimum antennas for valid solution
```
