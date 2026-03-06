# Reference: MS Conversion and QA

Source: `/data/dsa110-contimg/backend/src/dsa110_contimg/core/conversion/`
Files: `conversion_orchestrator.py`, `normalize.py`, `merge_spws.py`,
       `helpers_validation.py`, `helpers_telescope.py`, `direct_subband.py`
       `core/qa/`: `image_metrics.py`, `delay_validation.py`, `uvw_validation.py`,
       `calibration_quality.py`, `pipeline_quality.py`

---

## Subband grouping

16 subbands per observation: `{YYYY-MM-DDTHH:MM:SS}_sb{00..15}.hdf5`.
Subbands arrive with slightly different timestamps (±seconds). Before grouping:

`normalize_directory()` (`normalize.py` line 48): clusters files within
`DEFAULT_CLUSTER_TOLERANCE_S = 120.0 s` and atomically renames each to the
canonical (first-arrival) timestamp. After normalisation, all 16 subbands share
an identical filename prefix and `GROUP BY group_id` in SQLite is exact.

Default `expected_subbands = 16` (from `settings.conversion.expected_subbands`).

---

## PyUVData workarounds (conversion_orchestrator.py lines 586–601)

These are required and must not be "fixed":

| Issue | Fix |
|---|---|
| `uvw_array` is `float32` in DSA-110 HDF5 | Cast to `float64` before `__iadd__()` |
| UVData does not pass pyuvdata default dtype validation | Use `run_check=False`, `strict_uvw_antpos_check=False` |
| `blt_order` warning when combining subbands | Set `blt_order=("time", "baseline")` explicitly |

---

## Frequency order: subbands arrive descending, MS must be ascending

`validate_ms_frequency_order()` (`helpers_validation.py` lines 21–72):

DSA-110 subbands arrive in descending order (sb00 = highest freq, sb15 = lowest).
CASA imaging requires ascending frequency order. Failing to sort produces:
- MFS fringe artefacts across the image
- Bandpass calibration failures (CASA applies solutions in frequency order)

This is checked by the post-conversion validation gate.

---

## TELESCOPE_NAME: dual-identity scheme

The MS must carry different `OBSERVATION::TELESCOPE_NAME` values depending on
which tool is consuming it:

| Consumer | Required name | Where set |
|---|---|---|
| CASA (`gaincal`, `mstransform`, `listobs`) | `OVRO_MMA` | `merge_spws.py` lines 147–162 |
| WSClean / EveryBeam | `DSA_110` | `helpers_telescope.py` `set_ms_telescope_name()` line 227 |
| pyuvdata in-memory | `DSA_110` | `helpers_telescope.py` `set_telescope_identity()` line 84 |

`OVRO_MMA` exists in CASA's observatory table and suppresses `listobs` errors.
`DSA_110` is required for EveryBeam to load the native Airy disk beam model
(added in EveryBeam 0.7.2).

`set_ms_telescope_name(ms_path, "DSA_110")` is called in `cli_imaging.py` lines
193–200 immediately before every WSClean invocation.

---

## SIGMA_SPECTRUM column removal after mstransform

`merge_spws.py` lines 132–143: after `mstransform`, removes the `SIGMA_SPECTRUM`
column. Non-fatal but saves significant disk space; the column is created
redundantly and automatically by mstransform.

---

## Memory and GPU safety decorators

Introduced after a Dec 2 2025 OOM incident that caused a disk disconnection.
All compute-heavy functions use:

```python
@memory_safe(max_system_gb=6.0)
@gpu_safe(max_gpu_gb=9.0)
```

These are safety-critical for production. Omitting them risks OOM-induced
filesystem damage.

---

## Post-conversion validation steps (helpers_validation.py)

All four must be called after every conversion:

| Function | What it checks |
|---|---|
| `validate_ms_frequency_order()` | Each SPW has ascending frequency |
| `validate_phase_center_coherence(tolerance_arcsec=1.0)` | All FIELD entries agree to within 1 arcsec (or time-dependent meridian is detected) |
| `validate_uvw_precision()` | Baseline lengths 5–3210 m; not all-zero; not bimodal |
| `validate_antenna_positions(tolerance_m=0.05)` | ITRF positions vs RevF reference, tolerance 5 cm |

---

## UVW / phaseshift QA (uvw_validation.py)

`check_uvw_after_phaseshift()` raises `ValueError` on failure.

| Check | Threshold |
|---|---|
| Max UVW vs max baseline | `max_baseline_m × 1.1 = 2707 × 1.1 = 2978 m` |
| Fraction of failing baselines | > 1% → fail |
| Coefficient of variation | > 0.8 → warn (mixed conventions) |

**chgcentre 2× convention mismatch detector** (uvw_validation.py lines 236–242):
If `1.8 < max_uvw / max_baseline < 2.2`, flags the specific chgcentre 2× UVW
convention bug. Fix: `use_chgcentre=False` (use CASA phaseshift instead).

---

## Delay QA (delay_validation.py)

DSA-110-specific constants:
- Max baseline: **~2707 m**
- Max geometric delay: **~9 µs** (= 2707/c)
- Safety factor: **1.5×** → QA limit ~13.5 µs
- Outlier threshold: **±500 ns** (hardware error)
- Pass criterion (non-strict): < 10% of antennas outside geometric bounds

Reference antenna inferred as antenna with smallest absolute delay (< 1 ns threshold).

---

## Image QA metrics (image_metrics.py)

| Metric | Implementation |
|---|---|
| Dynamic range | Peak / RMS using four 10%-of-min-dimension corner regions |
| RMS | `1.4826 × MAD` (robust) from FITS array |
| Residual stats | mean, std, max, min, RMS; Shapiro-Wilk normality p-value (sampled to 5000 px) |
| PSF correlation | Pearson r between central 64×64 dirty image and PSF |

Dagster validation grades (from `ValidationRunConfig`):
- Astrometric tolerance: 5 arcsec
- Flux scale tolerance: 20%
- Minimum cross-matched sources for grade A: 5

---

## Dagster pipeline structure

Asset graph partitioned by calendar day (`DailyPartitionsDefinition`, start 2024-01-01 UTC):

```
measurement_sets → calibration_tables → calibrated_ms → images → validated_images
                                                                       ↓
                                                                    mosaics
                                                                       ↓
                                                               photometry → light_curves
```

Retry policies:
| Policy | Max retries | Delay | Use |
|---|---|---|---|
| IO | 3 | 1 s exp + 20% jitter | Conversion, file I/O |
| Compute | 2 | 5 s exp | Calibration, imaging |
| Quick | 5 | 0.5 s exp | DB queries, health checks |
| None | 0 | — | Validation, QA |

Alerting: Slack webhook on failure/success (`CONTIMG_SLACK_WEBHOOK_URL` env var).
WSClean threads: `DSA110_NUM_THREADS` (default 8). IDG mode: `DSA110_IDG_MODE` (default `gpu`).
