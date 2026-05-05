# DSA-110 Ground Truth

This file is an evidence index for DSA-110 parameters used by this repository.
It is not allowed to become an unsourced authority document: each claim below
has a status, evidence pointer, and verification date.

Status vocabulary:

- `repo_verified`: directly supported by current checked-in code or config.
- `computed`: recomputed from current checked-in code or data files.
- `test_asserted`: asserted by current checked-in tests.
- `external_verified`: confirmed against an external DSA-specific source (real H17 HDF5/MS metadata or sibling `dsa110-antpos` checkout) on the date below.
- `external_pending`: plausible or cited externally, but not verified in this pass.
- `stale_rejected`: present in older docs but contradicted by current evidence.

External validation sources for DSA-110 instrument facts are limited to:

- Connor et al. 2025.
- Real HDF5/MS antenna metadata from H17 `/data/incoming/`, read via `pyuvdata`.
- A sibling DSA-specific `dsa110-antpos` checkout if present.

Do not use ASKAP/VAST references as authority for DSA-110 antenna counts,
station positions, or active-array composition.

## Current Claims

| Claim | Status | Evidence | Last verified |
|---|---|---|---|
| The repository currently contains 117 allocated DSA-110 station rows in `antennas.csv`. | repo_verified | `dsa110_continuum/simulation/pyuvsim/antennas.csv`; `wc -l` gives 118 lines including header. | 2026-05-05 |
| The repository currently contains 117 DSA station rows in `DSA110_Station_Coordinates.csv`; DSA-117 has a blank elevation field. | repo_verified | `dsa110_continuum/simulation/pyuvsim/DSA110_Station_Coordinates.csv`; `wc -l` gives 123 lines including metadata/header rows; tail shows DSA-117 blank elevation. | 2026-05-05 |
| `load_geodetic_enu()` reads `DSA110_Station_Coordinates.csv`, converts WGS84 coordinates to ECEF, and rotates into local ENU. | repo_verified | `dsa110_continuum/simulation/harness.py` function `load_geodetic_enu`. | 2026-05-05 |
| The default `SimulationHarness` real-position path uses `load_geodetic_enu()`. | repo_verified | `dsa110_continuum/simulation/harness.py` property `antenna_enu`. | 2026-05-05 |
| `antennas.csv` is a bundled projected-coordinate station file, but it is not the current default path for `SimulationHarness` real positions. | repo_verified | `dsa110_continuum/simulation/harness.py` has `_load_antenna_enu_from_csv()` for `antennas.csv`; `antenna_enu` calls `load_geodetic_enu()` by default. | 2026-05-05 |
| `SimulationHarness` defaults to `n_antennas=117`, `n_integrations=24`, `n_sky_sources=20`, `seed=42`, and `pointing_dec_deg=16.15`. | repo_verified | `dsa110_continuum/simulation/harness.py` dataclass defaults. | 2026-05-05 |
| Full-geometry simulation should use `n_antennas=117`; `n_antennas=96` selects the first 96 station rows and excludes outrigger station rows, so it is not a proxy for the 96 active operational array. | repo_verified | `SimulationHarness.antenna_enu` slices the first `n_antennas` rows through `load_geodetic_enu(n_antennas=...)`; station rows DSA103-DSA117 are outside the first 96 rows. | 2026-05-05 |
| The harness OVRO constants are latitude `37.2339 deg`, longitude `-118.2825 deg`, altitude `1222.0 m`. | repo_verified | `dsa110_continuum/simulation/harness.py` constants `_OVRO_LAT_DEG`, `_OVRO_LON_DEG`, `_OVRO_ALT_M`. | 2026-05-05 |
| The simulation spectral setup is 16 subbands, 48 channels per subband, 768 channels total, and channel width `244140.625 Hz`. | repo_verified | `dsa110_continuum/simulation/harness.py` constants; `dsa110_continuum/simulation/config/dsa110_measured_parameters.yaml` `spectral`. | 2026-05-05 |
| The YAML channel-width value is currently `244140.625 Hz`; the old `325.520833 kHz` warning is obsolete. | repo_verified | `dsa110_continuum/simulation/config/dsa110_measured_parameters.yaml` `spectral.channel_width_hz` and `system_parameters.channel_width` history note. | 2026-05-05 |
| Current harness subband channel centers span `1311.3720703125 MHz` through `1498.6279296875 MHz`; 768 channel widths span `187.5 MHz`. | computed | Recomputed with `SimulationHarness.subband_freqs(0..15)` under `/opt/miniforge/envs/casa6/bin/python`. | 2026-05-05 |
| Current harness tile timing is 24 integrations times `12.884902 s`, giving `309.237648 s`. | computed | Recomputed from `24 * 12.884902`; integration constant is in `dsa110_continuum/simulation/harness.py`. | 2026-05-05 |
| The current default simulation tile start is `2026-01-25T22:26:05`; the median RA for that tile is about `344.124049 deg`. | computed | `SimulationHarness.make_time_array()` default start plus recomputed apparent sidereal time at `_OVRO_LON_DEG`. | 2026-05-05 |
| The current default simulation declination `16.15 deg` is a canary/simulation default, not an operational fixed survey declination. | repo_verified | `SimulationHarness.pointing_dec_deg` default; `docs/pipeline-specs.md` states survey declination is read from incoming observation metadata. | 2026-05-05 |
| Current `load_geodetic_enu(117)` produces E span about `1768.56 m` and N span about `2219.73 m`. | computed | Recomputed with `load_geodetic_enu(117)` under `/opt/miniforge/envs/casa6/bin/python`. | 2026-05-05 |
| The synthetic simulation sky uses random point sources with a power-law flux distribution and spectral-index scatter. | repo_verified | `dsa110_continuum/simulation/harness.py` function `_make_sky_model()`. | 2026-05-05 |
| The default synthetic sky uses `n_sky_sources=20` and `seed=42`. | repo_verified | `SimulationHarness` dataclass defaults. | 2026-05-05 |
| `scripts/plot_tile_image.py` uses `AMP_SCATTER = 0.05` and `PHASE_SCATTER = 5.0`. | repo_verified | `scripts/plot_tile_image.py` constants. | 2026-05-05 |
| The QA noise model default uses `num_antennas=96`. | repo_verified | `dsa110_continuum/qa/noise_model.py` `calculate_theoretical_rms()` signature. | 2026-05-05 |
| Noise-model behavior around 96 versus 117 antennas is asserted by tests. | test_asserted | `tests/test_noise_model_fix.py`. | 2026-05-05 |
| The operational active-antenna total is 96 (`Nants_data=96`, `Nants_telescope=117`). | external_verified | Read from real H17 HDF5 `/data/incoming/2026-01-25T00:00:10_sb00.hdf5` `Header/Nants_data`; cross-checked by counting unique IDs in `ant_1_array ∪ ant_2_array`. | 2026-05-05 |
| The active per-arm breakdown is **47 E-W + 35 N-S + 14 outriggers**. | external_verified | Active station-number list extracted from real H17 HDF5 metadata; intersected with slot pools DSA001-051 (E-W), DSA052-102 (N-S), DSA103-117 (outriggers). See `outputs/ground_truth_audit_2026-05-04/active_antennas_2026-01-25.md`. | 2026-05-05 |
| The exact active station-number list for the 96 active antennas is enumerated. | external_verified | Full list in `outputs/ground_truth_audit_2026-05-04/active_antennas_2026-01-25.md`. Inactive stations are DSA-{10, 21, 22, 23, 52-67, 117}. Source: real H17 HDF5 `/data/incoming/2026-01-25T00:00:10_sb00.hdf5`. | 2026-05-05 |
| Commissioning configuration counts: `ant_ids.csv` = 24 (20 E-W + 3 N-S + 1 outrigger), `ant_ids_case1.csv` = 24 (same composition), `ant_ids_case2.csv` = 25 (20 E-W + 3 N-S + 2 outriggers), `ant_ids_mid.csv` = 66 (48 E-W + 3 N-S + 15 outriggers). | external_verified | Read from sibling `/data/dsa110-antpos/antpos/data/ant_ids*.csv`. | 2026-05-05 |
| Reference figure metrics such as PSF HPBW, peak sidelobe, and UV fill are not ground truth until regenerated or checked from their source scripts. | external_pending | `docs/images/*` exists, but this pass did not regenerate or validate figure numerical content. | 2026-05-05 |

## Rejected Old Claims

| Claim | Status | Evidence | Last verified |
|---|---|---|---|
| `OLD_GROUND_TRUTH.md` is a single authoritative reference that every agent should trust. | stale_rejected | The file was created in commit `6ad8a00` alongside many simulation changes and contains claims contradicted by current source. | 2026-05-05 |
| `DSA110_Station_Coordinates.csv` is human reference only and `antennas.csv` is used by `load_geodetic_enu()`. | stale_rejected | Current `load_geodetic_enu()` reads `DSA110_Station_Coordinates.csv`. | 2026-05-05 |
| `SimulationHarness` default `n_antennas` is 8. | stale_rejected | Current dataclass default is `117`. | 2026-05-05 |
| DSA-110 operational declination is fixed at `16.15 deg`. | stale_rejected | `16.15 deg` is a simulation default; operational docs say declination is read from observation metadata. | 2026-05-05 |
| The YAML still contains a stale `325.520833 kHz` channel-width value to ignore. | stale_rejected | Current YAML records `244.140625 kHz` and a correction note. | 2026-05-05 |
| Inactive station slots produce zero-weighted baselines in the simulation harness. | stale_rejected | Current harness creates baselines across the selected antenna rows; no zero-weighting of inactive slots was found. | 2026-05-05 |
| The active per-arm breakdown is `51 EW + 35 NS + 14 outriggers`. | stale_rejected | Real H17 HDF5 metadata gives **47 E-W + 35 N-S + 14 outriggers**. The 51 number reflects E-W *slot* count, not active-antenna count. | 2026-05-05 |

## Working Audit

The claim matrix that led to this rewrite is preserved at:

- `outputs/ground_truth_audit_2026-05-04/README.md`
- `outputs/ground_truth_audit_2026-05-04/ground_truth_claims_audit.csv`
- `outputs/ground_truth_audit_2026-05-04/active_antennas_2026-01-25.md` (real H17 HDF5 active-antenna resolution, 2026-05-05)

These audit artifacts are part of this docs change set and should be committed
with `docs/GROUND_TRUTH.md`; otherwise this pointer will intentionally be
removed.
