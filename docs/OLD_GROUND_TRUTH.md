# DSA-110 Simulation Ground Truth Reference

**Purpose:** Single authoritative reference for all DSA-110 parameters used in
the simulation pipeline. Every agent session should consult this file before
making assumptions about instrument geometry, timing, or spectral setup.

**Last updated:** 2026-04-17  
**Sources:** `docs/dsa110-instrument.md`, `dsa110_measured_parameters.yaml`,
`dsa110_continuum/simulation/pyuvsim/antennas.csv`,
Connor et al. 2025 (arXiv:2510.18136)

---

## 1. Array Geometry

### 1.1 Reference Location (OVRO)

| Parameter | Value | Source |
|---|---|---|
| Latitude | 37.2339° N | `harness.py` `_OVRO_LAT_DEG` |
| Longitude | −118.2825° E | `harness.py` `_OVRO_LON_DEG` |
| Altitude | 1222.0 m | `harness.py` `_OVRO_ALT_M` |

### 1.2 Active Antenna Count

**96 active antennas** out of 117 allocated station slots (Connor et al. 2025):

| Arm | Active | Slot pool | Notes |
|---|---|---|---|
| E-W core | **51** | DSA001–DSA051 | All 51 slots built and active. Physical gaps (DSA022→023: 63 m; around DSA029/030/035: ~23 m each) are unbuilt pad *locations*, not offline antennas. |
| N-S arm | **35** | DSA052–DSA102 | 51 slots allocated; 45 built out; 35 active. Which 16 are inactive is **not enumerated in this repo**. |
| Outriggers | **14** | DSA103–DSA117 | 15 slots; 14 active. DSA-117 lacks an elevation entry in the canonical CSV and is the likely inactive outrigger, but **not confirmed in any repo file**. |
| **Total** | **96** | — | — |

> **Critical caveat:** The exact list of which 96 station numbers are active is
> **not machine-readable anywhere in this repository or in `dsa110-antpos`**.
> The authoritative source is the HDF5/MS antenna metadata from real data on `h17`,
> accessed via `pyuvdata`'s `telescope.antenna_numbers` field at conversion time.
>
> For simulation purposes the correct approach is to **use all 117 station
> positions** from `antennas.csv` (§1.3) — this gives the correct array
> geometry with all built positions. The 21 inactive slots produce zero-weighted
> baselines in the imager and do not degrade image quality.

### 1.3 Position Files

Two files encode station positions. Use `antennas.csv` for simulation:

| File | Format | Use |
|---|---|---|
| `dsa110_continuum/simulation/pyuvsim/DSA110_Station_Coordinates.csv` | WGS84 lat/lon/elevation, 117 rows, header at row 5 | Human reference only |
| `dsa110_continuum/simulation/pyuvsim/antennas.csv` | Projected ECEF columns `east_m`, `north_m`, `up_m`, 117 rows | **Used by `load_geodetic_enu()` and the simulation harness** |

**Relative ENU (subtract DSA-001 raw values to get metres from DSA-001):**

```
DSA-001 raw:  east_m = 411169.605,  north_m = −375417.782,  up_m = −24373.994
Δeast  = east_m  − 411169.605
Δnorth = north_m − (−375417.782)   ← note: north_m values are large negative numbers
Δup    = up_m    − (−24373.994)
```

**Key relative positions (metres from DSA-001, from `load_geodetic_enu(117)`):**

> These are the values the harness actually uses (geodetic → ECEF → local-ENU
> rotation). They are the authoritative simulation positions.

| Station | Δeast (m) | Δnorth (m) | Notes |
|---|---|---|---|
| DSA-001 | 0.0 | 0.0 | West end of E-W arm |
| DSA-022 | +120.8 | 0.0 | Last station before 63 m gap |
| DSA-023 | +184.1 | 0.0 | First station after 63 m gap |
| DSA-051 | +396.9 | 0.0 | East end of E-W arm |
| DSA-052 | +198.4 | +8.6 | South end of N-S arm (T-junction) |
| DSA-102 | +198.4 | +441.4 | North end of N-S arm |
| DSA-103 | +388.0 | −374.4 | Closest outrigger (S) |
| DSA-104 | +781.8 | +209.0 | Outrigger (E) |
| DSA-108 | +790.5 | +1845.4 | Far N outrigger |
| DSA-114 | −905.2 | +1840.1 | Far NW outrigger |
| DSA-116 | −795.0 | −217.3 | SW outrigger |
| DSA-117 | −978.1 | −204.3 | **Likely inactive** (missing elevation in CSV) |

**Array extents (all 117 stations, relative to DSA-001):**

| Dimension | Extent |
|---|---|
| E-W arm span | ~397 m (DSA-001 to DSA-051) |
| N-S arm span | ~432 m (DSA-052 to DSA-102) |
| Max E baseline | ~1769 m (total E-W including outriggers) |
| Max N baseline | ~2220 m (total N-S including outriggers) |
| Nearest outrigger | DSA-103, 374 m S of T-junction |

### 1.4 Correct Harness Configuration

**Always use `n_antennas=117`** (all built positions). Do not use `n_antennas=96`
(takes first 96 CSV rows = 51 EW + 45 NS + 0 outriggers — geometrically wrong,
misses all outriggers).

```python
from dsa110_continuum.simulation.harness import SimulationHarness
harness = SimulationHarness(n_antennas=117, n_integrations=24)
```

The `SimulationHarness` default of `n_antennas=8` is **for unit tests only**.
Scripts that simulate real DSA-110 data must pass `n_antennas=117`.

---

## 2. Timing

| Parameter | Value | Source |
|---|---|---|
| Integration time | **12.884902000427246 s** | `dsa110_measured_parameters.yaml` → `temporal.integration_time_sec` |
| Fields per tile | **24** | One drift-scan tile = 24 integrations × 12.885 s ≈ 309.2 s |
| Tile duration | **~309.2 s** | 24 × 12.884902 s |
| Tile 0 start (UTC) | **2026-01-25T22:26:05** | `scripts/plot_tile_image.py` `T_START` |
| Tile 0 median RA | **344.124°** | Harness drift-scan calculation for T_START |
| Declination | **16.15°** | Fixed (OVRO transit strip at Dec ≈ +16°) |

---

## 3. Spectral Setup

| Parameter | Value | Source |
|---|---|---|
| Subbands | **16** | `harness.py` |
| Channels per subband | **48** | `harness.py` |
| Total channels | **768** | 16 × 48 |
| Channel width | **244.140625 kHz** | 250 MHz / 1024 total correlator channels; `dsa110_measured_parameters.yaml` → `spectral.channel_width_hz` |
| Frequency range | **1311.372 – 1498.628 MHz** | `harness.subband_freqs(0)` min → `harness.subband_freqs(15)` max |
| Bandwidth | **~187.3 MHz** | 768 × 244.14 kHz |
| Polarizations | **XX, YY** (2 pols) | pyuvdata convention |

> **YAML inconsistency (known bug):** `dsa110_measured_parameters.yaml` contains
> a stale entry `frequency_setup.channel_width.value = 325.520833 kHz` based on
> the derivation "15.625 MHz / 48 channels". This is **wrong for the operational
> pipeline**: the per-subband bandwidth is 11.71875 MHz (not 15.625 MHz), giving
> 11718750 / 48 = **244.14 kHz**. The 325.5 kHz figure reflects a different
> correlator mode and should be ignored. The correct value is in
> `spectral.channel_width_hz: 244140.625`.

---

## 4. Sky Model (Simulation)

The simulation sky model is **entirely synthetic** — random point sources drawn
from a power-law flux distribution using `seed=42`. Sources are **not** from
NVSS or any real catalog.

| Parameter | Value |
|---|---|
| Number of sources | `n_sky_sources=20` (harness default) |
| Seed | `seed=42` (reproducible) |
| Flux distribution | Power law, calibrated to realistic L-band source counts |
| Real source catalog | Exists at `outputs/diagnostics/2026-03-09/tile_2026-01-25T22-26-05_source_list.txt` but is **not connected to the simulation** |

For PSF and calibration testing the synthetic sky is sufficient. For end-to-end
validation against real data, the sky model must be replaced with a real catalog
(e.g. NVSS/FIRST/RACS) matched to the tile field of view.

---

## 5. Commissioning Configuration History

Do not confuse these with the operational 96-antenna array:

| Config | File | Count | Composition |
|---|---|---|---|
| Early commissioning | `antpos/data/ant_ids.csv` | 24 | DSA013–020, 024–035, 100–102, 116 |
| Mid-campaign | `antpos/data/ant_ids_mid.csv` | 66 | 48 EW + 3 NS (100–102) + 15 outriggers |
| Operational (current) | **not in any CSV** | **96** | 51 EW + 35 NS + 14 outriggers |

---

## 6. Known Simulation Limitations

1. **Active antenna list:** No machine-readable 96-antenna ID list exists in this
   repo. Using all 117 positions is the correct approximation for simulation.

2. **Sky model:** Synthetic only. Real-data validation requires a catalog sky model.

3. **Gain corruption:** The simulation applies per-antenna gain errors with
   amplitude scatter ~10% and phase scatter ~5°. These are representative but
   not calibrated against real DSA-110 gain stability measurements.

4. **No ionosphere:** The simulation does not model ionospheric phase fluctuations,
   which are significant at L-band for DSA-110's 2 km baselines.

5. **WSClean `-pol I` convention:** Returns `(XX + YY) / 2` per baseline in
   Jy/beam. Source fluxes in the sky model are Stokes I; the output image is
   in Stokes I units after WSClean combination.

---

## 7. Reference Figures

All figures are stored in `docs/images/`. See [`docs/images/README.md`](images/README.md)
for full descriptions and regeneration instructions.

| Figure | Key result |
|---|---|
| `docs/images/dsa110_antenna_layout.png` | ENU layout — T-core + outrigger distribution clearly visible |
| `docs/images/dsa110_psf_analysis.png` | PSF at Dec +16°: HPBW 75"×332", peak sidelobe 0.415, UV fill 0.90% |
| `docs/images/step2_calibrator_visibility.png` | Approved calibrator visibility output |
| `docs/images/step3_gain_solutions.png` | Approved gain solution output |
