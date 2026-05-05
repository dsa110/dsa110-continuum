# Ground Truth Audit 2026-05-04

This is a first-pass audit of `docs/OLD_GROUND_TRUTH.md`. Treat it as a review
surface, not as a replacement `docs/GROUND_TRUTH.md`.

## Method

I treated every statement in `OLD_GROUND_TRUTH.md` as a claim requiring evidence.
I used these evidence classes, in descending order of trust:

1. Current executable code or checked-in configuration.
2. Current tests that assert the behavior.
3. Current checked-in documentation that cites primary sources.
4. Recomputed values from current code.
5. Negative repo search results.
6. External paper / real-data claims, marked `needs_external_validation` unless
   directly checked in this pass.

Commands used included:

```bash
git status --short --branch
git log --oneline -- docs/GROUND_TRUTH.md
git show --stat --oneline 6ad8a00
rg -n 'GROUND_TRUTH|SimulationHarness|n_antennas|dsa110_measured_parameters|antennas.csv|DSA110_Station|integration_time|channel_width|subband|n_sky_sources|seed=42|OVRO|_OVRO|T_START|tile_2026' . -g '!outputs/**' -g '!*.ms/**'
PYTHONPATH=/data/dsa110-continuum /opt/miniforge/envs/casa6/bin/python - <<'PY'
from astropy.time import Time
import astropy.units as u
import numpy as np
from dsa110_continuum.simulation.harness import SimulationHarness, load_geodetic_enu, _OVRO_LON_DEG
h = SimulationHarness(n_antennas=117, n_integrations=24)
print(h.n_antennas, h.n_integrations, h.n_sky_sources, h.seed, h.pointing_dec_deg)
print(h.subband_freqs(0)[0] / 1e6, h.subband_freqs(15)[-1] / 1e6)
print(24 * 12.884902)
mid = Time(Time('2026-01-25T22:26:05', format='isot', scale='utc').jd + (24 * 12.884902) / 2 / 86400, format='jd', scale='utc')
print(float(mid.sidereal_time('apparent', longitude=_OVRO_LON_DEG * u.deg).deg))
enu = load_geodetic_enu(117)
print(float(enu[:,0].max() - enu[:,0].min()), float(enu[:,1].max() - enu[:,1].min()))
PY
```

## First-Pass Findings

The old file should not be restored as-is. It contains a mixture of current
facts, stale simulation decisions, and unverified external claims.

High-confidence current facts:

- `SimulationHarness` currently defaults to `n_antennas=117`, `n_integrations=24`,
  `n_sky_sources=20`, `seed=42`, and `pointing_dec_deg=16.15`.
- The current harness constants use OVRO lat `37.2339`, lon `-118.2825`, and
  altitude `1222.0 m`.
- The current spectral setup is 16 subbands, 48 channels per subband, channel
  width `244140.625 Hz`, 768 channels total.
- The current harness channel-center range is `1311.3720703125 MHz` to
  `1498.6279296875 MHz`; the full 768-channel width is `187.5 MHz`.
- Current `load_geodetic_enu(117)` reproduces the old file's station position
  table and spans: E span `1768.56 m`, N span `2219.73 m`.
- The simulation sky model is synthetic: random point sources with a power-law
  flux distribution and spectral-index scatter.

Stale or incorrect claims:

- `OLD_GROUND_TRUTH.md` called itself the single authoritative reference. It
  should not be treated that way.
- The old file says `DSA110_Station_Coordinates.csv` is human reference only and
  `antennas.csv` is used by `load_geodetic_enu()`. Current code does the
  opposite for the default real-position path: `load_geodetic_enu()` reads
  `DSA110_Station_Coordinates.csv`.
- The old file says the `SimulationHarness` default is `n_antennas=8`; current
  default is `117`.
- The old file says declination is fixed at `16.15 deg`. That is true for the
  simulation canary default, but not for the operational pipeline; current
  pipeline docs say real survey declination is read from observation metadata.
- The old YAML-inconsistency warning about a stale `325.520833 kHz` channel width
  is no longer current in the inspected YAML.
- The old active-antenna breakdown says `51 EW + 35 NS + 14 outriggers`; current
  `docs/dsa110-instrument.md` says `47 EW + 35 NS + 14 outriggers`.

Claims needing external validation before they become new ground truth:

- The active operational antenna count and exact active-ID breakdown. Repo docs
  cite Connor et al. 2025, but this pass did not independently verify the paper
  or real HDF5/MS antenna metadata.
- The assertion that no machine-readable active-96 list exists anywhere in the
  repo or sibling `dsa110-antpos`; this audit supports absence in this repo but
  did not inspect every external/sibling source.
- Commissioning configuration counts for `ant_ids.csv` and `ant_ids_mid.csv`.
- Reference figure numerical claims such as PSF HPBW, sidelobe level, and UV fill;
  these should be regenerated or checked against their source scripts.

## Proposed New `docs/GROUND_TRUTH.md` Shape

The replacement should be small and evidence-indexed:

1. Scope and trust policy.
2. Current repo-verified constants.
3. Simulation-only defaults.
4. Operational-pipeline facts.
5. External or real-data facts pending validation.
6. Explicitly rejected/stale facts from `OLD_GROUND_TRUTH.md`.

Every row should include:

- Claim
- Status: `repo_verified`, `computed`, `test_asserted`, `external_pending`,
  `stale_rejected`
- Evidence path and line number or command
- Last verified date

The CSV matrix in this directory is the working table for that rewrite.
