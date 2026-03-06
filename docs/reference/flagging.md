# Reference: Flagging

Source: `/data/dsa110-contimg/backend/src/dsa110_contimg/core/calibration/flagging.py`
        `/data/dsa110-contimg/config/dsa110-default.lua`

---

## Validated strategy: two-stage pipeline

The production flagging pipeline is a two-stage operation. Running only Stage 1
leaves a kurtosis of ~536 and >5σ outliers at 4× the Gaussian rate. Both stages
are required.

### Stage 1 — AOFlagger SumThreshold

Lua strategy: `/data/dsa110-contimg/config/dsa110-default.lua` (version 2026-02-08b)

Call: `flag_rfi_aoflagger(ms_path, strategy=<lua_path>, threads=<cpu_count>)`
in `flagging.py` lines 1529–1744.

The Lua file is located by `_get_default_aoflagger_strategy()`: checks
`CONTIMG_BASE_DIR/config/dsa110-default.lua`, then source-tree-relative, then CWD.
Binary: native `aoflagger` in PATH, Docker `aoflagger:latest` as fallback.

### Stage 2 — Post-AOFlagger 7σ MAD clip

`flag_residual_rfi_clip(ms_path, sigma=7.0)` in `flagging.py` lines 1211–1363.

Cross-correlation rows only (`ANTENNA1 != ANTENNA2`). Per-polarisation:
1. Compute median and `σ_MAD = 1.4826 × median(|x − median|)` of unflagged amplitudes.
2. Flag any visibility with `amplitude > median + 7.0 × σ_MAD`.

The default is global (not per-channel).

### Stage 3 — Flag extension

`flagdata(mode="extend", flagnearfreq=True, flagneartime=True, extendpols=True)`.
Falls back to direct casacore if CASA log lock is held.

### Canonical order

```
1. flag_zeros()
2. flag_autocorrelations()
3. flag_clip_amplitude(clip_range=[0, 0.5])      # 10× typical amplitude of 0.05
4. detect_and_flag_dead_antennas(threshold=0.95) # >95% flagged = dead
5. flag_rfi_aoflagger()                          # Stage 1
6. flag_residual_rfi_clip(sigma=7.0)             # Stage 2
7. flag_extend()                                 # Stage 3
```

Steps 1–4 must run BEFORE calibration (presets.py `detect_and_flag_dead_antennas`
docstring: "call AFTER zeros/autocorr flagging but BEFORE calibration").

---

## Validated Lua strategy parameters (dsa110-default.lua, 2026-02-08b)

| Parameter | Value | Notes |
|---|---|---|
| `base_threshold` | **1.2** | Was 1.0 in generic-default; raised for OVRO dense-core RFI |
| `iteration_count` | **4** | Was 3; extra iteration for heavier OVRO RFI environment |
| `threshold_factor_step` | **2.0** | Multiplier per iteration → per-iteration thresholds: 24σ, 12σ, 6σ |
| `transient_threshold_factor` | **0.8** | Was 1.0; more aggressive for Starlink/Iridium satellite passes |
| `frequency_resize_factor` | **2.0** | Was 1.0; extra frequency smoothing for 16-SPW structure |
| Low-pass kernel | **(21, 31)** | (time_width, freq_width) in pixels |
| Low-pass σ | **(2.5, 5.0)** | σ_time=2.5, σ_freq=5.0. σ_freq < 3.0 causes strategy failure |
| `threshold_timestep_rms` (inner loop) | **3.5σ** | Flags entire timesteps with anomalous RMS |
| `threshold_channel_rms` (inner loop) | **3.0 × threshold_factor** | Catches narrowband terrestrial transmitters per iteration |
| `threshold_timestep_rms` (final) | **4.0σ** | After morphological cleanup |

**Critical warning from Lua source (lines 67–72):**
Do NOT use `aoflagger.normalize_subbands()` with DSA-110 data.
- `normalize_subbands(data, 48)` → zeroes residuals → 0% flags (silent failure).
- `normalize_subbands(data, 768)` → AOFlagger hangs.

---

## Validated flagging fractions (2026-01-25T22:26:05.ms)

From Lua file header comments, lines 21–25:

| Stage | Total flagged | Kurtosis | >5σ outliers |
|---|---|---|---|
| AOFlagger only | 0.99% | 536 | 1.8% |
| + 7σ MAD clip | **2.44%** | **3.0** | 0.7% |
| Short baselines (<100 m) alone | — | 1492 → 0.7 | — |

The short-baseline kurtosis (1492 before MAD clip) is substantially higher than the
bulk array because short spacings are more susceptible to narrowband RFI.

---

## OVRO RFI environment (from dsa110-default.lua)

Dominant interference sources at the site (Lua lines 40–45):
- **LEO satellites** (Iridium, Starlink): broadband transient; the dominant RFI mode
  at OVRO. This is why `transient_threshold_factor=0.8` (more aggressive than generic).
- **GPS L2 (1227 MHz) / L1 (1575 MHz)**: at band edges, aliases possible.
- **Aircraft transponders (1090 MHz)**: below band, aliased.
- **Local electronics**: narrowband, intermittent.

Channels at `1350, 1395, 1405, 1422, 1475 MHz` show −8% to −27% amplitude dips in
the bandpass solutions. These are **not RFI in the raw visibilities** — the raw
cross-correlation amplitudes are flat to ±0.4% across all channels (validated
2026-01-25). The dips are bandpass solver artifacts: HI 21cm absorption at 1422 MHz
in the calibrator spectrum, and SPW edge effects at boundary channels.

AOFlagger correctly does not flag these channels. They are handled post-solve by
`clip_bandpass_outlier_channels()` in `calibration.py`, which flags channels in the
`.b` table that deviate >5% from a median-smoothed envelope. On the 2026-01-25
bandpass table this clipped 8 channels (0.76% additional flag fraction).

---

## CASA fallback flagging parameters (not recommended for production)

`flag_rfi(..., backend="casa")`:
- `tfcrop`: `timecutoff=4.0`, `freqcutoff=4.0`, `timefit="line"`, `freqfit="poly"`,
  `maxnpieces=5`, `winsize=3`, `extendflags=False`
- `rflag`: `timedevscale=4.0`, `freqdevscale=4.0`, `extendflags=False`

These are slow on a 24-timestep 768-channel MS and are not the validated path.
Use AOFlagger.

---

## DSA-110 instrument constants relevant to flagging

| Constant | Value | Source |
|---|---|---|
| Frequency range | 1311–1499 MHz, 187 MHz total | Lua line 33 |
| Spectral setup | 16 SPW × 48 ch = 768 channels, 244 kHz/ch | Lua line 34 |
| Time samples per scan | **24** (12.88 s integration, ~5 min total) | Lua line 35 |
| Antenna count | 117 (102 core + 15 outriggers) | Lua line 36 |
| Polarisations | XX, YY only (no cross-pols) | Lua line 37 |
| Median cross-corr amplitude | ~0.059 | Lua line 39 |
| Amplitude clip max | **0.5** (~10× typical) | flagging.py line 757 |
| Baseline distribution | 18% short (<100 m), 57% medium, 25% long | Lua line 38 |

The 24-timestep constraint is the most important. SumThreshold requires ≥~100
timesteps to achieve low false-positive rate; at 24 samples it leaves substantial
residuals (kurtosis ~536). The two-stage design exists specifically to compensate.
