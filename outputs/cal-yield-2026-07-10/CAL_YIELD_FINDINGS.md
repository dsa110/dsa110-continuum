# Why 30% of antennas produce nulled (SNR=0) calibration solutions

**Session:** 2026-07-10 (continuing handoff-2026-07-10-14-25). Evidence in
`/data/dsa110-continuum/outputs/cal-yield-2026-07-10/`.

## The mechanism, proven end to end

The nulled bandpass solutions are NOT caused by pre-calibration flagging, NOT
by dead hardware (for ~11 antenna-pols), and (pending the 104/105 re-solve)
NOT primarily by refant choice. The cascade is:

1. **Pre-cal flagging is innocent.** After autocorr + AOFlagger on a fresh
   copy of the cal MS, total flags are 6.19% and the dead-antenna pass finds
   **zero** dead antennas (`refant_103/cal_antenna_health.json`:
   `dead_antennas: []`). The "30 dead" banner in the solve log is the
   *caltable* summary printed after the solve, not an MS-flagging event.

2. **The pre-bandpass phase solve is the guillotine.** `solve_prebandpass_phase`
   (gaincal, `solint=60s`, `combine_spw=True`, `uvrange='>1klambda'`,
   `minsnr=3`, **no K-table because `do_k=False`**) computes *real* solutions
   for the suspect antenna-pols with SNR clustered at **1–3**, just under
   minsnr=3, and flags them (production `.prebp`, 5 time bins):
   - 72 pol0: 0.9–2.1 (pol1: 7.8–9.5 passes)
   - 81 pol0: 0.9–4.4 (1 of 5 bins passes; pol1: 10.6–13.1)
   - 88 pol1: 0.8–1.5 (pol0: 2.9–4.3 mostly passes)
   - 90: 1.9–3.1 both pols; 92: 0.3–0.5 both; 96 pol1: 0.9–1.2;
     98 pol0: 1.6–3.1, pol1: 0.5–1.9
   - Genuinely dead/broken (SNR ≈ 0.0–0.2, incoherent or amp≈0): 18, 108,
     100, 47 pol0, 48 pol0, 70 pol1 (+70 pol0 huge delay).

3. **The flagged prebp table nulls them in the bandpass solve.** The prebp
   table is passed as `gaintable` to the bandpass solve; CASA excludes
   on-the-fly any data whose applied gain solution is flagged, so the
   bandpass solver never sees those antenna-pols. Result: B-table entries
   flagged with **SNR ≡ 0** (no solution attempted) — the "nulled" solutions.
   Proof: the `.b` null pattern equals the `.prebp` null pattern in BOTH the
   production tables and the fresh refant-103 rerun, degrading only in the
   expected direction (a pol with 1/5 surviving prebp bins can still die at
   the bandpass `minsnr=5`, e.g. 81 pol0, 90 pol1, 93 pol1). The 21
   correlator-absent antennas (9, 20, 21, 22, 51–66, 116) are nulled
   trivially — no rows in the data.

4. **applycal then flags those antenna-pols in every target MS** → the
   depleted (~62-antenna-equivalent) snapshot PSF lattice from the July-4
   forensics.

## Why their prebp SNR is 1–3 instead of ~35

Per-antenna-pol coherence on the phaseshifted cal MS
(`prebp_null_diagnostics.json`; baseline to refant, 12.5 Jy calibrator):

- The failing pols are **time-stable** (time coherence ≈ 0.93–0.97, same as
  healthy controls) but **frequency-decohered**: freq coherence 0.006–0.10.
  Their band spectra carry large uncorrected instrumental delays — the
  −15…−19 ns cluster (72, 88, 90, 93, 95, 96, 98), −434 ns (92), +193 ns
  (70 pol0), ~+2035 ns (100), ~+1550–1745 ns (18, 48 pol0, 70 pol1).
- `combine_spw=True` phase-averages the full 187 MHz band; a delay of a few
  tens of ns winds phase by many turns across the band and cancels the
  average → SNR collapses from ~35 to ~1–3 exactly at the minsnr cliff.
  Identical-looking pols split pass/fail by decimals (93 pol0 4.7–5.5 passes,
  93 pol1 2.2–3.0 fails), which is why the flagged list is stable
  epoch-to-epoch: it is a deterministic function of each pol's delay.
- `do_k=False` in `run_calibrator`, so no K-delay correction is ever applied —
  despite `solve_prebandpass_phase`'s own docstring: "Applying K-table is
  CRITICAL if instrumental delays are significant, as they cause phase
  decoherence when averaging over frequency (combine_spw=True)."

**Falsification test (conclusive).** Numerically removing a single fitted
delay term exp(2πiτf) from each nulled pol's band spectrum restores its
frequency coherence to healthy-control level:

| antenna-pol | fitted delay | freq coherence before → after |
| --- | --- | --- |
| 92 pol0 / pol1 | −434 ns | 0.010 / 0.008 → **0.973** |
| 90 pol0 | −18 ns | 0.093 → **0.978** |
| 93 pol1 | −16 ns | 0.018 → **0.973** |
| 98 pol0 | −17.7 ns | 0.075 → **0.976** |
| 70 pol0 | +193 ns | 0.006 → **0.974** |
| control: 1 pol0 | −1.3 ns | 0.887 → 0.978 |

A pure instrumental delay is the *entire* pathology for the recoverable
group; the signal chain behind these pols is otherwise healthy.

## Antenna classification (indices; name = index+1)

| class | antenna-pols | evidence |
| --- | --- | --- |
| absent from correlator | 9, 20, 21, 22, 51–66, 116 | no MS rows |
| broken (incoherent/dead/hot) | 18, 108 (both pols); 47 pol0 (amp 12× + incoherent); 48 pol0 (amp 0.2×); 70 pol1 (amp 0.04×); 100 (marginal amp, SNR~0) | time coherence ≤ 0.18 or amp anomaly |
| **recoverable: delay-wound, alive** | 72 pol0; 81 pol0; 88 pol1; 90 both; 92 both; 93 pol1; 96 pol1(+pol0 marginal); 98 both; 70 pol0; 47 pol1 | normal amp, time-coh ≈ 0.96, freq-coh ≤ 0.1 |

Recoverable ≈ 13 antenna-pols ≈ the "~11 good antennas unnecessarily lost"
from the July-4 forensics (which measured amplitude only, hence "apparently
normal").

## Fix directions (in expected-impact order)

1. **Solve K (delay) before prebp and pass it through** (`do_k=True`; the
   plumbing already exists: `prebp_gaintable=[ktable]`). Directly removes the
   decoherence that puts these pols at SNR 1–3.
2. **Or solve prebp per-SPW** (`combine_spw=False`): a 12 MHz SPW tolerates
   ~25× more delay before winding; costs √16 in SNR — irrelevant against a
   12.5 Jy calibrator.
3. Lowering prebp `minsnr` (3 → 2) would recover a few pols but keeps the
   decohered phase solutions; inferior to fixing the decoherence.
4. Refant change (104/105) — **tested, dead end.** With refant 104 the B-table
   yield change is net ZERO: antenna 90 recovers pol1 (prebp SNR 2.2–2.9 →
   2.8–3.3, 3/5 bins pass) but antenna 85 pol0 becomes a new casualty
   (3.3/3.2 passing → 2.3/2.5 failing). 92/96/98 unchanged. Refant choice
   shifts marginal SNRs by ±0.5 around the minsnr=3 cliff — it reshuffles
   which marginals die, it does not recover the delay-decohered population.
   The `bandpass_diagnostics.py` "try refant 104/105" suggestion does not fix
   yield. (Refant 105 run pending; expected same conclusion.)

## Rebuild gate (per handoff)

No production rebuild until a controlled canary (one tile) with the fixed
solve shows: recovered antenna-pols unflagged in `.b`/`.g`, flux scale
unchanged (3C454.3 12.5 Jy), and lattice amplitude reduced in the v1 gate
metrics.
