# Reference Implementation Docs

This directory contains two kinds of reference material:

1. **Instrument & simulation ground truth** — verified DSA-110 parameters that
   every agent session should read before writing simulation or pipeline code.
2. **Pipeline implementation knowledge** — distillations from the original
   `dsa110-contimg` and ASKAP VAST pipelines so developers don't re-discover
   validated parameters or known failure modes.

## Ground Truth (read this first)

| File | Contents |
|---|---|
| [`../GROUND_TRUTH.md`](../GROUND_TRUTH.md) | **Single authoritative reference** for array geometry (117-station positions, active antenna count, ENU extents), timing (12.885 s integrations, tile structure), spectral setup (244.14 kHz channels, 768-channel bandwidth), sky model, and known simulation limitations. Updated 2026-04-17. |

> **Start every simulation session by reading `docs/GROUND_TRUTH.md`.**
> It contains the exact harness configuration to use (`n_antennas=117`) and
> explains why common alternatives (96, 8) are wrong.

## Pipeline Reference Files

| File | Source | Contents |
|---|---|---|
| `flagging.md` | `dsa110-contimg` | AOFlagger Lua strategy, validated fractions, OVRO RFI, flagging order |
| `calibration.md` | `dsa110-contimg` | K/B/G parameters, DEFAULT_PRESET, self-cal strategy, flux validation |
| `conversion-and-qa.md` | `dsa110-contimg` | UVH5 ingest, PyUVData workarounds, TELESCOPE_NAME, QA thresholds |
| `imaging.md` | `dsa110-contimg` | WSClean defaults, sky model seeding, IDG merge, Galvin mask |
| `mosaicking.md` | `dsa110-contimg` | QUICKLOOK and SCIENCE/DEEP configs, mean-RA wrap bug |
| `photometry-and-ese.md` | `dsa110-contimg` | Condon matched-filter, differential photometry, ESE scoring |
| `vast-crossref.md` | `askap-vast` | Variability metrics, forced photometry, source association, Condon errors |

## How to use

Before implementing any subsystem listed above, read the corresponding file here.
The files tell you:
1. What the validated parameters are and why
2. Which subtle bugs and instrument-specific quirks must be preserved
3. The exact file path in the reference codebase if you need to read the original

## Reference codebase paths

- `dsa110-contimg`: `/data/dsa110-contimg/backend/src/dsa110_contimg/core/`
- VAST: `/data/radio-pipelines/askap-vast/`
