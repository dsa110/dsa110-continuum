# Reference Implementation Docs

This directory contains distillations of validated implementation knowledge from two
reference codebases: the original DSA-110 pipeline (`dsa110-contimg`) and the ASKAP
VAST pipeline (`askap-vast`). These files exist so that developers working in
`dsa110-continuum` do not have to re-discover validated parameters, known failure
modes, or design decisions that were already worked out.

## Files

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
