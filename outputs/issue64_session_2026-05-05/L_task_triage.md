# L task triage — `test_full_pipeline_recovers_sources`

## Outcome: deferred

## Original-fixture run (`--run-slow`)
The test hung in `pyuvdata.UVData.write_uvh5` during the corruption stage on H17,
captured at line 84 of the trace in `full_pipeline_run.log`:

```
File ".../dsa110_continuum/simulation/gain_corruption.py", line 105, in corrupt_uvh5
    uv.write_uvh5(str(output_path), clobber=True)
File ".../pyuvdata/uvh5.py", line 1171, in _write_header
    header["channel_width"] = self.channel_width
+++ Timeout +++
```

This is the same fuseblk + h5py interaction that bit the slow `TestSlow` cases
during this session (see commit log) — `/data` is fuseblk and per-syscall HDF5
write latency stalls under load. The 600 s pytest timeout fired well before
WSClean could even start.

## Shrunk-fixture probes (raw Python, with my own tempdir under `outputs/`)

| n_ant | n_sub | niter | imsize | cell | runtime | cal/img | recovered |
|-------|-------|-------|--------|------|---------|---------|-----------|
| 10    | 1     | 20    | 128    | 60"  | 55.7 s  | T/T     | 0 of 1    |
| 24    | 2     | 100   | 256    | 30"  | 122.5 s | T/T     | 0 of 1    |

The runtime / recovery tradeoff is genuinely tight. Smaller fixtures finish but
the source falls below the `n_recovered >= 1` threshold; larger fixtures recover
but exceed the 60 s default-suite budget *or* trip the pyuvdata write hang.

## What's needed to fix this properly (out of scope)

1. Decide whether the fuseblk write hang is a real concern for cloud CI (where
   tests run on a different filesystem) — if cloud-CI doesn't hang, the test
   may already be runnable elsewhere.
2. Tune the fixture so calibration / imaging / source recovery all pass at
   under 60 s on the target environment. The dominant levers are
   `image_size`, `niter`, `n_subbands`, and `n_antennas`; recovery requires
   enough UV coverage and SNR to clear the photometry threshold.
3. Possibly split the test: keep an end-to-end smoke that asserts
   `calibration_passed and imaging_passed and mosaic exists` (no recovery
   assertion), and move the recovered-source assertion behind a separate
   marker for opportunistic CI runs.

## Action taken

- Slow marker left in place on `TestEndToEnd::test_full_pipeline_recovers_sources`.
- Followup issue filed: #69.
- Issue #64 left open with this test called out as deferred and linked to #69.
