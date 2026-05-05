## Context

Default `pytest tests/` run currently reports `1037 passed, 11 skipped`. Six of those 11 skips come from `@pytest.mark.skipif(not MOSAIC.exists())` in `tests/test_two_stage_photometry.py`, where `MOSAIC = Path("pipeline_outputs/step6/step6_mosaic.fits")`. That artifact is not in the repo or working tree, and the artifact-generation scripts still reference a stale path: `/home/user/workspace/dsa110-continuum/pipeline_outputs/step6`.

Three are redundant with existing synthetic-FITS coverage and should be **deleted**. Three are still in scope but need to lose the artifact dependency in favour of `tmp_path`-generated FITS — so they should be **fixed**.

Companion issue tracks the other 5 skips (slow-marker gated): see "Skipped tests triage Group 1 — `@pytest.mark.slow`".

Triage doc: `outputs/skipped_tests_triage_2026-05-04/skipped_tests_issue.md` (committed locally; not pushed).

## Tests in scope

| # | test id | recommendation | effort |
|---|---------|----------------|--------|
| 6 | `tests/test_two_stage_photometry.py::test_coarse_pass_returns_finite` | delete | S |
| 7 | `tests/test_two_stage_photometry.py::test_snr_gate_all_pass_with_low_rms` | delete | S |
| 8 | `tests/test_two_stage_photometry.py::test_snr_gate_all_fail_with_high_rms` | delete | S |
| 9 | `tests/test_two_stage_photometry.py::test_beam_correction_ratio_bright_sources` | fix | M |
| 10 | `tests/test_two_stage_photometry.py::test_cli_simple_peak_sim_produces_csv` | fix | M |
| 11 | `tests/test_two_stage_photometry.py::test_cli_two_stage_sim_produces_coarse_snr_column` | fix | M |

## Per-test detail

### `test_coarse_pass_returns_finite` — delete (S)
Already covered by `test_coarse_pass_synthetic_fits`, which creates a small synthetic FITS and asserts finite peak/SNR behavior without a checked-out Step 6 artifact. Remove this redundant test.

### `test_snr_gate_all_pass_with_low_rms` — delete (S)
Adds only artifact coupling; the SNR-gate-pass scenario is covered by synthetic FITS tests already. If extra gate coverage is wanted, fold the low-RMS assertion into the existing synthetic coarse-pass test instead of keeping a skipped artifact-dependent test.

### `test_snr_gate_all_fail_with_high_rms` — delete (S)
`test_fine_pass_skips_failing_coarse` already covers the high-RMS failure path using a generated FITS. Redundant.

### `test_beam_correction_ratio_bright_sources` — fix (M)
Compares measured coarse peaks against `SimulationHarness(seed=42)` injected fluxes using the historical Step 6 mosaic. Replace the disk artifact with a generated synthetic FITS fixture under `tmp_path`, with WCS, BMAJ/BMIN, injected source positions, and expected ratio all created inside the test. Keep the flux-ratio assertion; make data provenance local to pytest.

### `test_cli_simple_peak_sim_produces_csv` — fix (M)
Depends on the missing Step 6 mosaic, writes its CSV through `tempfile.NamedTemporaryFile`, and runs the CLI with `cwd="/home/user/workspace/dsa110-continuum"` (a stale checkout path). Generate a small FITS mosaic in `tmp_path`, invoke the script from the actual repo root, write `--output` under `tmp_path`, and use `sys.executable` instead of hard-coded paths.

### `test_cli_two_stage_sim_produces_coarse_snr_column` — fix (M)
Same shape as the simple-peak CLI test: missing Step 6 dep, stale cwd, temp-CSV outside pytest basetemp. Apply the same generated-FITS / real-repo-cwd pattern. If `--sim` source positions are awkward for a small generated mosaic, add a test-only catalog path or call `run_forced_photometry()` directly for column-shape coverage and keep one subprocess CLI smoke for argument wiring.

## Proposed action plan
1. Delete the three redundant tests (`test_coarse_pass_returns_finite`, `test_snr_gate_all_pass_with_low_rms`, `test_snr_gate_all_fail_with_high_rms`) in one commit (S).
2. Replace artifact dependency in `test_beam_correction_ratio_bright_sources` with a `tmp_path`-generated FITS (M).
3. Rewrite the two CLI tests to generate a small FITS mosaic in `tmp_path`, drop the hard-coded `/home/user/workspace` cwd, and use `sys.executable` (M each).
4. Optional follow-up: scrub the artifact-generation scripts in `scripts/` for any other `/home/user/workspace/dsa110-continuum/...` paths.

## Acceptance
- `pytest tests/ -q` shows ≤ 5 skipped (all six of these resolved).
- No tests reference `pipeline_outputs/step6/step6_mosaic.fits` or `/home/user/workspace/...` paths.
- Triage doc updated or removed once all six are resolved.
