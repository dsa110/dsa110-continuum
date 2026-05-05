# Skipped tests triage — 2026-05-04

Default `pytest tests/` run currently reports `1037 passed, 11 skipped`.
The ideal end state has zero skips; this doc enumerates each skip
with a recommendation: **fix** or **delete**.

Note: the requested full-suite grep command was started from the CASA6 Python environment, but it did not complete in this session before exceeding the expected runtime. A targeted pytest run over the 11 known skipped test ids confirmed the current skip ids and reasons below.

## Summary
| # | test id | mechanism | recommendation | effort |
|---|---------|-----------|----------------|--------|
| 1 | `tests/test_integration_e2e.py::TestSlow::test_full_96_antenna_subband` | `@pytest.mark.slow` gated by `tests/conftest.py` | **fix** | M |
| 2 | `tests/test_integration_e2e.py::TestSlow::test_all_16_subbands_closure` | `@pytest.mark.slow` gated by `tests/conftest.py` | **fix** | M |
| 3 | `tests/test_scattering_qa.py::test_check_tile_scattering_integration` | `@pytest.mark.slow` gated by `tests/conftest.py` | **fix** | M |
| 4 | `tests/test_simulated_pipeline.py::TestEndToEnd::test_full_pipeline_recovers_sources` | `@pytest.mark.slow` gated by `tests/conftest.py` | **fix** | L |
| 5 | `tests/test_simulated_pipeline.py::TestEndToEnd::test_full_pipeline_result_is_serializable` | `@pytest.mark.slow` gated by `tests/conftest.py` | **fix** | S |
| 6 | `tests/test_two_stage_photometry.py::test_coarse_pass_returns_finite` | `@pytest.mark.skipif` missing `pipeline_outputs/step6/step6_mosaic.fits` | **delete** | S |
| 7 | `tests/test_two_stage_photometry.py::test_snr_gate_all_pass_with_low_rms` | `@pytest.mark.skipif` missing `pipeline_outputs/step6/step6_mosaic.fits` | **delete** | S |
| 8 | `tests/test_two_stage_photometry.py::test_snr_gate_all_fail_with_high_rms` | `@pytest.mark.skipif` missing `pipeline_outputs/step6/step6_mosaic.fits` | **delete** | S |
| 9 | `tests/test_two_stage_photometry.py::test_beam_correction_ratio_bright_sources` | `@pytest.mark.skipif` missing `pipeline_outputs/step6/step6_mosaic.fits` | **fix** | M |
| 10 | `tests/test_two_stage_photometry.py::test_cli_simple_peak_sim_produces_csv` | `@pytest.mark.skipif` missing `pipeline_outputs/step6/step6_mosaic.fits` | **fix** | M |
| 11 | `tests/test_two_stage_photometry.py::test_cli_two_stage_sim_produces_coarse_snr_column` | `@pytest.mark.skipif` missing `pipeline_outputs/step6/step6_mosaic.fits` | **fix** | M |

## Group 1 — `@pytest.mark.slow`
### `tests/test_integration_e2e.py::TestSlow::test_full_96_antenna_subband`
**Skip reason.** `slow test (use --run-slow to enable)`.
**Why it is skipped.** `tests/conftest.py` adds a skip marker to all tests marked `slow` unless pytest is invoked with `--run-slow`. This test generates a realistic 96-antenna, 24-integration UVH5 subband through `SimulationHarness`.
**Underlying functionality in scope?** Yes — realistic synthetic UVH5 generation is a core cloud-safe validation path for conversion and simulation work.
**Recommendation: fix.** Keep the coverage but remove the default skip by making the test fit the normal-suite budget. Options are to reduce the fixture while preserving the 96-antenna metadata assertion, split the expensive file-generation check from the antenna-count invariant, or cache nothing and simply accept the runtime if measured under the project timeout.
**Effort estimate.** M — requires measuring runtime and possibly reshaping the fixture without weakening the science assertion.

### `tests/test_integration_e2e.py::TestSlow::test_all_16_subbands_closure`
**Skip reason.** `slow test (use --run-slow to enable)`.
**Why it is skipped.** The slow marker gates a loop over all 16 simulated subbands and closure checks.
**Underlying functionality in scope?** Yes — subband coverage and noiseless closure are important simulation invariants for the HDF5/UVH5 pipeline.
**Recommendation: fix.** Convert this into normal-suite coverage by parameterizing subbands or by using the smallest array that still exercises all 16 frequency slices. If the all-subband loop is already fast enough in CASA6, remove the slow marker directly.
**Effort estimate.** M — the work is mostly test restructuring and runtime measurement.

### `tests/test_scattering_qa.py::test_check_tile_scattering_integration`
**Skip reason.** `slow test (use --run-slow to enable)`.
**Why it is skipped.** The test is slow-gated, and the real `scattering` package is not installed in the current CASA6 environment. Most nearby tests mock the scattering module, but this one exercises `check_tile_scattering()` against a synthetic FITS and the real scattering transform path.
**Underlying functionality in scope?** Yes, but optional — scattering texture QA is wired into epoch QA, while the external scattering package is a non-PyPI QA dependency.
**Recommendation: fix.** Decide whether this is a dependency-infra test or a production-code test. If the project wants real scattering QA in default tests, install/package the `scattering` dependency in CASA6 test setup. If not, rewrite this as an integration-shaped test with a mocked scattering calculator so `check_tile_scattering()` is still exercised without external CPU-heavy synthesis.
**Effort estimate.** M — either dependency setup must be made reproducible, or the test needs a faithful mock of the calculator and synthesis path.

### `tests/test_simulated_pipeline.py::TestEndToEnd::test_full_pipeline_recovers_sources`
**Skip reason.** `slow test (use --run-slow to enable)`.
**Why it is skipped.** The test runs the simulated end-to-end path through corruption, calibration, WSClean imaging, mosaicking, and photometry. WSClean is available on this host, but the test is excluded from default collection by the slow gate.
**Underlying functionality in scope?** Yes — simulated end-to-end recovery is directly aligned with the project goal of validating pipeline behavior without relying on telescope data.
**Recommendation: fix.** Keep the test, but make it a default-running smoke with bounded runtime and deterministic resources. First run it with `--run-slow` to establish whether the blocker is runtime, flakiness, WSClean parameters, or production code. If runtime is the only issue, lower the image size, niter, tile count, or subband count until it is a true smoke while retaining one recovered-source assertion.
**Effort estimate.** L — this needs an actual run and may expose WSClean, simulation, or photometry failures rather than a simple marker removal.

### `tests/test_simulated_pipeline.py::TestEndToEnd::test_full_pipeline_result_is_serializable`
**Skip reason.** `slow test (use --run-slow to enable)`.
**Why it is skipped.** It sits inside the slow-gated end-to-end class even though the assertion is only that a `SimulatedPipelineResult` can be converted with `dataclasses.asdict()`.
**Underlying functionality in scope?** Yes — result serialization is useful, but it does not require a full WSClean-backed pipeline run.
**Recommendation: fix.** Rewrite this as a pure unit test that constructs `SimulatedPipelineResult` and `SourceFluxResult` directly, then calls `dataclasses.asdict()`. Remove it from the slow class.
**Effort estimate.** S — no production code investigation is needed.

## Group 2 — Missing `pipeline_outputs/step6/step6_mosaic.fits`
### `tests/test_two_stage_photometry.py::test_coarse_pass_returns_finite`
**Skip reason.** `Step 6 mosaic not on disk`.
**Why it is skipped.** The test is guarded by `@pytest.mark.skipif(not MOSAIC.exists())`, where `MOSAIC = Path("pipeline_outputs/step6/step6_mosaic.fits")`. That artifact is not present in the repo or current working tree.
**Underlying functionality in scope?** Yes — `run_coarse_pass()` should return finite measurements on valid in-footprint sources.
**Recommendation: delete.** This coverage is obsolete because `test_coarse_pass_synthetic_fits` already creates a small synthetic FITS and asserts finite peak/SNR behavior without a checked-out Step 6 artifact.
**Effort estimate.** S — remove the redundant artifact-dependent test.

### `tests/test_two_stage_photometry.py::test_snr_gate_all_pass_with_low_rms`
**Skip reason.** `Step 6 mosaic not on disk`.
**Why it is skipped.** It depends on the missing Step 6 mosaic and checks that a very low global RMS makes the source pass the coarse SNR gate.
**Underlying functionality in scope?** Yes — the SNR gate is part of two-stage forced photometry.
**Recommendation: delete.** This scenario is better covered by synthetic FITS tests; the current test adds only artifact coupling. If extra gate coverage is wanted, fold the low-RMS assertion into the existing synthetic coarse-pass test instead of keeping a skipped test.
**Effort estimate.** S — delete or merge into synthetic coverage.

### `tests/test_two_stage_photometry.py::test_snr_gate_all_fail_with_high_rms`
**Skip reason.** `Step 6 mosaic not on disk`.
**Why it is skipped.** It depends on the missing Step 6 mosaic and checks that a very high global RMS makes the source fail the coarse SNR gate.
**Underlying functionality in scope?** Yes — failing coarse-gate behavior is part of two-stage photometry.
**Recommendation: delete.** `test_fine_pass_skips_failing_coarse` already covers the high-RMS failure path using a generated FITS. This skipped test is redundant and artifact-dependent.
**Effort estimate.** S — remove the redundant test.

### `tests/test_two_stage_photometry.py::test_beam_correction_ratio_bright_sources`
**Skip reason.** `Step 6 mosaic not on disk`.
**Why it is skipped.** The test compares measured coarse peaks against injected fluxes from `SimulationHarness(seed=42)` using a historical Step 6 mosaic artifact. The artifact-generation scripts still point at `/home/user/workspace/dsa110-continuum/pipeline_outputs/step6`, which is not this checkout path.
**Underlying functionality in scope?** Yes — flux-ratio sanity checks for forced photometry are useful, but they should not rely on an untracked historical artifact.
**Recommendation: fix.** Replace the disk artifact with a generated synthetic FITS fixture whose WCS, BMAJ/BMIN, injected source positions, and expected ratio are created inside the test. Keep the flux-ratio assertion, but make the data provenance local to pytest.
**Effort estimate.** M — the test needs a realistic synthetic image fixture and revised expected ratio.

### `tests/test_two_stage_photometry.py::test_cli_simple_peak_sim_produces_csv`
**Skip reason.** `Step 6 mosaic not on disk`.
**Why it is skipped.** The test depends on the missing Step 6 mosaic, writes its CSV through `tempfile.NamedTemporaryFile`, and runs the CLI with `cwd="/home/user/workspace/dsa110-continuum"`, a stale checkout path.
**Underlying functionality in scope?** Yes — CLI smoke coverage for `scripts/forced_photometry.py --method simple_peak --sim` is in scope.
**Recommendation: fix.** Generate a small FITS mosaic in `tmp_path`, invoke the script from the actual repo root, and write `--output` under `tmp_path`. Use the CASA6 Python executable via `sys.executable`, but remove the hard-coded `/home/user/workspace` cwd and the Step 6 fixture dependency.
**Effort estimate.** M — mostly test-infra cleanup, with possible follow-up if `--sim` source positions do not land inside the synthetic WCS.

### `tests/test_two_stage_photometry.py::test_cli_two_stage_sim_produces_coarse_snr_column`
**Skip reason.** `Step 6 mosaic not on disk`.
**Why it is skipped.** Like the simple-peak CLI test, it depends on the missing Step 6 mosaic, uses a stale cwd, and writes to a temporary CSV outside the project-controlled pytest basetemp.
**Underlying functionality in scope?** Yes — CLI smoke coverage for the two-stage method and `coarse_snr`/`passed_coarse` output columns is in scope.
**Recommendation: fix.** Use the same generated FITS/real repo cwd pattern as the simple-peak CLI test. If `--sim` positions are awkward for a small generated mosaic, add a test-only catalog path or call `run_forced_photometry()` directly for column-shape coverage and keep one subprocess CLI smoke for argument wiring.
**Effort estimate.** M — same fixture work as the simple-peak CLI test, plus two-stage SNR tuning.

## Proposed action plan
- Delete the three redundant Step 6 coarse-pass artifact tests: `test_coarse_pass_returns_finite`, `test_snr_gate_all_pass_with_low_rms`, and `test_snr_gate_all_fail_with_high_rms`.
- Move `test_full_pipeline_result_is_serializable` out of the slow end-to-end class and make it a direct dataclass unit test.
- Replace Step 6 artifact dependencies in `test_beam_correction_ratio_bright_sources` and the two forced-photometry CLI tests with generated FITS fixtures under pytest `tmp_path`.
- Measure the two `tests/test_integration_e2e.py::TestSlow` runtimes with `--run-slow`; if they fit the normal-suite budget, remove the slow markers, otherwise shrink the fixtures while preserving 96-antenna and 16-subband coverage.
- Decide the scattering QA policy: install the non-PyPI `scattering` dependency for default tests, or rewrite `test_check_tile_scattering_integration` to use the same mocked-scattering pattern as the surrounding tests.
- Run the simulated-pipeline recovered-source test with `--run-slow` and triage the actual blocker. Treat this as a production-code or integration-parameter fix, not just a marker cleanup.

Recommendation tally: **8 fix**, **3 delete**, **0 unclear**.
