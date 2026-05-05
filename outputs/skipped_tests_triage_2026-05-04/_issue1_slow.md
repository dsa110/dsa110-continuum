## Context

Default `pytest tests/` run currently reports `1037 passed, 11 skipped`. Five of those 11 skips come from `@pytest.mark.slow` gating (gate added in `tests/conftest.py`, commit `ed9e303`/`7445f0b`). All five are in scope for the project, so they should be **fixed** — not deleted — by either tightening the fixtures, splitting the assertions, or removing the marker once runtime is measured.

Companion issue tracks the other 6 skips: see "Skipped tests triage Group 2 — missing Step 6 mosaic artifact".

Triage doc: `outputs/skipped_tests_triage_2026-05-04/skipped_tests_issue.md` (committed locally; not pushed).

## Tests in scope

| # | test id | recommendation | effort |
|---|---------|----------------|--------|
| 1 | `tests/test_integration_e2e.py::TestSlow::test_full_96_antenna_subband` | fix | M |
| 2 | `tests/test_integration_e2e.py::TestSlow::test_all_16_subbands_closure` | fix | M |
| 3 | `tests/test_scattering_qa.py::test_check_tile_scattering_integration` | fix | M |
| 4 | `tests/test_simulated_pipeline.py::TestEndToEnd::test_full_pipeline_recovers_sources` | fix | L |
| 5 | `tests/test_simulated_pipeline.py::TestEndToEnd::test_full_pipeline_result_is_serializable` | fix | S |

## Per-test detail

### `tests/test_integration_e2e.py::TestSlow::test_full_96_antenna_subband` — fix (M)
Generates a realistic 96-antenna, 24-integration UVH5 subband through `SimulationHarness`. Realistic UVH5 generation is a core cloud-safe validation path. Either reduce the fixture while preserving the 96-antenna metadata assertion, split the expensive file-generation check from the antenna-count invariant, or accept the runtime if it lands under the project timeout.

### `tests/test_integration_e2e.py::TestSlow::test_all_16_subbands_closure` — fix (M)
Loops over all 16 simulated subbands and runs closure checks. Subband coverage and noiseless closure are important simulation invariants. Parameterize subbands or use the smallest array that still exercises all 16 frequency slices; if it's already fast under CASA6, drop the slow marker.

### `tests/test_scattering_qa.py::test_check_tile_scattering_integration` — fix (M)
Exercises `check_tile_scattering()` against a synthetic FITS and the real `scattering` package, which is not installed in CASA6. Decide the policy: install the non-PyPI `scattering` dependency for default tests, or rewrite this as a mocked-scattering integration shape (matching the surrounding mocked tests) so `check_tile_scattering()` is still exercised without external CPU-heavy synthesis.

### `tests/test_simulated_pipeline.py::TestEndToEnd::test_full_pipeline_recovers_sources` — fix (L)
Runs the simulated end-to-end path: corruption → calibration → WSClean imaging → mosaic → photometry. WSClean is available on H17, but the test is excluded from default collection. Run it once with `--run-slow` to identify the actual blocker (runtime, flakiness, WSClean params, or production code). If runtime is the only issue, lower image size / niter / tile count / subband count until it's a true smoke retaining one recovered-source assertion.

### `tests/test_simulated_pipeline.py::TestEndToEnd::test_full_pipeline_result_is_serializable` — fix (S)
Sits inside the slow end-to-end class even though the assertion is only that a `SimulatedPipelineResult` survives `dataclasses.asdict()`. Rewrite as a pure unit test that constructs `SimulatedPipelineResult` and `SourceFluxResult` directly, then calls `asdict()`. Drop it from the slow class entirely.

## Proposed action plan
1. Pull `test_full_pipeline_result_is_serializable` out of `TestEndToEnd` and make it a unit test (S, blocker: none).
2. Decide scattering policy and either install the dep or convert to a mocked integration test (M).
3. Measure `TestSlow` runtimes in `test_integration_e2e.py` under `--run-slow`; remove slow marker if budget allows, else shrink the fixtures (M each).
4. Run `test_full_pipeline_recovers_sources` under `--run-slow` and triage the actual failure mode before assuming this is just a marker cleanup (L).

## Acceptance
- `pytest tests/ -q` shows ≤ 6 skipped (i.e. all 5 of these have been removed or unmarked).
- No new flaky tests introduced — slow fixes either preserve existing assertions or replace them with strictly equivalent ones.
- Triage doc updated or removed once all 5 are resolved.
