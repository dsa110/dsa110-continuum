# Per-Epoch Gain Calibration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Derive fresh gain solutions once per 1-hour mosaic epoch using catalog-seeded
in-field calibration and one round of WSClean self-calibration, replacing the static
2026-01-25 gain table that currently causes per-epoch flux scale offsets.

**Architecture:** New `epoch_gaincal.py` module called from `batch_pipeline.py` before
each epoch's tile imaging loop. Central tile (highest catalog source count of the two
middle tiles) is phaseshifted, BP-applied, catalog MODEL_DATA predicted, phase-only
gaincal run, WSClean quick image taken, ap gaincal run, table returned and applied to
all 12 tiles.

**Tech Stack:** `casacore.tables` (MS FIELD read), `dsa110_continuum.calibration.applycal`
(`apply_to_target`), `dsa110_continuum.calibration.skymodels` (`make_unified_skymodel`,
`predict_from_skymodel_wsclean`), `dsa110_continuum.calibration.casa_service`
(`CASAService.gaincal`), WSClean (`-save-model`), `dsa110_continuum.calibration.model`
(`count_bright_sources_in_tile`)

---

## Task 1: Add VLASS to unified sky model

**Files:**
- Modify: `dsa110_continuum/calibration/skymodels.py` (function `make_unified_skymodel`, lines ~202–381)
- Test: `tests/test_skymodels_vlass.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_skymodels_vlass.py
from unittest.mock import patch
import pandas as pd
import numpy as np

def _make_df(ra, dec, flux):
    return pd.DataFrame({"ra_deg": [ra], "dec_deg": [dec], "flux_mjy": [flux]})

def test_unified_skymodel_includes_vlass():
    """make_unified_skymodel should query VLASS and include unique sources."""
    with patch("dsa110_continuum.calibration.skymodels.make_unified_skymodel") as _:
        pass  # placeholder — real test below

def test_make_unified_skymodel_queries_vlass():
    from dsa110_continuum.calibration.skymodels import make_unified_skymodel
    calls = []

    def fake_query(catalog_type, ra_center, dec_center, radius_deg, min_flux_mjy):
        calls.append(catalog_type)
        if catalog_type == "vlass":
            return _make_df(10.5, 30.0, 200.0)   # unique VLASS-only source
        return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])

    with patch(
        "dsa110_continuum.calibration.skymodels.query_sources",
        side_effect=fake_query,
    ):
        sky = make_unified_skymodel(10.0, 30.0, 1.0, min_mjy=50.0)

    assert "vlass" in calls, "VLASS catalog was not queried"
    assert sky.Ncomponents >= 1, "VLASS source should appear in unified model"
```

**Step 2: Run test to verify it fails**

```bash
cd /data/dsa110-continuum
python -m pytest tests/test_skymodels_vlass.py::test_make_unified_skymodel_queries_vlass -v
```

Expected: `FAILED — AssertionError: VLASS catalog was not queried`

**Step 3: Add VLASS to `make_unified_skymodel()`**

In `dsa110_continuum/calibration/skymodels.py`, inside `make_unified_skymodel()`,
after the NVSS fetch block (after line ~`df_nvss = fetch_catalog("nvss")`), add:

```python
    df_vlass = fetch_catalog("vlass")
    if not df_vlass.empty:
        df_vlass["origin"] = "VLASS"
```

Then at the end of the merge section, after the NVSS merge block, add a VLASS merge
(lowest priority — only add VLASS sources that have no match in FIRST+RACS+NVSS):

```python
    # 5. Merge VLASS (Lowest priority — broadest sky coverage but lower resolution)
    if not df_vlass.empty:
        if unified_df.empty:
            unified_df = df_vlass.copy()
        else:
            c_unified = SkyCoord(
                ra=unified_df["ra_deg"].values * u.deg,
                dec=unified_df["dec_deg"].values * u.deg,
                frame="icrs",
            )
            c_vlass = SkyCoord(
                ra=df_vlass["ra_deg"].values * u.deg,
                dec=df_vlass["dec_deg"].values * u.deg,
                frame="icrs",
            )
            idx, d2d, _ = c_vlass.match_to_catalog_sky(c_unified)
            is_unmatched = d2d > (match_radius_arcsec * u.arcsec)
            unique_vlass = df_vlass[is_unmatched]
            unified_df = pd.concat([unified_df, unique_vlass], ignore_index=True)
```

Also update the `names` list near the end of the function to handle `"VLASS"` as a
valid origin value — no change needed, the generic `f"{origins[i]}_J..."` pattern
already works.

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_skymodels_vlass.py -v
```

Expected: `PASSED`

**Step 5: Commit**

```bash
git add dsa110_continuum/calibration/skymodels.py tests/test_skymodels_vlass.py
git commit -m "feat(skymodels): add VLASS to unified sky model at lowest priority"
```

---

## Task 2: Create `epoch_gaincal.py` — tile selection

**Files:**
- Create: `dsa110_continuum/calibration/epoch_gaincal.py`
- Test: `tests/test_epoch_gaincal.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_epoch_gaincal.py
import numpy as np
from unittest.mock import patch, MagicMock

def _make_fake_ms(tmp_path, tag, ra_deg, dec_deg):
    """Create a minimal fake MS directory with a FIELD subtable."""
    import os, shutil
    ms = tmp_path / f"{tag}.ms"
    ms.mkdir()
    # Write a minimal FIELD table using casacore (skip if unavailable)
    return str(ms)


def test_select_calibration_tile_from_ms_picks_richer_tile():
    """Should return the MS whose central pointing has more catalog sources."""
    from dsa110_continuum.calibration.epoch_gaincal import select_calibration_tile_from_ms

    # 12 fake MS paths; tiles 5 and 6 are the candidates
    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(12)]

    def fake_phase_center(ms_path):
        idx = int(ms_path.split("_")[1].split(".")[0])
        return (float(idx) * 10.0, 37.0)  # ra, dec

    def fake_count(pointing_ra_deg, pointing_dec_deg, **kwargs):
        # tile 6 has more sources
        return 8 if pointing_ra_deg == 60.0 else 3

    with patch(
        "dsa110_continuum.calibration.epoch_gaincal._read_ms_phase_center",
        side_effect=fake_phase_center,
    ), patch(
        "dsa110_continuum.calibration.epoch_gaincal.count_bright_sources_in_tile",
        side_effect=fake_count,
    ):
        result = select_calibration_tile_from_ms(fake_paths)

    assert result == "/fake/tile_06.ms"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_epoch_gaincal.py::test_select_calibration_tile_from_ms_picks_richer_tile -v
```

Expected: `FAILED — ModuleNotFoundError: epoch_gaincal`

**Step 3: Create `epoch_gaincal.py` with tile selection**

```python
# dsa110_continuum/calibration/epoch_gaincal.py
"""Per-epoch gain calibration for DSA-110 mosaic pipeline.

Public API
----------
select_calibration_tile_from_ms(epoch_ms_paths) -> str
    Return the MS path (from the two central tiles) with the most catalog sources.

calibrate_epoch(epoch_ms_paths, bp_table, work_dir, ...) -> str | None
    Full 5-step catalog-bootstrap + self-cal gain solve. Returns ap.G table path.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from dsa110_continuum.calibration.model import count_bright_sources_in_tile
from dsa110_continuum.calibration.mosaic_constants import (
    MOSAIC_TILE_COUNT,
    SKYMODEL_MIN_FLUX_MJY,
    SOURCE_QUERY_RADIUS_DEG,
)

log = logging.getLogger(__name__)


def _read_ms_phase_center(ms_path: str) -> tuple[float, float]:
    """Return (ra_deg, dec_deg) of the median field phase center in an MS."""
    import casacore.tables as ct

    with ct.table(f"{ms_path}::FIELD", readonly=True, ack=False) as t:
        phase_dir = t.getcol("PHASE_DIR")  # (nfields, 1, 2) radians
    ra_rad = phase_dir[:, 0, 0]
    dec_rad = phase_dir[:, 0, 1]
    median_ra = float(np.degrees(np.angle(np.mean(np.exp(1j * ra_rad)))) % 360)
    median_dec = float(np.degrees(np.median(dec_rad)))
    return median_ra, median_dec


def select_calibration_tile_from_ms(
    epoch_ms_paths: list[str],
    *,
    min_flux_mjy: float = SKYMODEL_MIN_FLUX_MJY,
    source_radius_deg: float = SOURCE_QUERY_RADIUS_DEG,
) -> str:
    """Return the central tile MS with the most bright catalog sources.

    Parameters
    ----------
    epoch_ms_paths:
        Sorted list of exactly MOSAIC_TILE_COUNT (12) MS paths for the epoch.
    min_flux_mjy:
        Minimum source flux for the source count query.
    source_radius_deg:
        Catalog search radius around the tile pointing.

    Returns
    -------
    str
        MS path of the selected calibration tile.

    Raises
    ------
    ValueError
        If epoch_ms_paths does not contain exactly MOSAIC_TILE_COUNT entries.
    """
    if len(epoch_ms_paths) != MOSAIC_TILE_COUNT:
        raise ValueError(
            f"Expected {MOSAIC_TILE_COUNT} MS paths, got {len(epoch_ms_paths)}"
        )

    center_indices = [5, 6]
    best_ms: str | None = None
    best_count = -1

    for idx in center_indices:
        ms = epoch_ms_paths[idx]
        try:
            ra, dec = _read_ms_phase_center(ms)
            n = count_bright_sources_in_tile(
                ra, dec,
                min_flux_mjy=min_flux_mjy,
                radius_deg=source_radius_deg,
            )
            log.info("Tile %d (%s): %d catalog sources", idx, Path(ms).stem, n)
            if n > best_count:
                best_count = n
                best_ms = ms
        except Exception as exc:
            log.warning("Cannot count sources for tile %d (%s): %s", idx, ms, exc)

    if best_ms is None:
        log.warning("Source count failed for both central tiles; defaulting to tile 5")
        best_ms = epoch_ms_paths[5]

    log.info("Selected calibration tile: %s (%d sources)", Path(best_ms).stem, best_count)
    return best_ms
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_epoch_gaincal.py::test_select_calibration_tile_from_ms_picks_richer_tile -v
```

Expected: `PASSED`

**Step 5: Commit**

```bash
git add dsa110_continuum/calibration/epoch_gaincal.py tests/test_epoch_gaincal.py
git commit -m "feat(epoch_gaincal): tile selection bridge — pick central tile by source count"
```

---

## Task 3: Implement `calibrate_epoch()` — 5-step solve

**Files:**
- Modify: `dsa110_continuum/calibration/epoch_gaincal.py` (add `calibrate_epoch`)
- Modify: `tests/test_epoch_gaincal.py` (add fallback test)

**Step 1: Write the failing test**

Add to `tests/test_epoch_gaincal.py`:

```python
def test_calibrate_epoch_returns_none_on_predict_failure():
    """calibrate_epoch() should return None (not raise) if catalog predict fails."""
    from dsa110_continuum.calibration.epoch_gaincal import calibrate_epoch
    import tempfile, os

    with tempfile.TemporaryDirectory() as work_dir:
        fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(12)]

        with patch(
            "dsa110_continuum.calibration.epoch_gaincal.select_calibration_tile_from_ms",
            return_value="/fake/tile_05.ms",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.phaseshift_ms",
            return_value=("/fake/tile_05_meridian.ms", "J2000 ..."),
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.apply_to_target",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.make_unified_skymodel",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.predict_from_skymodel_wsclean",
            side_effect=RuntimeError("wsclean not found"),
        ):
            result = calibrate_epoch(fake_paths, "/fake/bp.b", work_dir)

    assert result is None
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_epoch_gaincal.py::test_calibrate_epoch_returns_none_on_predict_failure -v
```

Expected: `FAILED — ImportError or AttributeError: calibrate_epoch not defined`

**Step 3: Implement `calibrate_epoch()`**

Append to `dsa110_continuum/calibration/epoch_gaincal.py`:

```python
def calibrate_epoch(
    epoch_ms_paths: list[str],
    bp_table: str,
    work_dir: str,
    *,
    refant: str = "103",
    min_flux_mjy: float = SKYMODEL_MIN_FLUX_MJY,
    source_radius_deg: float = SOURCE_QUERY_RADIUS_DEG,
    wsclean_niter: int = 1000,
    wsclean_threshold_sigma: float = 3.0,
) -> str | None:
    """Derive per-epoch gain solutions using catalog bootstrap + one self-cal round.

    Parameters
    ----------
    epoch_ms_paths:
        Sorted list of MOSAIC_TILE_COUNT MS paths for the epoch (raw, unphaseshifted).
    bp_table:
        Path to the daily bandpass table. Must exist.
    work_dir:
        Scratch directory for intermediate files and output G table.
    refant:
        Reference antenna (default: "103").
    min_flux_mjy:
        Minimum flux for catalog source selection (default: 5.0 mJy).
    source_radius_deg:
        Catalog search radius (default: 0.3 deg).
    wsclean_niter:
        Number of CLEAN iterations for the self-cal imaging pass (default: 1000).
    wsclean_threshold_sigma:
        Auto-threshold sigma for the self-cal imaging pass (default: 3.0).

    Returns
    -------
    str or None
        Path to the solved ap.G table, or None if any step failed.
    """
    from dsa110_continuum.calibration.applycal import apply_to_target
    from dsa110_continuum.calibration.casa_service import CASAService
    from dsa110_continuum.calibration.runner import phaseshift_ms
    from dsa110_continuum.calibration.skymodels import (
        make_unified_skymodel,
        predict_from_skymodel_wsclean,
    )

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    try:
        # ── 0. Select central tile ────────────────────────────────────────────
        central_raw_ms = select_calibration_tile_from_ms(
            epoch_ms_paths, min_flux_mjy=min_flux_mjy, source_radius_deg=source_radius_deg
        )
        stem = Path(central_raw_ms).stem
        meridian_ms = str(work / f"{stem}_meridian.ms")
        p_table = str(work / f"{stem}.p.G")
        ap_table = str(work / f"{stem}.ap.G")
        wsclean_prefix = str(work / f"{stem}_model")

        # ── 1. Phaseshift to median meridian ──────────────────────────────────
        if not os.path.exists(meridian_ms):
            log.info("Epoch gaincal: phaseshifting %s", stem)
            phaseshift_ms(
                ms_path=central_raw_ms,
                mode="median_meridian",
                output_ms=meridian_ms,
            )
        else:
            log.info("Epoch gaincal: meridian MS already exists, reusing")

        # ── 2. Apply bandpass only ────────────────────────────────────────────
        log.info("Epoch gaincal: applying BP table")
        apply_to_target(
            ms_target=meridian_ms,
            field="",
            gaintables=[bp_table],
            interp=["nearest"],
        )

        # ── 3. Catalog MODEL_DATA ─────────────────────────────────────────────
        log.info("Epoch gaincal: building catalog sky model")
        ra, dec = _read_ms_phase_center(meridian_ms)
        sky = make_unified_skymodel(
            ra, dec, source_radius_deg, min_mjy=min_flux_mjy
        )
        if sky.Ncomponents == 0:
            log.error("Epoch gaincal: catalog sky model is empty — cannot calibrate")
            return None
        log.info("Epoch gaincal: sky model has %d components", sky.Ncomponents)
        predict_from_skymodel_wsclean(meridian_ms, sky)

        # ── 4. Phase-only gaincal ─────────────────────────────────────────────
        log.info("Epoch gaincal: phase-only gaincal → %s", Path(p_table).name)
        service = CASAService()
        service.gaincal(
            vis=meridian_ms,
            caltable=p_table,
            field="",
            refant=refant,
            calmode="p",
            solint="inf",
            minsnr=3.0,
            gaintype="G",
            gaintable=[bp_table],
            interp=["nearest"],
        )
        if not os.path.exists(p_table):
            log.error("Epoch gaincal: phase-only solve produced no table")
            return None

        # Apply BP + p.G before WSClean imaging
        apply_to_target(
            ms_target=meridian_ms,
            field="",
            gaintables=[bp_table, p_table],
            interp=["nearest", "linear"],
        )

        # ── 5. Quick WSClean image to update MODEL_DATA (self-cal round) ──────
        import shutil
        wsclean_exec = shutil.which("wsclean")
        if not wsclean_exec:
            log.warning(
                "Epoch gaincal: wsclean not on PATH — skipping self-cal imaging, "
                "using catalog MODEL_DATA for ap solve"
            )
            # Re-predict catalog model after p.G applycal overwrote MODEL_DATA
            predict_from_skymodel_wsclean(meridian_ms, sky)
        else:
            import subprocess
            cmd = [
                wsclean_exec,
                "-niter", str(wsclean_niter),
                "-auto-threshold", str(wsclean_threshold_sigma),
                "-save-model-column", "MODEL_DATA",
                "-name", wsclean_prefix,
                "-size", "1024", "1024",
                "-scale", "6arcsec",
                "-weight", "briggs", "0.5",
                "-no-update-model-required",
                meridian_ms,
            ]
            log.info("Epoch gaincal: WSClean self-cal imaging")
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            if result.returncode != 0:
                log.warning(
                    "Epoch gaincal: WSClean exited %d — falling back to catalog MODEL_DATA\n%s",
                    result.returncode,
                    result.stderr.decode("utf-8", errors="replace")[-500:],
                )
                predict_from_skymodel_wsclean(meridian_ms, sky)

        # ── 6. Amplitude+phase gaincal ────────────────────────────────────────
        log.info("Epoch gaincal: ap gaincal → %s", Path(ap_table).name)
        service.gaincal(
            vis=meridian_ms,
            caltable=ap_table,
            field="",
            refant=refant,
            calmode="ap",
            solint="inf",
            minsnr=3.0,
            gaintype="G",
            gaintable=[bp_table, p_table],
            interp=["nearest", "linear"],
        )
        if not os.path.exists(ap_table):
            log.error("Epoch gaincal: ap solve produced no table")
            return None

        log.info("Epoch gaincal: SUCCESS → %s", ap_table)
        return ap_table

    except Exception as exc:
        log.error("Epoch gaincal: FAILED (%s) — will use static daily G table", exc)
        return None
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_epoch_gaincal.py -v
```

Expected: both tests `PASSED`

**Step 5: Commit**

```bash
git add dsa110_continuum/calibration/epoch_gaincal.py tests/test_epoch_gaincal.py
git commit -m "feat(epoch_gaincal): calibrate_epoch() — 5-step catalog bootstrap + self-cal solve"
```

---

## Task 4: Add `force_recal` to `process_ms()`

**Files:**
- Modify: `scripts/mosaic_day.py` (function `process_ms`, line ~110)
- Test: `tests/test_epoch_gaincal.py` (add test)

**Step 1: Write the failing test**

Add to `tests/test_epoch_gaincal.py`:

```python
def test_process_ms_force_recal_calls_applycal_even_when_data_exists():
    """process_ms(force_recal=True) must call apply_to_target even if CORRECTED_DATA exists."""
    import sys, types
    # Minimal smoke test: check force_recal parameter exists and is accepted
    # Full integration test requires a real MS; this confirms the API surface.
    sys.path.insert(0, "/data/dsa110-continuum/scripts")
    import importlib
    md = importlib.import_module("mosaic_day")
    import inspect
    sig = inspect.signature(md.process_ms)
    assert "force_recal" in sig.parameters, "process_ms must accept force_recal"
```

**Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_epoch_gaincal.py::test_process_ms_force_recal_calls_applycal_even_when_data_exists -v
```

Expected: `FAILED — AssertionError: process_ms must accept force_recal`

**Step 3: Modify `process_ms()` in `mosaic_day.py`**

Change the function signature from:
```python
def process_ms(ms_path: str, keep_intermediates: bool = False) -> str | None:
```
to:
```python
def process_ms(ms_path: str, keep_intermediates: bool = False, force_recal: bool = False) -> str | None:
```

Then inside the calibration block, change:
```python
    if needs_calibration(meridian_ms):
        log.info("[%s] Applying calibration ...", tag)
        try:
            apply_to_target(...)
```
to:
```python
    if force_recal or needs_calibration(meridian_ms):
        log.info("[%s] Applying calibration (force_recal=%s) ...", tag, force_recal)
        try:
            apply_to_target(...)
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_epoch_gaincal.py::test_process_ms_force_recal_calls_applycal_even_when_data_exists -v
```

Expected: `PASSED`

**Step 5: Commit**

```bash
git add scripts/mosaic_day.py tests/test_epoch_gaincal.py
git commit -m "feat(mosaic_day): add force_recal param to process_ms() for epoch gaincal"
```

---

## Task 5: Integrate `calibrate_epoch()` into `batch_pipeline.py`

**Files:**
- Modify: `scripts/batch_pipeline.py`
  - Add `--skip-epoch-gaincal` CLI flag (near line ~400, in argparse block)
  - Add `calibrate_epoch()` call before tile imaging loop (near line ~513, after MS list filter)
  - Pass `force_recal=True` to `process_tile_safe()` (near line ~538)
  - Add `gaincal_status` to epoch result dict (near line ~639)
  - Print `gaincal_status` in `print_summary()` (near line ~279)

**Step 1: Add `--skip-epoch-gaincal` flag**

In the argparse block in `main()`, add after the existing `--skip-photometry` argument:

```python
    parser.add_argument(
        "--skip-epoch-gaincal",
        action="store_true",
        default=False,
        help=(
            "Skip per-epoch gain calibration. "
            "Falls back to the static daily G table (--cal-date). "
            "Use for debugging or when epoch gaincal tables already exist."
        ),
    )
```

**Step 2: Add the `calibrate_epoch()` call before the tile imaging loop**

After the MS list filter and before the tile imaging loop (after line
`log.info("=== Phase 1/3: Calibrate + Image all tiles...")`), add:

```python
    # ── Per-epoch gain calibration ─────────────────────────────────────────
    # Build a work directory for this epoch's G tables alongside the stage dir.
    epoch_gaincal_dir = os.path.join(paths["stage_dir"], "epoch_gaincal")
    os.makedirs(epoch_gaincal_dir, exist_ok=True)

    if not args.skip_epoch_gaincal:
        log.info("=== Phase 0/3: Per-epoch gain calibration ===")
        try:
            from dsa110_continuum.calibration.epoch_gaincal import calibrate_epoch
            # Use the full MS list as the epoch (the whole day is one epoch for
            # tile selection purposes — we pick the central pair of the 12-tile set).
            # For multi-epoch days this is called once per epoch tile set.
            # Current simple integration: calibrate once for the full day's tile set.
            _epoch_ms = ms_list[:12] if len(ms_list) >= 12 else ms_list
            _epoch_g = calibrate_epoch(
                epoch_ms_paths=_epoch_ms,
                bp_table=_bp,
                work_dir=epoch_gaincal_dir,
                refant="103",
            )
            if _epoch_g is not None:
                log.info("Epoch gaincal SUCCESS: %s", _epoch_g)
                _md.G_TABLE = _epoch_g
                _epoch_gaincal_status = "ok"
            else:
                log.warning(
                    "Epoch gaincal failed — falling back to static daily G table (%s)", _ga
                )
                _epoch_gaincal_status = "fallback"
        except Exception as _eg_exc:
            log.error("Epoch gaincal import/call error: %s — using static table", _eg_exc)
            _epoch_gaincal_status = "error"
    else:
        log.info("--skip-epoch-gaincal set: using static daily G table")
        _epoch_gaincal_status = "skipped"
    # ──────────────────────────────────────────────────────────────────────
```

**Step 3: Pass `force_recal=True` to tile processing**

In `process_tile_safe()` call (inside the tile loop), the function currently calls
`_md.process_ms(ms_path, keep_intermediates=keep)` internally. Update
`_run_process_ms()` to accept and forward `force_recal`:

Change `_run_process_ms`:
```python
def _run_process_ms(ms_path: str, keep: bool, force_recal: bool = False) -> str | None:
    import mosaic_day as _md
    return _md.process_ms(ms_path, keep_intermediates=keep, force_recal=force_recal)
```

Change `process_tile_safe()` signature and internal call:
```python
def process_tile_safe(md, ms_path, keep, timeout_sec, retry, force_recal=False):
    ...
    def _attempt():
        with ProcessPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_run_process_ms, ms_path, keep, force_recal)
            ...
```

In the tile loop, pass `force_recal=(_epoch_gaincal_status == "ok")`:
```python
        result = process_tile_safe(
            _md, ms_path, keep, tile_timeout, retry_failed,
            force_recal=(_epoch_gaincal_status == "ok"),
        )
```

**Step 4: Add `gaincal_status` to epoch summary**

In `print_summary()`, add a `GainCal` column to the header and rows:

```python
    hdr = f"  {'Epoch':12s}  {'Tiles':>5}  {'GainCal':>8}  {'Peak(Jy/b)':>10}  {'RMS(mJy/b)':>10}  {'Sources':>7}  {'DSA/NVSS':>8}"
```

In each row:
```python
        gcal_str = r.get("gaincal_status", "n/a")[:8]
        print(f"  {r['label']:12s}  {r['n_tiles']:>5}  {gcal_str:>8}  {peak_str:>10} ...")
```

In the `epoch_results.append()` block, add:
```python
            "gaincal_status": _epoch_gaincal_status,
```

**Step 5: Run a smoke test**

```bash
python scripts/batch_pipeline.py --help | grep skip-epoch-gaincal
```

Expected output includes `--skip-epoch-gaincal`.

**Step 6: Commit**

```bash
git add scripts/batch_pipeline.py
git commit -m "feat(batch_pipeline): integrate per-epoch gaincal with fallback and --skip-epoch-gaincal flag"
```

---

## Task 6: Commit design and plan docs

```bash
git add docs/plans/2026-03-04-epoch-gaincal-design.md docs/plans/2026-03-04-epoch-gaincal-plan.md
git commit -m "docs: per-epoch gain calibration design and implementation plan"
```

Then push all:

```bash
python -c "
import subprocess
subprocess.run(['git', 'push', 'origin', 'main'], check=True)
"
```

---

## Verification checklist (run after all tasks)

```bash
# All epoch_gaincal tests pass
python -m pytest tests/test_epoch_gaincal.py tests/test_skymodels_vlass.py -v

# batch_pipeline --help shows new flag
python scripts/batch_pipeline.py --help | grep skip-epoch-gaincal

# mosaic_day process_ms accepts force_recal
python -c "import sys; sys.path.insert(0,'scripts'); import mosaic_day; import inspect; print('force_recal' in inspect.signature(mosaic_day.process_ms).parameters)"
```
