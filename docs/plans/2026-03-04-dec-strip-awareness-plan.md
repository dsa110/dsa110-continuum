# Dec-Strip Awareness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent the pipeline from silently producing bad images when the observed declination strip does not match the calibration tables, and expose the Dec strip in the data inventory.

**Architecture:** Thin detection layer — a single `read_ms_dec()` utility is shared by `batch_pipeline.py` and `inventory.py`. No new dependencies. All changes confined to three files plus one new utility and two new test files.

**Tech Stack:** Python 3.11, casacore (via `casacore.tables`), astropy, pytest

---

### Task 1: Add `read_ms_dec()` utility

**Files:**
- Create: `dsa110_continuum/calibration/dec_utils.py`
- Test: `tests/test_dec_utils.py`

The function reads FIELD::PHASE_DIR from a Measurement Set and returns the median Dec in degrees. This already exists privately in `epoch_gaincal.py` as `_read_ms_phase_center()`; we promote it to a public, tested utility and add a FITS fallback.

**Step 1: Write the failing test**

```python
# tests/test_dec_utils.py
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

def test_read_ms_dec_from_ms(tmp_path):
    """read_ms_dec returns Dec from MS FIELD table."""
    from dsa110_continuum.calibration.dec_utils import read_ms_dec

    mock_table = MagicMock()
    mock_table.__enter__ = lambda s: s
    mock_table.__exit__ = MagicMock(return_value=False)
    # PHASE_DIR shape: (n_fields, 1, 2) in radians
    mock_table.getcol.return_value = np.array([[[0.5, 0.28]]]) # RA=0.5 rad, Dec=0.28 rad≈16.04°
    mock_table.nrows.return_value = 1

    with patch("casacore.tables.table", return_value=mock_table):
        dec = read_ms_dec(str(tmp_path / "test.ms"))

    assert abs(dec - np.degrees(0.28)) < 0.01


def test_read_ms_dec_falls_back_to_fits(tmp_path):
    """read_ms_dec falls back to CRVAL2 when MS read fails."""
    from astropy.io import fits
    from dsa110_continuum.calibration.dec_utils import read_ms_dec

    fits_path = tmp_path / "tile.fits"
    hdu = fits.PrimaryHDU()
    hdu.header["CRVAL1"] = 40.0
    hdu.header["CRVAL2"] = 33.0
    hdu.writeto(fits_path)

    dec = read_ms_dec(str(tmp_path / "nonexistent.ms"), fits_fallback=str(fits_path))
    assert abs(dec - 33.0) < 0.01


def test_read_ms_dec_raises_when_both_fail(tmp_path):
    """read_ms_dec raises RuntimeError when MS and FITS fallback both fail."""
    from dsa110_continuum.calibration.dec_utils import read_ms_dec

    with pytest.raises(RuntimeError, match="Cannot determine Dec"):
        read_ms_dec(str(tmp_path / "nonexistent.ms"))
```

**Step 2: Run test to verify it fails**

```bash
cd /data/dsa110-continuum
/opt/miniforge/envs/casa6/bin/python -m pytest tests/test_dec_utils.py -v
```

Expected: `ImportError: cannot import name 'read_ms_dec'`

**Step 3: Write minimal implementation**

```python
# dsa110_continuum/calibration/dec_utils.py
"""Utilities for reading the observed declination from a Measurement Set or FITS tile."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def read_ms_dec(ms_path: str, fits_fallback: str | None = None) -> float:
    """Return the median observed declination (degrees) for a Measurement Set.

    Reads FIELD::PHASE_DIR (column shape n_fields × 1 × 2, values in radians).
    If the MS cannot be opened, tries *fits_fallback* CRVAL2.

    Raises
    ------
    RuntimeError
        If neither source yields a valid Dec.
    """
    try:
        import casacore.tables as ct
        with ct.table(ms_path + "/FIELD", readonly=True, ack=False) as t:
            phase_dir = t.getcol("PHASE_DIR")   # (n_fields, 1, 2)
        dec_rad = np.median(phase_dir[:, 0, 1])
        return float(np.degrees(dec_rad))
    except Exception as e:
        log.debug("read_ms_dec: MS read failed (%s), trying FITS fallback", e)

    if fits_fallback is not None:
        try:
            from astropy.io import fits as _fits
            with _fits.open(fits_fallback) as hdul:
                crval2 = hdul[0].header.get("CRVAL2")
            if crval2 is not None:
                return float(crval2)
        except Exception as e2:
            log.debug("read_ms_dec: FITS fallback failed (%s)", e2)

    raise RuntimeError(
        f"Cannot determine Dec for {ms_path!r}: MS unreadable and no valid FITS fallback."
    )
```

**Step 4: Run tests to verify they pass**

```bash
/opt/miniforge/envs/casa6/bin/python -m pytest tests/test_dec_utils.py -v
```

Expected: 3 PASSED

**Step 5: Commit**

```bash
git add dsa110_continuum/calibration/dec_utils.py tests/test_dec_utils.py
git commit -m "feat: add read_ms_dec() utility for dec-strip detection"
```

---

### Task 2: Add Dec-strip guard to `batch_pipeline.py`

**Files:**
- Modify: `scripts/batch_pipeline.py` (cal-table validation block, lines ~460–490)
- Test: `tests/test_batch_dec_guard.py`

After the existing cal-table existence check, detect the epoch Dec and abort if it
differs from `--expected-dec` by more than `DEC_CHANGE_THRESHOLD_DEG`.

**Step 1: Write the failing test**

```python
# tests/test_batch_dec_guard.py
"""Tests for the Dec-strip guard in batch_pipeline."""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_dec_guard_passes_when_dec_matches():
    from batch_pipeline import check_dec_strip
    # Should not raise when observed Dec matches expected
    check_dec_strip(observed_dec=16.1, expected_dec=16.1, threshold_deg=5.0)


def test_dec_guard_passes_within_threshold():
    from batch_pipeline import check_dec_strip
    # 3° offset is inside the 5° threshold
    check_dec_strip(observed_dec=19.0, expected_dec=16.1, threshold_deg=5.0)


def test_dec_guard_raises_on_mismatch():
    from batch_pipeline import check_dec_strip
    with pytest.raises(SystemExit):
        check_dec_strip(observed_dec=33.0, expected_dec=16.1, threshold_deg=5.0)


def test_dec_guard_raises_on_large_negative_mismatch():
    from batch_pipeline import check_dec_strip
    with pytest.raises(SystemExit):
        check_dec_strip(observed_dec=54.5, expected_dec=16.1, threshold_deg=5.0)
```

**Step 2: Run test to verify it fails**

```bash
/opt/miniforge/envs/casa6/bin/python -m pytest tests/test_batch_dec_guard.py -v
```

Expected: `ImportError: cannot import name 'check_dec_strip'`

**Step 3: Add `check_dec_strip()` and wire it into `main()`**

In `scripts/batch_pipeline.py`, add the function near the top of the file (after imports):

```python
def check_dec_strip(
    observed_dec: float,
    expected_dec: float,
    threshold_deg: float = 5.0,
) -> None:
    """Abort if the observed Dec strip differs from the expected calibration strip."""
    from dsa110_continuum.calibration.mosaic_constants import DEC_CHANGE_THRESHOLD_DEG
    thr = threshold_deg if threshold_deg != 5.0 else DEC_CHANGE_THRESHOLD_DEG
    delta = abs(observed_dec - expected_dec)
    if delta > thr:
        log.error(
            "ABORT: observed Dec %.1f° differs from expected %.1f° by %.1f° "
            "(threshold %.1f°). "
            "Calibration tables derived at %.1f° — applying them here will produce "
            "invalid flux scale. "
            "Re-run with --expected-dec %.1f if you have cal tables for that strip, "
            "or select only Dec≈%.1f° observations for this cal date.",
            observed_dec, expected_dec, delta, thr,
            expected_dec, observed_dec, expected_dec,
        )
        sys.exit(1)
    log.info(
        "Dec-strip check passed: observed %.1f° vs expected %.1f° (delta %.1f°)",
        observed_dec, expected_dec, delta,
    )
```

Add `--expected-dec` to the argparser (near `--cal-date`):

```python
parser.add_argument(
    "--expected-dec",
    type=float,
    default=16.1,
    metavar="DEG",
    help="Expected pointing declination for this cal-table strip (default: 16.1°). "
         "Pipeline aborts if the first MS differs by more than DEC_CHANGE_THRESHOLD_DEG.",
)
```

Wire the check into `main()` immediately after the existing cal-table validation block (after the `log.info("Cal tables verified …")` line):

```python
# ── Dec-strip validation ───────────────────────────────────────────────────
from dsa110_continuum.calibration.dec_utils import read_ms_dec
_first_ms = sorted(
    f for f in os.listdir(MS_DIR)
    if f.endswith(".ms") and f.startswith(date)
)
if _first_ms:
    try:
        _obs_dec = read_ms_dec(os.path.join(MS_DIR, _first_ms[0]))
        check_dec_strip(_obs_dec, args.expected_dec)
    except RuntimeError as _e:
        log.warning("Could not determine observed Dec (%s) — continuing without check", _e)
else:
    log.warning("No MS files found yet for Dec-strip check — check will be skipped")
# ───────────────────────────────────────────────────────────────────────────
```

**Step 4: Run tests to verify they pass**

```bash
/opt/miniforge/envs/casa6/bin/python -m pytest tests/test_batch_dec_guard.py -v
```

Expected: 4 PASSED

**Step 5: Commit**

```bash
git add scripts/batch_pipeline.py tests/test_batch_dec_guard.py
git commit -m "feat(batch_pipeline): abort on dec-strip mismatch between obs and cal tables"
```

---

### Task 3: Add Dec column to `inventory.py`

**Files:**
- Modify: `scripts/inventory.py` (CSV writer block and human-readable report, lines ~140–200)

**Step 1: Add helper in inventory.py**

Add this function after the MS scan function (`scan_ms`):

```python
def read_dec_for_timestamp(ms_dir: Path, timestamp: str) -> float | None:
    """Return Dec (deg) for a converted MS, or None if not yet converted."""
    from dsa110_continuum.calibration.dec_utils import read_ms_dec
    ms_path = ms_dir / f"{timestamp}.ms"
    if not ms_path.exists():
        return None
    try:
        return round(read_ms_dec(str(ms_path)), 1)
    except Exception:
        return None
```

**Step 2: Wire into CSV rows and report**

In the CSV fieldnames list, add `"dec_deg"` after `"has_ms"`:

```python
fieldnames = ["date", "timestamp", "n_subbands_found", "n_subbands_expected",
              "is_complete", "size_bytes", "has_ms", "dec_deg", "group_source"]
```

In the CSV row construction, add:

```python
"dec_deg": read_dec_for_timestamp(MS_DIR, ts),
```

In the human-readable per-row print, add Dec after the MS column:

```python
dec_str = f"{info.get('dec_deg', '?'):>5}" if info.get('dec_deg') is not None else "    ?"
print(
    f"   {ts:<22}  {info['source']:>3}  "
    f"{info['n_subbands']:>5}  "
    f"{'Y' if info['is_complete'] else 'N':>3}  "
    f"{'Y' if ts in converted_ts else 'N':>3}  "
    f"{dec_str}°  "
    f"{fmt_bytes(info['size_bytes'])}{flag}"
)
```

Update the column header line to match:

```python
print(f"   {'TIMESTAMP':<22}  {'SRC':>3}  {'FOUND':>5}  {'OK':>3}  {'MS':>3}  {'DEC':>5}  SIZE")
```

Note: `read_dec_for_timestamp` is called per-row so it only fires for already-converted MS files; there is no performance impact for unconverted timestamps.

**Step 3: Run inventory to verify output**

```bash
/opt/miniforge/envs/casa6/bin/python scripts/inventory.py 2>&1 | head -60
```

Expected: table now shows a `DEC` column; entries for Jan25, Feb15 show `16.1°`, Feb23 shows `33.0°`, Feb26 shows `54.5°`.

**Step 4: Commit**

```bash
git add scripts/inventory.py
git commit -m "feat(inventory): add dec_deg column to show observed dec strip per epoch"
```

---

### Task 4: Diagnostic re-run of Feb 15 (correct Dec strip)

**Goal:** Determine whether the Feb 15 near-zero flux ratios are a pre-existing pipeline bug or a data quality issue. Run Feb 15 T00h through the current pipeline from scratch and compare the new mosaic to the old one.

**Step 1: Clear the old Feb 15 products**

```bash
rm -rf /stage/dsa110-contimg/images/mosaic_2026-02-15/
rm -rf /data/dsa110-continuum/products/mosaics/2026-02-15/
```

**Step 2: Run batch pipeline for Feb 15**

```bash
nohup /opt/miniforge/envs/casa6/bin/python scripts/batch_pipeline.py \
    --date 2026-02-15 \
    --cal-date 2026-01-25 \
    --expected-dec 16.1 \
    > /tmp/feb15_rerun.log 2>&1 &
echo "PID: $!"
```

**Step 3: Monitor and capture mosaic stats**

Once the run completes (watch `/tmp/feb15_rerun.log`):

```bash
/opt/miniforge/envs/casa6/bin/python -c "
from astropy.io import fits
import numpy as np, glob
for f in sorted(glob.glob('/stage/dsa110-contimg/images/mosaic_2026-02-15/*mosaic.fits')):
    d = fits.getdata(f).squeeze()
    h = fits.getheader(f)
    peak = np.nanmax(d); rms = np.nanstd(d[np.isfinite(d)])
    print(f'{f}: peak={peak:.3f} Jy/bm  rms={rms*1e3:.1f} mJy  DR={peak/rms:.0f}  RA={h[\"CRVAL1\"]:.1f}  Dec={h[\"CRVAL2\"]:.1f}')
"
```

**Step 4: Compare forced photometry**

```bash
/opt/miniforge/envs/casa6/bin/python -c "
import pandas as pd, glob
for f in sorted(glob.glob('/data/dsa110-continuum/products/mosaics/2026-02-15/*.csv')):
    df = pd.read_csv(f)
    print(f'{f}: n={len(df)}  median_ratio={df.dsa_nvss_ratio.median():.3f}  p25={df.dsa_nvss_ratio.quantile(.25):.3f}  p75={df.dsa_nvss_ratio.quantile(.75):.3f}')
"
```

Expected if fix is working: median DSA/NVSS ratio 0.7–1.2 (target: 0.8–1.2).
If ratio is still near-zero: escalate to imaging/mosaicking investigation (see TODOS.md "Investigate leftmost mosaic tiles").

**Step 5: Update lightcurves parquet**

Once Feb 15 photometry CSVs look healthy:

```bash
/opt/miniforge/envs/casa6/bin/python scripts/stack_lightcurves.py
/opt/miniforge/envs/casa6/bin/python scripts/variability_metrics.py
/opt/miniforge/envs/casa6/bin/python scripts/plot_lightcurves.py --candidates-only
```

**Step 6: Commit results (no code change, only products)**

```bash
git add TODOS.md   # update Phase 1 checkbox for dec-strip work
git commit -m "chore: re-run Feb15 with dec-strip guard; update TODOS"
```

---

### Task 5: Update TODOS.md

**Files:**
- Modify: `TODOS.md`

Mark the dec-strip work done and add the per-strip cal table task.

```markdown
<!-- Add to Phase 1, after existing checkboxes: -->
- [x] **Dec-strip guard**: `batch_pipeline.py` now detects observed Dec, aborts if it
  mismatches the cal-table strip by > 5°. `inventory.py` reports Dec per epoch.
  Feb 23 (Dec=+33°) and Feb 26 (Dec=+54°) correctly rejected.

<!-- Add to Phase 2: -->
- [ ] **Per-strip calibration tables**: derive B/G solutions for Dec=+33° and +54°
  strips. Run `batch_pipeline.py --expected-dec 33.0` once tables exist. Until then
  these strips are blocked by the dec-strip guard.
```

**Step 1: Edit TODOS.md** with the additions above.

**Step 2: Commit**

```bash
git add TODOS.md
git commit -m "docs(todos): mark dec-strip guard complete, add per-strip cal table task"
```

---

## Execution options

Plan complete and saved to `docs/plans/2026-03-04-dec-strip-awareness-plan.md`.

**1. Subagent-Driven (this session)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Parallel Session (separate)** — open a new session with executing-plans, batch execution with checkpoints.

Which approach?
