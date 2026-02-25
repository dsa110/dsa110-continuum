# pylint: disable=no-member  # astropy.units uses dynamic attributes (deg, etc.)
r"""
MODEL_DATA calculation for interferometric calibration.

This module provides functions to populate the MODEL_DATA column in CASA
Measurement Sets using manual calculation that correctly handles per-field
phase centers.

Notes
-----
**Coordinate Conventions**

This implementation follows standard interferometry conventions as defined in:

- Thompson, Moran & Swenson (2017), "Interferometry and Synthesis in Radio
  Astronomy", 3rd ed.
- CASA MeasurementSet v2 Specification (Note 264)

**UVW Coordinates:**

- u-axis: Points East (increasing RA) in array plane
- v-axis: Points North (increasing Dec) in array plane
- w-axis: Points toward phase center (line of sight)
- Units: meters
- Convention: CASA/MeasurementSet standard (right-handed)

**Phase Calculation:**

.. math::

    \phi = 2\pi(ul + vm + w(n-1)) / \lambda

Where:

- l = (α_source - α_pc) × cos(δ_pc)  [direction cosine, small angle approx]
- m = (δ_source - δ_pc)  [direction cosine, small angle approx]
- n - 1 ≈ -(l² + m²)/2 ≈ 0  [w-term, omitted for small FOV]
- λ = c/ν  [wavelength in meters, c = 299792458 m/s exact]

**Sign Convention:**

- Positive phase: exp(+2πi(ul + vm)/λ)
- Matches standard FT convention in radio interferometry
- Positive RA offset (East) → positive phase for positive u baseline
- Positive Dec offset (North) → positive phase for positive v baseline

**Phase Centers:**

- Uses per-field PHASE_DIR from FIELD table (critical!)
- PHASE_DIR updated by CASA phaseshift task for meridian tracking
- Ensures MODEL_DATA phase structure matches DATA column
- Falls back to REFERENCE_DIR if PHASE_DIR unavailable

**Polarization:**

- Assumes unpolarized point source (Stokes I only)
- Same visibility broadcast to all polarization correlations
- Not valid for polarized sources (Q, U, V ≠ 0)

See Also
--------
docs/reference/coordinates.md : Detailed coordinate documentation
backend/tests/unit/calibration/test_model_calculation.py : Validation tests
"""

import logging
import os
import time
from pathlib import Path

# IMPORTANT: Import casa_init BEFORE casacore to ensure CWD is set to logs directory
from dsa110_contimg.common.utils.casa_init import ensure_casa_path

ensure_casa_path()

import astropy.units as u  # noqa: E402
import casacore.tables as tb  # noqa: E402
import numpy as np  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402

# Set up logger
logger = logging.getLogger(__name__)

# Import cached MS metadata helper
try:
    from dsa110_contimg.common.utils.ms_helpers import get_ms_metadata
except ImportError:
    # Fallback if helper not available
    get_ms_metadata = None


def _ensure_imaging_columns(ms_path: str) -> None:
    """Ensure imaging columns (MODEL_DATA, CORRECTED_DATA) exist in MS.

    Uses CASA's clearcal() to create imaging columns, NOT casacore's addImagingColumns.
    This is critical because casacore creates columns with different storage format
    that CASA cannot read properly (column index mismatch, TSM incompatibility).

    IMPORTANT: This function now returns immediately if columns already exist,
    to avoid modifying the MS during concurrent operations.

    SAFETY: If CORRECTED_DATA contains applied calibration, this function will
    NOT run clearcal (which would destroy the calibration). Instead, it will
    raise CalibrationProtectionError.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set

    Raises
    ------
    CalibrationProtectionError
        If clearcal would destroy valid calibration in CORRECTED_DATA.
    """
    # Check if columns already exist using casacore (faster check, same result for column names)
    # We use readonly mode to avoid any locks
    try:
        import casacore.tables as ct

        with ct.table(ms_path, readonly=True) as t:
            cols = set(t.colnames())
            has_model = "MODEL_DATA" in cols
            has_corrected = "CORRECTED_DATA" in cols

        if has_model and has_corrected:
            logger.debug(f"Imaging columns already exist in {ms_path}")
            return
    except Exception as e:
        logger.debug(f"Could not check columns with casacore: {e}")
        # If we can't even check, columns probably don't exist - continue to create

    # Double-check with casatools for consistency (optional, for debugging)
    try:
        from casatools import table as tb_tool

        tb = tb_tool()
        tb.open(ms_path, nomodify=True)
        cols = set(tb.colnames())
        tb.close()

        if "MODEL_DATA" in cols and "CORRECTED_DATA" in cols:
            logger.debug(f"Imaging columns confirmed via casatools in {ms_path}")
            return
    except Exception as e:
        logger.debug(f"Could not check columns with casatools: {e}")

    # Avoid running clearcal when the MS directory is incomplete (e.g., test fixtures)
    ms_dir = Path(ms_path)
    table_dat_path = ms_dir / "table.dat"
    if not table_dat_path.exists():
        logger.debug(
            "Skipping clearcal for %s: table.dat missing (likely test fixture)",
            ms_path,
        )
        return

    # Use CASA's clearcal to create the columns - this ensures CASA compatibility
    # The CASAService.clearcal method has protect_calibration=True by default,
    # which will raise CalibrationProtectionError if CORRECTED_DATA contains
    # valid calibration that would be destroyed.
    try:
        from dsa110_contimg.core.calibration.casa_service import (
            CalibrationProtectionError,
            CASAService,
        )

        service = CASAService()

        logger.info(f"Creating imaging columns using clearcal: {ms_path}")
        # protect_calibration=True by default - will raise if calibration would be lost
        service.clearcal(vis=ms_path, addmodel=True)
        logger.debug(f"Imaging columns created via clearcal for {ms_path}")
    except CalibrationProtectionError:
        # Re-raise with clear message about what to do
        raise
    except Exception as e:
        # DO NOT fall back to casacore's addImagingColumns - it creates incompatible columns
        # and modifies existing columns (adding CHANNEL_SELECTION keyword) which can
        # corrupt the MS during concurrent access
        logger.warning(
            f"clearcal failed for {ms_path}: {e}. "
            "Not falling back to casacore to avoid compatibility issues."
        )
        # Raise so caller knows columns weren't created
        raise RuntimeError(
            f"Failed to create imaging columns with clearcal: {e}. "
            "This may indicate the MS is locked or corrupted."
        ) from e


def _initialize_corrected_from_data(ms_path: str) -> None:
    """Initialize CORRECTED_DATA column from DATA column.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    """
    import os
    import sys
    import time

    try:
        with tb.table(ms_path, readonly=False) as t:
            if "DATA" in t.colnames() and "CORRECTED_DATA" in t.colnames():
                nrows = t.nrows()
                if nrows <= 0:
                    return

                chunk_rows_env = os.getenv("CONTIMG_CORRECTED_CHUNK_ROWS", "")
                try:
                    chunk_rows = int(chunk_rows_env) if chunk_rows_env else 50000
                except ValueError:
                    chunk_rows = 50000
                if chunk_rows <= 0:
                    chunk_rows = nrows

                logger.info(
                    "  Initializing CORRECTED_DATA from DATA (%s rows, chunk_rows=%s)...",
                    f"{nrows:,}",
                    f"{chunk_rows:,}",
                )
                print(
                    f"  Initializing CORRECTED_DATA from DATA ({nrows:,} rows, "
                    f"chunk_rows={chunk_rows:,})...",
                    flush=True,
                )
                sys.stdout.flush()

                t0 = time.time()
                read_time = 0.0
                write_time = 0.0
                rows_copied = 0
                progress_rows = max(chunk_rows * 4, chunk_rows)

                for start in range(0, nrows, chunk_rows):
                    n = min(chunk_rows, nrows - start)
                    read_start = time.time()
                    data = t.getcol("DATA", start, n)
                    read_end = time.time()
                    t.putcol("CORRECTED_DATA", data, start, n)
                    write_end = time.time()

                    read_time += read_end - read_start
                    write_time += write_end - read_end
                    rows_copied += n

                    if rows_copied == nrows or rows_copied % progress_rows == 0:
                        elapsed = time.time() - t0
                        print(
                            f"    ... copied {rows_copied:,}/{nrows:,} rows ({elapsed:.1f}s)",
                            flush=True,
                        )
                        sys.stdout.flush()

                total_time = time.time() - t0
                logger.info(
                    "  CORRECTED_DATA initialized (%.1fs total; read %.1fs, write %.1fs)",
                    total_time,
                    read_time,
                    write_time,
                )
                print(
                    f"  CORRECTED_DATA initialized ({total_time:.1f}s total; "
                    f"read {read_time:.1f}s, write {write_time:.1f}s)",
                    flush=True,
                )
    except Exception as e:
        logger.debug(f"Could not initialize CORRECTED_DATA from DATA in {ms_path}: {e}")
        # Non-fatal, continue


def _calculate_manual_model_data(
    ms_path: str,
    ra_deg: float,
    dec_deg: float,
    flux_jy: float,
    field: str | None = None,
    initialize_corrected: bool = True,
) -> None:
    """Manually calculate MODEL_DATA phase structure using correct phase center.

    This function calculates MODEL_DATA directly using the formula:
        phase = 2π * (u*ΔRA + v*ΔDec) / λ

    This bypasses ft() which may use incorrect phase center information.

    **CRITICAL**: Uses each field's own PHASE_DIR (falls back to REFERENCE_DIR if unavailable)
    to ensure correct phase structure. PHASE_DIR matches the DATA column phasing (updated by
    phaseshift), ensuring MODEL_DATA phase structure matches DATA column exactly.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    ra_deg :
        Right ascension in degrees (component position)
    dec_deg :
        Declination in degrees (component position)
    flux_jy :
        Flux in Jy
    field :
        Optional field selection (default: all fields). Can be:
        - Single field index: "0"
        - Field range: "0~15"
        - Field name: "MyField"
        If None, writes to all fields.
    initialize_corrected :
        If True, copy DATA into CORRECTED_DATA after MODEL_DATA.
    """
    import sys

    import casacore.tables as casatables

    print("  Starting _calculate_manual_model_data...")
    sys.stdout.flush()

    casa_table = casatables.table  # noqa: N816

    _ensure_imaging_columns(ms_path)
    print("    Imaging columns ensured")
    sys.stdout.flush()

    # Parse field selection to get list of field indices
    field_indices = None
    if field is not None:
        if "~" in str(field):
            # Field range: "0~15"
            try:
                parts = str(field).split("~")
                start_idx = int(parts[0])
                end_idx = int(parts[1])
                field_indices = list(range(start_idx, end_idx + 1))
            except (ValueError, IndexError):
                field_indices = None
        elif field.isdigit():
            # Single field index: "0"
            field_indices = [int(field)]
        # If field is a name or invalid, field_indices stays None (use all fields)

    # OPTIMIZATION: Use cached MS metadata if available to avoid redundant table reads
    # This is especially beneficial when MODEL_DATA is calculated multiple times
    # for the same MS (e.g., during calibration iteration).
    use_cached_metadata = False
    if get_ms_metadata is not None:
        try:
            metadata = get_ms_metadata(ms_path)
            phase_dir = metadata.get("phase_dir")
            chan_freq = metadata.get("chan_freq")
            if phase_dir is not None and chan_freq is not None:
                nfields = len(phase_dir)
                nspw = len(chan_freq)
                # Check if cached metadata is actually valid (non-empty)
                if nfields > 0 and nspw > 0:
                    use_cached_metadata = True
                    logger.debug(
                        f"Using cached MS metadata for {ms_path} ({nfields} fields, {nspw} SPWs)"
                    )
                else:
                    # Cached metadata is empty/invalid, fall back to direct read
                    raise ValueError("Cached metadata incomplete")
            else:
                # Fallback to direct read if cache doesn't have required fields
                raise ValueError("Cached metadata incomplete")
        except Exception as e:
            # Fallback to direct read if cache fails
            logger.debug(
                f"Metadata cache lookup failed for {ms_path}: {e}. Falling back to direct read."
            )
            use_cached_metadata = False

    if not use_cached_metadata:
        # Fallback: Read MS phase center from PHASE_DIR for all fields
        # PHASE_DIR matches the actual phase center used for DATA column phasing
        # (updated by phaseshift). This ensures MODEL_DATA matches DATA column phase structure.
        logger.debug(f"Reading MS metadata directly from tables for {ms_path}")
        with casa_table(f"{ms_path}::FIELD", readonly=True) as field_tb:
            if "PHASE_DIR" in field_tb.colnames():
                phase_dir = field_tb.getcol("PHASE_DIR")  # Shape: (nfields, 1, 2)
                logger.debug("Using PHASE_DIR for phase centers")
            else:
                # Fallback to REFERENCE_DIR if PHASE_DIR not available
                phase_dir = field_tb.getcol("REFERENCE_DIR")  # Shape: (nfields, 1, 2)
                logger.debug("PHASE_DIR not available, using REFERENCE_DIR")
            nfields = len(phase_dir)

        # Read spectral window information for frequencies
        with casa_table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True) as spw_tb:
            chan_freq = spw_tb.getcol("CHAN_FREQ")  # Shape: (nspw, nchan)
            nspw = len(chan_freq)

    # Log field selection
    if field_indices is not None:
        logger.debug(f"Field selection: {field_indices} ({len(field_indices)} fields)")
    else:
        logger.debug("No field selection: processing all fields")

    # Read main table data
    start_time = time.time()
    with casa_table(ms_path, readonly=False) as main_tb:
        nrows = main_tb.nrows()
        logger.info(
            f"Calculating MODEL_DATA for {ms_path} (field={field}, flux={flux_jy:.2f} Jy, {nrows:,} rows)"
        )
        # Progress output for background processes
        import sys

        print(f"  Reading {nrows:,} rows from MS...")
        sys.stdout.flush()

        # Read UVW coordinates
        uvw = main_tb.getcol("UVW")  # Shape: (nrows, 3)
        u = uvw[:, 0]
        v = uvw[:, 1]
        print(f"    UVW read ({uvw.nbytes / 1e6:.1f} MB)")
        sys.stdout.flush()

        # Read DATA_DESC_ID and map to SPECTRAL_WINDOW_ID
        # DATA_DESC_ID indexes the DATA_DESCRIPTION table, not SPECTRAL_WINDOW directly
        data_desc_id = main_tb.getcol("DATA_DESC_ID")  # Shape: (nrows,)

        # Read DATA_DESCRIPTION table to get SPECTRAL_WINDOW_ID mapping
        with casa_table(f"{ms_path}::DATA_DESCRIPTION", readonly=True) as dd_tb:
            dd_spw_id = dd_tb.getcol("SPECTRAL_WINDOW_ID")  # Shape: (ndd,)
            # Map DATA_DESC_ID -> SPECTRAL_WINDOW_ID
            spw_id = dd_spw_id[data_desc_id]  # Shape: (nrows,)

        # Read FIELD_ID to apply field selection and get per-field phase centers
        field_id = main_tb.getcol("FIELD_ID")  # Shape: (nrows,)
        print("    Metadata columns read")
        sys.stdout.flush()

        # Apply field selection if specified
        if field_indices is not None:
            field_mask = np.isin(field_id, field_indices)
        else:
            field_mask = np.ones(nrows, dtype=bool)

        nselected = np.sum(field_mask)
        logger.debug(f"Processing {nselected:,} rows ({nselected / nrows * 100:.1f}% of total)")
        print(f"  Processing {nselected:,} selected rows...")
        sys.stdout.flush()

        # Read DATA shape to create MODEL_DATA with matching shape
        print("    Reading DATA shape...")
        sys.stdout.flush()
        data_sample = main_tb.getcell("DATA", 0)
        data_shape = data_sample.shape  # In CASA: (nchan, npol)
        nchan, npol = data_shape[0], data_shape[1]
        logger.debug(f"Data shape: {nchan} channels, {npol} polarizations")
        print(f"    DATA shape: {nchan} channels × {npol} pols")
        sys.stdout.flush()

        # Initialize MODEL_DATA array with correct shape (nrows, nchan, npol)
        print(f"    Allocating MODEL_DATA array ({nrows * nchan * npol * 8 / 1e9:.2f} GB)...")
        sys.stdout.flush()
        model_data = np.zeros((nrows, nchan, npol), dtype=np.complex64)
        logger.debug(f"Allocated MODEL_DATA array: {model_data.nbytes / 1e9:.2f} GB")
        print("    MODEL_DATA allocated")
        sys.stdout.flush()

        # CHUNKED VECTORIZED CALCULATION: Process rows in chunks to limit memory
        # This replaces the row-by-row loop while avoiding memory exhaustion
        CALC_CHUNK_SIZE = 200000  # Process 200k rows at a time (~150MB per chunk)
        print("    Starting chunked MODEL_DATA calculation...")
        sys.stdout.flush()

        # Filter to selected rows only
        selected_indices = np.where(field_mask)[0]
        if len(selected_indices) == 0:
            logger.warning("No rows match field selection criteria")
            main_tb.putcol("MODEL_DATA", model_data)
            main_tb.flush()
            return

        # Get field and SPW indices for selected rows
        selected_field_id = field_id[selected_indices]  # (nselected,)
        selected_spw_id = spw_id[selected_indices]  # (nselected,)
        selected_u = u[selected_indices]  # (nselected,)
        selected_v = v[selected_indices]  # (nselected,)

        # Validate field and SPW indices
        valid_field_mask = (selected_field_id >= 0) & (selected_field_id < nfields)
        valid_spw_mask = (selected_spw_id >= 0) & (selected_spw_id < nspw)
        valid_mask = valid_field_mask & valid_spw_mask

        if not np.all(valid_mask):
            n_invalid = np.sum(~valid_mask)
            logger.warning(f"Skipping {n_invalid} rows with invalid field/SPW indices")
            selected_indices = selected_indices[valid_mask]
            selected_field_id = selected_field_id[valid_mask]
            selected_spw_id = selected_spw_id[valid_mask]
            selected_u = selected_u[valid_mask]
            selected_v = selected_v[valid_mask]

        nselected = len(selected_indices)
        if nselected == 0:
            logger.warning("No valid rows after filtering")
            main_tb.putcol("MODEL_DATA", model_data)
            main_tb.flush()
            return

        print(
            f"    Calculating MODEL_DATA for {nselected:,} rows in chunks of {CALC_CHUNK_SIZE:,}..."
        )
        sys.stdout.flush()

        # Pre-compute constants
        amplitude = float(flux_jy)
        SPEED_OF_LIGHT_M_S = 299792458.0  # Exact by definition

        # Process in chunks to limit memory usage
        calc_start = time.time()
        n_chunks = (nselected + CALC_CHUNK_SIZE - 1) // CALC_CHUNK_SIZE
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * CALC_CHUNK_SIZE
            chunk_end = min((chunk_idx + 1) * CALC_CHUNK_SIZE, nselected)

            # Get chunk slices
            chunk_indices = selected_indices[chunk_start:chunk_end]
            chunk_field_id = selected_field_id[chunk_start:chunk_end]
            chunk_spw_id = selected_spw_id[chunk_start:chunk_end]
            chunk_u = selected_u[chunk_start:chunk_end]
            chunk_v = selected_v[chunk_start:chunk_end]

            # Get phase centers for chunk rows
            phase_centers_ra_rad = phase_dir[chunk_field_id, 0, 0]
            phase_centers_dec_rad = phase_dir[chunk_field_id, 0, 1]

            # Convert to degrees
            phase_centers_ra_deg = np.degrees(phase_centers_ra_rad)
            phase_centers_dec_deg = np.degrees(phase_centers_dec_rad)

            # Calculate offsets from phase centers to component
            # CRITICAL: Handle RA wrap-around (e.g., -16.5° vs 343.5° are the same)
            # Normalize RA difference to [-180°, +180°] before calculating offset
            ra_diff_deg = ra_deg - phase_centers_ra_deg
            # Wrap to [-180, +180] range
            ra_diff_deg = ((ra_diff_deg + 180.0) % 360.0) - 180.0
            offset_ra_rad = np.radians(ra_diff_deg) * np.cos(phase_centers_dec_rad)
            offset_dec_rad = np.radians(dec_deg - phase_centers_dec_deg)

            # Get frequencies for chunk rows
            chunk_freqs = chan_freq[chunk_spw_id]  # (chunk_size, nchan)
            chunk_wavelengths = SPEED_OF_LIGHT_M_S / chunk_freqs  # (chunk_size, nchan)

            # Vectorize phase calculation using broadcasting
            u_broadcast = chunk_u[:, np.newaxis]  # (chunk_size, 1)
            v_broadcast = chunk_v[:, np.newaxis]  # (chunk_size, 1)
            offset_ra_broadcast = offset_ra_rad[:, np.newaxis]  # (chunk_size, 1)
            offset_dec_broadcast = offset_dec_rad[:, np.newaxis]  # (chunk_size, 1)

            # Phase calculation: 2π * (u*ΔRA + v*ΔDec) / λ
            phase = (
                2
                * np.pi
                * (u_broadcast * offset_ra_broadcast + v_broadcast * offset_dec_broadcast)
                / chunk_wavelengths
            )
            phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi  # Wrap to [-π, π]

            # Create complex model: amplitude * exp(i*phase)
            model_complex = amplitude * (np.cos(phase) + 1j * np.sin(phase))

            # Broadcast to all polarizations and store in output array
            model_data[chunk_indices, :, :] = model_complex[:, :, np.newaxis]

            # Progress output
            if (chunk_idx + 1) % 3 == 0 or chunk_idx == n_chunks - 1:
                elapsed = time.time() - calc_start
                pct = (chunk_end / nselected) * 100
                print(
                    f"      Chunk {chunk_idx + 1}/{n_chunks}: {chunk_end:,}/{nselected:,} rows ({pct:.0f}%, {elapsed:.1f}s)"
                )
                sys.stdout.flush()

        calc_time = time.time() - start_time
        logger.info(
            f"MODEL_DATA calculation completed in {calc_time:.2f}s ({nselected:,} rows, {calc_time / nselected * 1e6:.2f} μs/row)"
        )
        # Flush stdout for visibility in background processes
        import sys

        print(f"  MODEL_DATA calc done ({calc_time:.1f}s), writing {nselected:,} rows...")
        sys.stdout.flush()

        # Write MODEL_DATA column in chunks to avoid memory issues and show progress
        write_start = time.time()
        CHUNK_SIZE = 100000  # Write 100k rows at a time
        for chunk_start in range(0, nrows, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, nrows)
            main_tb.putcol(
                "MODEL_DATA",
                model_data[chunk_start:chunk_end],
                chunk_start,
                chunk_end - chunk_start,
            )
            if chunk_start % 500000 == 0 and chunk_start > 0:
                elapsed = time.time() - write_start
                print(f"    ... written {chunk_start:,}/{nrows:,} rows ({elapsed:.1f}s)")
                sys.stdout.flush()
        main_tb.flush()  # Ensure data is written to disk
        write_time = time.time() - write_start
        logger.debug(f"MODEL_DATA written to disk in {write_time:.2f}s")
        print(f"  MODEL_DATA write done ({write_time:.1f}s)")
        sys.stdout.flush()

        total_time = time.time() - start_time
        logger.info(f":check: MODEL_DATA populated for {ms_path} (total: {total_time:.2f}s)")

    if initialize_corrected:
        _initialize_corrected_from_data(ms_path)


def write_point_model_with_ft(
    ms_path: str,
    ra_deg: float,
    dec_deg: float,
    flux_jy: float,
    *,
    reffreq_hz: float = 1.4e9,
    spectral_index: float | None = None,
    field: str | None = None,
    use_manual: bool = True,
    initialize_corrected: bool = True,
) -> None:
    """Write a physically-correct complex point-source model into MODEL_DATA.

    By default, uses manual calculation which handles per-field phase centers correctly.
    If use_manual=False, uses CASA ft() task, which reads phase center from FIELD parameters
    but uses ONE phase center for ALL fields. This causes phase errors when fields have
    different phase centers (e.g., each field phased to its own meridian). ft() works correctly
    when all fields share the same phase center (after rephasing), but manual calculation
    is more robust and handles per-field phase centers correctly in all scenarios.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    ra_deg :
        Right ascension in degrees
    dec_deg :
        Declination in degrees
    flux_jy :
        Flux in Jy
    reffreq_hz :
        Reference frequency in Hz (default: 1.4 GHz)
    spectral_index :
        Optional spectral index for frequency-dependent flux
    field :
        Optional field selection (default: all fields). If specified, MODEL_DATA
        will only be written to the selected field(s).
    use_manual :
        If True (default), use manual calculation (recommended).
        If False, use ft() which uses one phase center for all fields.
        Use False only when all fields share the same phase center.
    initialize_corrected :
        If True, copy DATA into CORRECTED_DATA after MODEL_DATA.
    """
    if use_manual:
        # Use manual calculation to bypass ft() phase center issues
        logger.info(
            "Writing point model using manual calculation (bypasses ft() phase center issues)"
        )
        _calculate_manual_model_data(
            ms_path,
            ra_deg,
            dec_deg,
            flux_jy,
            field=field,
            initialize_corrected=initialize_corrected,
        )
        return

    from dsa110_contimg.core.calibration.casa_service import CASAService

    service = CASAService()

    try:
        from casatools import componentlist as cltool
    except ImportError:
        from casatools import componentlist as cltool

    logger.info(
        "Writing point model using ft() (use_manual=False). "
        "WARNING: ft() uses one phase center for all fields. "
        "Use use_manual=True for per-field phase centers."
    )
    _ensure_imaging_columns(ms_path)

    comp_path = os.path.join(os.path.dirname(ms_path), "cal_component.cl")
    # Remove existing component list if it exists (cl.rename() will fail if it exists)
    if os.path.exists(comp_path):
        import shutil

        shutil.rmtree(comp_path, ignore_errors=True)
    cl = cltool()
    sc = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    dir_dict = {
        "refer": "J2000",
        "type": "direction",
        "long": f"{sc.ra.deg}deg",
        "lat": f"{sc.dec.deg}deg",
    }
    cl.addcomponent(
        dir=dir_dict,
        flux=float(flux_jy),
        fluxunit="Jy",
        freq=f"{reffreq_hz}Hz",
        shape="point",
    )
    if spectral_index is not None:
        try:
            cl.setspectrum(
                which=0,
                type="spectral index",
                index=[float(spectral_index)],
            )
            cl.setfreq(which=0, value=reffreq_hz, unit="Hz")
        except (RuntimeError, ValueError):
            pass
    cl.rename(comp_path)
    cl.close()

    # CRITICAL: Explicitly clear MODEL_DATA with zeros before calling ft()
    # This ensures MODEL_DATA is properly cleared; clearcal() may not fully
    # clear MODEL_DATA, especially after rephasing
    try:
        import numpy as np

        t = tb.table(ms_path, readonly=False)
        if "MODEL_DATA" in t.colnames() and t.nrows() > 0:
            # Get DATA shape to match MODEL_DATA shape
            if "DATA" in t.colnames():
                data_sample = t.getcell("DATA", 0)
                data_shape = getattr(data_sample, "shape", None)
                data_dtype = getattr(data_sample, "dtype", None)
                if data_shape and data_dtype:
                    # Clear MODEL_DATA with zeros matching DATA shape
                    zeros = np.zeros((t.nrows(),) + data_shape, dtype=data_dtype)
                    t.putcol("MODEL_DATA", zeros)
        t.close()
    except Exception as e:
        # Non-fatal: log warning but continue
        import warnings

        warnings.warn(
            f"Failed to explicitly clear MODEL_DATA before ft(): {e}. "
            "Continuing with ft() call, but MODEL_DATA may not be properly cleared.",
            RuntimeWarning,
        )

    # Pass field parameter to ensure MODEL_DATA is written to the correct field
    # NOTE: ft() reads phase center from FIELD parameters, but uses ONE phase center for ALL fields.
    # If fields have different phase centers (e.g., each field phased to its own meridian),
    # ft() will use the phase center from one field (typically field 0) for all fields,
    # causing phase errors for fields with different phase centers.
    # Manual calculation (use_manual=True) handles per-field phase centers correctly.
    ft_kwargs = {"vis": ms_path, "complist": comp_path, "usescratch": True}
    if field is not None:
        ft_kwargs["field"] = field
    service.ft(**ft_kwargs)
    if initialize_corrected:
        _initialize_corrected_from_data(ms_path)


# NOTE: write_point_model_quick() has been archived to archive/legacy/calibration/model_quick.py
# This function was testing-only and not used in production. It did not calculate
# phase structure (amplitude-only), making it unsuitable for calibration workflows.
# Use write_point_model_with_ft(use_manual=True) instead.


def write_image_model_with_ft(ms_path: str, image_path: str) -> None:
    """Apply a CASA image model into MODEL_DATA using ft.

    This function uses casatasks.ft() to predict visibilities from a model image.

    **Phase Center Behavior**:
    CASA's ft() uses REFERENCE_DIR (not PHASE_DIR) from the FIELD table.
    After phaseshift(), only PHASE_DIR is updated. This function automatically
    syncs REFERENCE_DIR with PHASE_DIR before calling ft() to ensure correct
    phase center handling.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    image_path :
        Path to CASA image model
    """
    from dsa110_contimg.core.calibration.casa_service import CASAService
    from dsa110_contimg.core.calibration.runner import sync_reference_dir_with_phase_dir

    service = CASAService()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Model image not found: {image_path}")

    _ensure_imaging_columns(ms_path)

    # Sync REFERENCE_DIR with PHASE_DIR so ft() uses correct phase center
    # CASA's ft() reads REFERENCE_DIR (not PHASE_DIR) for model visibility computation
    sync_reference_dir_with_phase_dir(ms_path)

    # usescratch=True ensures MODEL_DATA is populated/created
    service.ft(vis=ms_path, model=image_path, usescratch=True)
    _initialize_corrected_from_data(ms_path)


def export_model_as_fits(
    ms_path: str,
    output_path: str,
    field: str | None = None,
    imsize: int = 512,
    cell_arcsec: float = 1.0,
) -> None:
    """Export MODEL_DATA as a FITS image using WSClean.

    Creates a dirty image from MODEL_DATA column using WSClean and exports it to FITS.
    This is useful for visualizing the sky model used during calibration (NVSS sources
    or calibrator model) and for debugging calibration issues.

    Uses WSClean instead of CASA tclean for consistency with the main imaging pipeline
    and to avoid CASA warnings about unknown telescope positions.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    output_path :
        Output FITS file path (without .fits extension)
    field :
        Optional field selection (default: all fields)
    imsize :
        Image size in pixels (default: 512)
    cell_arcsec :
        Cell size in arcseconds (default: 1.0)
    """
    import logging
    import shutil
    import subprocess

    LOG = logging.getLogger(__name__)

    # Ensure imaging columns exist
    _ensure_imaging_columns(ms_path)

    try:
        from dsa110_contimg.common.utils.ms_permissions import (
            ensure_dir_writable,
            ensure_ms_writable,
        )

        ensure_ms_writable(ms_path)
        ensure_dir_writable(Path(output_path).parent)

        # Build WSClean command to create dirty image from MODEL_DATA
        # Using Docker WSClean for consistency with main imaging pipeline
        from dsa110_contimg.common.utils.gpu_utils import build_docker_command, get_gpu_config

        gpu_config = get_gpu_config()

        docker_base = build_docker_command(
            image="dsa110-contimg:gpu",
            command=["wsclean"],
            gpu_config=gpu_config,
            env_vars={
                "NVIDIA_DISABLE_REQUIRE": "1",
                "MAMBA_ROOT_PREFIX": "/dev/shm/micromamba",
                "HOME": "/dev/shm/dsa110-contimg",
            },
        )

        cmd = docker_base.copy()
        cmd.extend(["-name", output_path])
        cmd.extend(["-size", str(imsize), str(imsize)])
        cmd.extend(["-scale", f"{cell_arcsec:.3f}arcsec"])
        cmd.extend(["-data-column", "MODEL_DATA"])
        cmd.extend(["-niter", "0"])  # Dirty image only
        cmd.extend(["-weight", "natural"])
        cmd.extend(["-pol", "I"])
        cmd.extend(["-reorder"])  # Required for multi-SPW MS files
        cmd.extend(["-gridder", "wgridder"])  # CPU gridder - fast enough for dirty imaging
        cmd.extend(["-j", "4"])  # Limit threads
        cmd.extend(["-mem", "20"])  # Limit memory to 20% of system RAM

        if field is not None:
            cmd.extend(["-field", field])

        cmd.append(ms_path)

        LOG.info(f"Creating model image from {ms_path} MODEL_DATA using WSClean...")
        LOG.debug("WSClean command: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for dirty image
        )

        if result.returncode != 0:
            LOG.error("WSClean failed: %s", result.stderr)
            raise RuntimeError(f"WSClean failed with exit code {result.returncode}")

        # WSClean creates output_path-dirty.fits for niter=0
        dirty_fits = f"{output_path}-dirty.fits"
        final_fits = f"{output_path}.fits"

        if os.path.exists(dirty_fits):
            shutil.move(dirty_fits, final_fits)
            LOG.info(f":check: Model image exported to {final_fits}")
        else:
            # Check for MFS output
            mfs_dirty = f"{output_path}-MFS-dirty.fits"
            if os.path.exists(mfs_dirty):
                shutil.move(mfs_dirty, final_fits)
                LOG.info(f":check: Model image exported to {final_fits}")
            else:
                raise FileNotFoundError(
                    f"WSClean did not produce expected output: {dirty_fits} or {mfs_dirty}"
                )

    except ImportError as e:
        # DISABLED: tclean fallback - we want to know if WSClean fails
        # To re-enable, set environment variable DSA110_ALLOW_TCLEAN_FALLBACK=1
        if os.environ.get("DSA110_ALLOW_TCLEAN_FALLBACK", "0") == "1":
            LOG.warning(f"Docker utilities not available ({e}), falling back to tclean")
            _export_model_as_fits_tclean(ms_path, output_path, field, imsize, cell_arcsec)
        else:
            raise ImportError(
                f"WSClean/Docker utilities not available: {e}. "
                "Set DSA110_ALLOW_TCLEAN_FALLBACK=1 to enable tclean fallback."
            ) from e
    except Exception as e:
        LOG.error(f"Failed to export model image: {e}")
        raise


def _export_model_as_fits_tclean(
    ms_path: str,
    output_path: str,
    field: str | None = None,
    imsize: int = 512,
    cell_arcsec: float = 1.0,
) -> None:
    """Fallback: Export MODEL_DATA as FITS using CASA tclean.

    Used when WSClean/Docker is not available.

    Parameters
    ----------
    """
    import logging

    from dsa110_contimg.core.calibration.casa_service import CASAService

    service = CASAService()

    LOG = logging.getLogger(__name__)

    image_name = f"{output_path}.model"

    tclean_kwargs = {
        "vis": ms_path,
        "imagename": image_name,
        "datacolumn": "model",
        "imsize": [imsize, imsize],
        "cell": [f"{cell_arcsec}arcsec", f"{cell_arcsec}arcsec"],
        "specmode": "mfs",
        "niter": 0,
        "weighting": "natural",
        "stokes": "I",
    }
    if field is not None:
        tclean_kwargs["field"] = field

    LOG.info(f"Creating model image from {ms_path} MODEL_DATA (tclean fallback)...")
    service.tclean(**tclean_kwargs)

    fits_path = f"{output_path}.fits"
    service.exportfits(imagename=f"{image_name}.image", fitsimage=fits_path, overwrite=True)
    LOG.info(f":check: Model image exported to {fits_path}")


def _get_calibrator_flux_from_catalog(catalog, calibrator_name: str) -> float:
    """Get calibrator flux from catalog DataFrame.

    Parameters
    ----------
    catalog :
        VLA calibrator catalog DataFrame with 'flux_jy' column
    calibrator_name :
        Calibrator name (e.g., "0834+555")

    Returns
    -------
        Flux in Jy

    Raises
    ------
    ValueError
        If calibrator not found or flux not available

    """
    import pandas as pd

    if calibrator_name not in catalog.index:
        # 1. Try case-insensitive search on index (names)
        key = calibrator_name.strip().upper()
        for idx in catalog.index:
            if str(idx).strip().upper() == key:
                calibrator_name = idx
                break
        else:
            # 2. Try searching in alt_name column
            if "alt_name" in catalog.columns:
                mask = catalog["alt_name"].str.strip().str.upper() == key
                if mask.any():
                    calibrator_name = catalog[mask].index[0]
                else:
                    raise ValueError(f"Calibrator '{calibrator_name}' not found in catalog")
            else:
                raise ValueError(f"Calibrator '{calibrator_name}' not found in catalog")

    row = catalog.loc[calibrator_name]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    if "flux_jy" not in row.index:
        raise ValueError(
            f"Catalog does not contain flux_jy column for '{calibrator_name}'. "
            "Provide explicit flux with cal_flux_jy parameter."
        )

    flux_jy = row["flux_jy"]
    if isinstance(flux_jy, pd.Series):
        flux_jy = flux_jy.iloc[0]

    flux_jy = float(flux_jy)
    if np.isnan(flux_jy) or flux_jy <= 0:
        raise ValueError(
            f"Calibrator '{calibrator_name}' has invalid flux ({flux_jy}) in catalog. "
            "Provide explicit flux with cal_flux_jy parameter."
        )

    return flux_jy


def populate_model_from_catalog(
    ms_path: str,
    *,
    field: str | None = None,
    calibrator_name: str | None = None,
    cal_ra_deg: float | None = None,
    cal_dec_deg: float | None = None,
    cal_flux_jy: float | None = None,
    use_unified_model: bool = True,
    radius_deg: float = 1.0,
    min_mjy: float = 2.0,
    initialize_corrected: bool = True,
    force_unified: bool = False,
) -> None:
    """Populate MODEL_DATA from catalog source.

    Looks up calibrator coordinates and flux from catalog, then writes
    MODEL_DATA using manual calculation (bypasses ft() phase center bugs).

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    field :
        Field selection (default: all fields in the MS)
    calibrator_name :
        Calibrator name (e.g., "0834+555"). If not provided,
        attempts to auto-detect from MS field names.
    cal_ra_deg :
        Optional explicit RA in degrees (overrides catalog lookup)
    cal_dec_deg :
        Optional explicit Dec in degrees (overrides catalog lookup)
    cal_flux_jy :
        Optional explicit flux in Jy. If not provided, looks up
        from catalog. Raises ValueError if flux unavailable.
    use_unified_model :
        If True, use multi-source unified model for non-bandpass
        fields (default: True).
    radius_deg :
        Radius for unified model search in degrees (default: 1.0).
    min_mjy :
        Minimum flux for unified model sources in mJy (default: 2.0).
    initialize_corrected :
        If True, copy DATA into CORRECTED_DATA after MODEL_DATA.

    Raises
    ------
    ValueError
        If calibrator cannot be found, coordinates are invalid,
        or flux is not available (neither in catalog nor explicit)
    RuntimeError
        If MODEL_DATA population fails

    """
    from dsa110_contimg.core.calibration.catalogs import (
        get_calibrator_radec,
        load_vla_catalog,
    )
    from dsa110_contimg.core.calibration.skymodels import (
        make_unified_skymodel,
        predict_from_skymodel_wsclean,
    )

    # Bandpass calibrators that should always use single-source models
    # These are standard flux calibrators where we want to enforce the VLA scale
    # Standard VLA bandpass/flux calibrators + bright quasars often used in DSA-110
    BANDPASS_CALIBRATORS = [
        "3C286",
        "1331+305",
        "3C48",
        "0137+331",
        "3C147",
        "0542+498",
        "3C138",
        "0521+166",
        "3C196",
        "0813+482",
        "3C295",
        "1411+522",
        "3C454.3",
        "2253+161",
        "3C273",
        "1229+020",
        "3C84",
        "0319+415",
    ]

    # Default to ALL fields if not provided (DSA-110 has 24 fields)
    # Previously defaulted to "0", which silently applied the model to only 1 of 24 fields.
    if field is None:
        with tb.table(f"{ms_path}::FIELD", readonly=True) as field_tb:
            nfields = field_tb.nrows()
        if nfields > 1:
            field = f"0~{nfields - 1}"
            logger.info(f"No field specified — defaulting to all {nfields} fields: '{field}'")
        else:
            field = "0"

    # Determine if we should use unified model
    # 1. Must be enabled (use_unified_model=True)
    # 2. Must NOT be a bandpass calibrator (unless explicit coordinates are used)
    # 3. Must NOT have explicit coordinates (unless we want to force unified model around them?)
    #    Actually, if explicit coords are given, we usually mean a specific point source.
    #    But the user might want unified model around that point.
    #    Let's stick to: if bandpass calibrator -> single source. Else -> unified model.

    is_bandpass = False
    if calibrator_name:
        # Check if this is a standard bandpass/flux calibrator
        # We check the provided name against our list
        is_bandpass = any(
            bp.lower() == calibrator_name.lower().strip() for bp in BANDPASS_CALIBRATORS
        )

        # If not found directly, check if the calibrator name exists as an alt_name in the catalog
        # and if its primary name is in the list (or vice versa)
        if not is_bandpass:
            try:
                catalog = load_vla_catalog()
                # Use our updated get_calibrator_radec logic (implicitly) or just search the DF
                if calibrator_name in catalog.index:
                    primary_name = str(calibrator_name)
                    alt_name = (
                        str(catalog.loc[primary_name, "alt_name"])
                        if "alt_name" in catalog.columns
                        else ""
                    )
                else:
                    # Try to find by alt_name
                    mask = (
                        catalog["alt_name"].str.strip().str.upper()
                        == calibrator_name.strip().upper()
                    )
                    if mask.any():
                        primary_name = str(catalog[mask].index[0])
                        alt_name = calibrator_name
                    else:
                        primary_name = calibrator_name
                        alt_name = ""

                is_bandpass = any(
                    bp.lower() in [primary_name.lower(), alt_name.lower()]
                    for bp in BANDPASS_CALIBRATORS
                )
            except Exception:
                pass

    if use_unified_model and (not is_bandpass or force_unified):
        # Use multi-source unified model
        logger.info(
            f"Using multi-source unified model for {calibrator_name or 'target'} "
            f"(radius={radius_deg}°, min_flux={min_mjy} mJy)"
        )

        # Determine center coordinates
        if cal_ra_deg is not None and cal_dec_deg is not None:
            center_ra = float(cal_ra_deg)
            center_dec = float(cal_dec_deg)
        elif calibrator_name:
            try:
                catalog = load_vla_catalog()
                center_ra, center_dec = get_calibrator_radec(catalog, calibrator_name)
            except Exception:
                # Fallback to MS field center if catalog lookup fails
                # (e.g. for non-standard sources)
                # We need to read the PHASE_DIR from the MS
                # But wait, if we are here, we probably want to look up the source position first.
                # If it's not in VLA catalog, maybe we should just use the field center?
                # Let's try to get it from MS if catalog fails.
                logger.info(
                    f"Calibrator {calibrator_name} not in VLA catalog, using MS field center"
                )
                with tb.table(f"{ms_path}::FIELD", readonly=True) as t:
                    # Assuming field 0 or selected field
                    # We need to handle field selection properly
                    # For now, just take the first selected field's phase dir
                    # This is a bit tricky if multiple fields are selected.
                    # But typically calibration is done on one source.
                    # Let's assume the phaseshift has put the source at the phase center.
                    # So we can use the phase center of the first field.
                    phase_dir = t.getcol("PHASE_DIR")
                    # field argument might be "0~23" or "0".
                    # We'll just take field 0 of the MS (since it's likely phaseshifted)
                    # Or parse field.
                    # Let's just use the first field in the MS for center.
                    ra_rad = phase_dir[0, 0, 0]
                    dec_rad = phase_dir[0, 0, 1]
                    center_ra = np.degrees(ra_rad)
                    center_dec = np.degrees(dec_rad)
        else:
            # No name, no coords -> use MS field center
            with tb.table(f"{ms_path}::FIELD", readonly=True) as t:
                phase_dir = t.getcol("PHASE_DIR")
                ra_rad = phase_dir[0, 0, 0]
                dec_rad = phase_dir[0, 0, 1]
                center_ra = np.degrees(ra_rad)
                center_dec = np.degrees(dec_rad)

        # Create unified model
        try:
            # Generate model
            sky = make_unified_skymodel(
                center_ra,
                center_dec,
                radius_deg,
                min_mjy=min_mjy,
            )

            if sky.Ncomponents > 0:
                logger.info(f"Generated unified model with {sky.Ncomponents} components")

                # Use WSClean -draw-model + -predict instead of ft()
                # This is faster, more reliable, and avoids ft() phase center bugs
                predict_from_skymodel_wsclean(
                    ms_path,
                    sky,
                    field=field,
                    cleanup=True,
                )

                logger.info(":check: MODEL_DATA populated from unified model using WSClean")
                return
            else:
                logger.warning(
                    "Unified model returned 0 components. Falling back to single source."
                )
        except Exception as e:
            logger.error(
                f"Failed to use unified model with WSClean: {e}. Falling back to single source."
            )
            # Fall through to single source logic

    # Determine calibrator coordinates and flux
    if cal_ra_deg is not None and cal_dec_deg is not None:
        # Use explicit coordinates - flux MUST be provided
        if cal_flux_jy is None:
            raise ValueError(
                "cal_flux_jy is required when using explicit coordinates. "
                "No default flux is used to prevent silent calibration errors."
            )
        ra_deg = float(cal_ra_deg)
        dec_deg = float(cal_dec_deg)
        flux_jy = float(cal_flux_jy)
        name = calibrator_name or f"manual_{ra_deg:.2f}_{dec_deg:.2f}"
        logger.info(
            f"Using explicit calibrator coordinates: {name} @ ({ra_deg:.4f}°, {dec_deg:.4f}°), {flux_jy:.2f} Jy"
        )
    elif calibrator_name:
        # Look up from catalog - get BOTH coordinates AND flux
        try:
            catalog = load_vla_catalog()
            ra_deg, dec_deg = get_calibrator_radec(catalog, calibrator_name)
            # Get flux from catalog or use explicit override
            if cal_flux_jy is not None:
                flux_jy = float(cal_flux_jy)
                logger.info(f"Using explicit flux override: {flux_jy:.2f} Jy")
            else:
                # Look up flux from catalog
                flux_jy = _get_calibrator_flux_from_catalog(catalog, calibrator_name)
            name = calibrator_name
            logger.info(
                f"Found calibrator in catalog: {name} @ ({ra_deg:.4f}°, {dec_deg:.4f}°), {flux_jy:.2f} Jy"
            )
        except Exception as e:
            raise ValueError(
                f"Could not find calibrator '{calibrator_name}' in catalog: {e}. "
                "Provide explicit coordinates with cal_ra_deg and cal_dec_deg."
            ) from e
    else:
        # Try to auto-detect from MS field names
        try:
            with tb.table(ms_path + "::FIELD", readonly=True) as field_tb:
                if "NAME" in field_tb.colnames() and field_tb.nrows() > 0:
                    field_names = field_tb.getcol("NAME")
                    # Look for common calibrator names in field names
                    common_calibrators = ["0834+555", "3C286", "3C48", "3C147", "3C138"]
                    for cal_name in common_calibrators:
                        if any(cal_name.lower() in str(name).lower() for name in field_names):
                            catalog = load_vla_catalog()
                            ra_deg, dec_deg = get_calibrator_radec(catalog, cal_name)
                            # Get flux from catalog or use explicit override
                            if cal_flux_jy is not None:
                                flux_jy = float(cal_flux_jy)
                            else:
                                flux_jy = _get_calibrator_flux_from_catalog(catalog, cal_name)
                            name = cal_name
                            logger.info(
                                f"Auto-detected calibrator from field names: {name} @ ({ra_deg:.4f}°, {dec_deg:.4f}°), {flux_jy:.2f} Jy"
                            )
                            break
                    else:
                        raise ValueError(
                            "Could not auto-detect calibrator from MS field names. "
                            "Provide calibrator_name or explicit coordinates."
                        )
                else:
                    raise ValueError(
                        "Could not read field names from MS. "
                        "Provide calibrator_name or explicit coordinates."
                    )
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(
                f"Could not auto-detect calibrator: {e}. "
                "Provide calibrator_name or explicit coordinates."
            ) from e

    # NOTE: clearcal() is NOT called here because:
    # 1. It's extremely slow on large datasets (1.8M rows takes 20+ minutes)
    # 2. _calculate_manual_model_data() already handles column creation via _ensure_imaging_columns()
    # 3. _calculate_manual_model_data() initializes MODEL_DATA with zeros before writing

    # Write MODEL_DATA using manual calculation (bypasses ft() phase center bugs)
    logger.info(f"Populating MODEL_DATA for {name} using manual calculation...")
    write_point_model_with_ft(
        ms_path,
        ra_deg,
        dec_deg,
        flux_jy,
        field=field,
        use_manual=True,  # Critical: bypasses ft() phase center bugs
        initialize_corrected=initialize_corrected,
    )
    logger.info(f":check: MODEL_DATA populated for {name}")


def populate_model_from_image(
    ms_path: str,
    *,
    field: str | None = None,
    model_image: str,
) -> None:
    """Populate MODEL_DATA from image file using CASA ft().

    This is used for self-calibration workflows where the model comes from
    a CLEANed image rather than a catalog or point source.

    **Phase Center Handling**:
    Internally calls write_image_model_with_ft() which syncs REFERENCE_DIR
    with PHASE_DIR before calling ft(). This ensures correct phase center
    handling after phaseshift().

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    field :
        Field selection (currently unused - ft() applies to all fields).
        Accepted for API compatibility but has no effect on behavior.
    model_image :
        Path to model image file (CASA image or FITS)

    Raises
    ------
    FileNotFoundError
        If model image does not exist

    """
    # field parameter is unused — ft() applies model to all fields.
    # Kept for API compatibility with calibration_helpers.py.

    if not os.path.exists(model_image):
        raise FileNotFoundError(f"Model image not found: {model_image}")

    logger.info(f"Populating MODEL_DATA from image: {model_image}")
    write_image_model_with_ft(ms_path, model_image)
    logger.info(":check: MODEL_DATA populated from image")


def count_bright_sources_in_tile(
    pointing_ra_deg: float,
    pointing_dec_deg: float,
    min_flux_mjy: float = 5.0,
    radius_deg: float = 0.3,
) -> int:
    """Count bright sources within a tile for gain calibration.

    Queries unified catalog (NVSS/VLASS/RAX) for sources within the specified
    radius of the pointing center and counts those above the flux threshold.

    Parameters
    ----------
    pointing_ra_deg : float
        Pointing RA in degrees
    pointing_dec_deg : float
        Pointing declination in degrees
    min_flux_mjy : float, optional
        Minimum flux threshold in mJy (default: 5.0)
    radius_deg : float, optional
        Search radius in degrees (default: 0.3)

    Returns
    -------
    int
        Number of sources above flux threshold within radius
    """
    from dsa110_contimg.core.calibration.catalogs import query_catalog_sources
    from dsa110_contimg.core.calibration.mosaic_constants import (
        EARTH_ROTATION_DEG_PER_SEC,
        INTEGRATION_TIME_SEC,
        N_FIELDS,
    )

    # Predict all 24 field centers for this tile
    field_centers = []
    half_obs_sec = (N_FIELDS - 1) * INTEGRATION_TIME_SEC / 2.0

    for field_idx in range(N_FIELDS):
        offset_sec = field_idx * INTEGRATION_TIME_SEC - half_obs_sec
        ra_drift_deg = offset_sec * EARTH_ROTATION_DEG_PER_SEC
        field_ra_deg = (pointing_ra_deg + ra_drift_deg) % 360.0
        field_centers.append((field_ra_deg, pointing_dec_deg))

    # Query sources for each field and collect all unique sources
    all_sources = []

    for field_ra, field_dec in field_centers:
        # Try VLASS first (highest resolution), then NVSS
        for catalog_type in ["vlass", "nvss"]:
            try:
                sources_df = query_catalog_sources(
                    catalog_type=catalog_type,
                    ra_deg=field_ra,
                    dec_deg=field_dec,
                    radius_deg=radius_deg,
                    min_flux_mjy=min_flux_mjy,
                )

                if sources_df is not None and len(sources_df) > 0:
                    all_sources.extend(sources_df.to_dict("records"))
                    break  # Use first catalog with results

            except (ValueError, KeyError, RuntimeError) as e:
                logger.debug(f"Error querying {catalog_type} catalog: {e}")
                continue

    # Deduplicate sources by position (5 arcsec match radius)
    unique_sources = deduplicate_sources(all_sources, match_radius_arcsec=5.0)

    return len(unique_sources)


def deduplicate_sources(sources: list[dict], match_radius_arcsec: float = 5.0) -> list[dict]:
    """Deduplicate sources by position matching.

    Parameters
    ----------
    sources : list[dict]
        List of source dictionaries with 'ra_deg' and 'dec_deg' keys
    match_radius_arcsec : float, optional
        Match radius in arcseconds (default: 5.0)

    Returns
    -------
    list[dict]
        Deduplicated list of sources
    """
    if not sources:
        return []

    from astropy.coordinates import SkyCoord

    # Convert to SkyCoord for efficient matching
    coords = SkyCoord(
        ra=[s["ra_deg"] for s in sources],
        dec=[s["dec_deg"] for s in sources],
        unit="deg",
    )

    # Match radius in degrees
    match_radius_deg = match_radius_arcsec / 3600.0

    # Keep track of which sources to keep
    keep_indices = []
    matched = set()

    for i, coord in enumerate(coords):
        if i in matched:
            continue

        # Find all sources within match radius
        seps = coord.separation(coords)
        matches = np.where(seps.deg <= match_radius_deg)[0]

        # Keep the first (brightest if sorted) and mark others as matched
        keep_indices.append(i)
        matched.update(matches)

    return [sources[i] for i in keep_indices]


def select_sky_model_sources(pointing_ra_deg: float, pointing_dec_deg: float) -> list[dict]:
    """Select sources for gain calibration sky model.

    Predicts all 24 field centers, queries unified catalog for each field,
    and returns deduplicated sources for building the sky model.

    Parameters
    ----------
    pointing_ra_deg : float
        Pointing RA in degrees
    pointing_dec_deg : float
        Pointing declination in degrees

    Returns
    -------
    list[dict]
        List of source dictionaries with 'ra_deg', 'dec_deg', 'flux_mjy' keys.
        If fewer than MIN_SKYMODEL_SOURCES are found, returns only the brightest source.
    """
    from dsa110_contimg.core.calibration.catalogs import query_catalog_sources
    from dsa110_contimg.core.calibration.mosaic_constants import (
        EARTH_ROTATION_DEG_PER_SEC,
        INTEGRATION_TIME_SEC,
        MIN_SKYMODEL_SOURCES,
        N_FIELDS,
        SKYMODEL_MIN_FLUX_MJY,
        SOURCE_QUERY_RADIUS_DEG,
    )

    # Predict all 24 field centers
    field_centers = []
    half_obs_sec = (N_FIELDS - 1) * INTEGRATION_TIME_SEC / 2.0

    for field_idx in range(N_FIELDS):
        offset_sec = field_idx * INTEGRATION_TIME_SEC - half_obs_sec
        ra_drift_deg = offset_sec * EARTH_ROTATION_DEG_PER_SEC
        field_ra_deg = (pointing_ra_deg + ra_drift_deg) % 360.0
        field_centers.append((field_ra_deg, pointing_dec_deg))

    # Query unified catalog for each field
    all_sources = []

    for field_ra, field_dec in field_centers:
        # Try VLASS first (highest resolution), then NVSS
        for catalog_type in ["vlass", "nvss"]:
            try:
                sources_df = query_catalog_sources(
                    catalog_type=catalog_type,
                    ra_deg=field_ra,
                    dec_deg=field_dec,
                    radius_deg=SOURCE_QUERY_RADIUS_DEG,
                    min_flux_mjy=SKYMODEL_MIN_FLUX_MJY,
                )

                if sources_df is not None and len(sources_df) > 0:
                    all_sources.extend(sources_df.to_dict("records"))
                    break  # Use first catalog with results

            except (ValueError, KeyError, RuntimeError) as e:
                logger.debug(f"Error querying {catalog_type} catalog: {e}")
                continue

    # Deduplicate by position (5 arcsec match radius)
    unique_sources = deduplicate_sources(all_sources, match_radius_arcsec=5.0)

    if len(unique_sources) >= MIN_SKYMODEL_SOURCES:
        # Use all sources above threshold
        logger.info(f"Selected {len(unique_sources)} sources for sky model")
        return unique_sources
    else:
        # Fall back to single brightest source
        if unique_sources:
            brightest = max(unique_sources, key=lambda s: s.get("flux_mjy", 0))
            logger.warning(
                f"Only {len(unique_sources)} sources found (< {MIN_SKYMODEL_SOURCES}), "
                f"using brightest source only"
            )
            return [brightest]
        else:
            logger.error("No sources found for sky model")
            return []
