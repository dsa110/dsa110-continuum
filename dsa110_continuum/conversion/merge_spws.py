"""Utility to merge multiple SPWs into a single SPW Measurement Set.

This module provides functions to convert multi-SPW MS files (created by
direct-subband writer) into single-SPW MS files using CASA mstransform.
"""

from __future__ import annotations

import os
import shutil

import numpy as np

# Ensure CASAPATH is set before importing CASA modules
from dsa110_contimg.common.utils.casa_init import ensure_casa_path

ensure_casa_path()

import casacore.tables as casatables  # type: ignore[import]

table = casatables.table  # noqa: N816


from dsa110_contimg.common.utils.runtime_safeguards import require_casa6_python

# Use canonical angular_separation with numba→astropy fallback chain
from dsa110_contimg.core.conversion.helpers_coordinates import angular_separation


@require_casa6_python
def merge_spws(
    ms_in: str,
    ms_out: str,
    *,
    datacolumn: str = "DATA",
    regridms: bool = True,
    interpolation: str = "linear",
    keepflags: bool = True,
    remove_sigma_spectrum: bool = True,
) -> str:
    """Merge multiple SPWs into a single SPW using CASA mstransform.

    This function takes a multi-SPW MS (e.g., created by direct-subband writer
    with 16 SPWs) and creates a single-SPW MS with all frequencies combined.

    Parameters
    ----------
    ms_in :
        Input multi-SPW Measurement Set path
    ms_out :
        Output single-SPW Measurement Set path
    datacolumn :
        Data column to use ('DATA', 'CORRECTED_DATA', etc.)
    regridms :
        If True, regrid to a contiguous frequency grid. If False,
        combine SPWs without regridding (may have gaps).
    interpolation :
        Interpolation method when regridding
        ('linear', 'nearest', etc.)
    keepflags :
        Preserve flagging information
    remove_sigma_spectrum :
        If True, remove SIGMA_SPECTRUM column after
        merge to save disk space (default: True). SIGMA_SPECTRUM is
        automatically created by mstransform when combining SPWs, but
        contains redundant information (SIGMA repeated per channel).

    Returns
    -------
        Path to output MS

    Raises
    ------
    FileNotFoundError
        If input MS doesn't exist
    RuntimeError
        If mstransform fails

    """
    if not os.path.exists(ms_in):
        raise FileNotFoundError(f"Input MS not found: {ms_in}")

    # Remove existing output if present
    if os.path.isdir(ms_out):
        shutil.rmtree(ms_out, ignore_errors=True)

    kwargs = dict(
        vis=ms_in,
        outputvis=ms_out,
        datacolumn=datacolumn,
        combinespws=True,
        regridms=regridms,
        keepflags=keepflags,
    )

    if regridms:
        # Build global frequency grid from all SPWs
        with table(f"{ms_in}::SPECTRAL_WINDOW", readonly=True) as spw:
            cf = np.asarray(spw.getcol("CHAN_FREQ"))  # shape (nspw, nchan)

        # Flatten and sort all frequencies
        all_freq = np.sort(cf.reshape(-1))

        # Calculate channel width (median of frequency differences)
        freq_diffs = np.diff(all_freq)
        dnu = float(np.median(freq_diffs[freq_diffs > 0]))

        nchan = int(all_freq.size)
        start = float(all_freq[0])

        kwargs.update(
            mode="frequency",
            nchan=nchan,
            start=f"{start}Hz",
            width=f"{dnu}Hz",
            interpolation=interpolation,
        )

    # Use CASAService for mstransform
    from dsa110_contimg.core.calibration.casa_service import CASAService

    service = CASAService()
    try:
        service.mstransform(**kwargs)
    except Exception as e:
        raise RuntimeError(f"mstransform failed: {e}") from e

    if not os.path.exists(ms_out):
        raise RuntimeError(f"mstransform failed to create output MS: {ms_out}")

    # Remove SIGMA_SPECTRUM if requested (to save disk space)
    # SIGMA_SPECTRUM is automatically created by mstransform when combining
    # SPWs, but contains redundant information (SIGMA values repeated per
    # channel).
    if remove_sigma_spectrum:
        try:
            with table(ms_out, readonly=False) as tb:
                if "SIGMA_SPECTRUM" in tb.colnames():
                    tb.removecols(["SIGMA_SPECTRUM"])
        except (RuntimeError, OSError):
            # Non-fatal: continue if removal fails
            # RuntimeError: CASA table errors, OSError: file access issues
            pass

    # Fix telescope name to avoid listobs() errors with custom telescope names
    # CASA doesn't recognize "DSA_110", so change to "OVRO_MMA" which exists in
    # the CASA Observatories table (plain "OVRO" does NOT exist)
    try:
        from casatools import table as casa_table

        tb_obs = casa_table()
        tb_obs.open(ms_out + "/OBSERVATION", nomodify=False)
        current_name = tb_obs.getcol("TELESCOPE_NAME")
        # Only change if it's DSA_110 (or other unrecognized names)
        if current_name and "DSA_110" in str(current_name[0]):
            tb_obs.putcol("TELESCOPE_NAME", ["OVRO_MMA"])
        tb_obs.close()
        tb_obs.done()  # Required: casatools.table needs both close() and done()
    except (RuntimeError, OSError, ImportError):
        # Non-fatal: telescope name fix is cosmetic
        # RuntimeError: CASA errors, OSError: file issues, ImportError: casatools
        pass

    return ms_out


@require_casa6_python
def merge_spws_simple(
    ms_in: str,
    ms_out: str,
    *,
    datacolumn: str = "DATA",
    keepflags: bool = True,
    remove_sigma_spectrum: bool = True,
) -> str:
    """Simple SPW merging without regridding (combines SPWs but may have gaps).

    This is faster than merge_spws() but may result in discontinuous frequency
    coverage if subbands have gaps.

    Parameters
    ----------
    ms_in :
        Input multi-SPW Measurement Set path
    ms_out :
        Output single-SPW Measurement Set path
    datacolumn :
        Data column to use
    keepflags :
        Preserve flagging information
    remove_sigma_spectrum :
        If True, remove SIGMA_SPECTRUM column after
        merge

    Returns
    -------
        Path to output MS

    """
    return merge_spws(
        ms_in=ms_in,
        ms_out=ms_out,
        datacolumn=datacolumn,
        regridms=False,
        keepflags=keepflags,
        remove_sigma_spectrum=remove_sigma_spectrum,
    )


def get_spw_count(ms_path: str) -> int | None:
    """Get the number of spectral windows in an MS.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set

    Returns
    -------
        Number of SPWs, or None if unable to read

    """
    try:
        with table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True, ack=False) as spw:
            return spw.nrows()
    except (RuntimeError, OSError):
        # RuntimeError: CASA table errors, OSError: file access issues
        return None


def merge_fields(
    ms_in: str,
    ms_out: str,
    *,
    datacolumn: str = "DATA",
    keepflags: bool = True,
) -> str:
    """Merge multiple fields into a single field using direct table manipulation.

    This function takes a multi-field MS (e.g., with time-binned fields) and
    creates a single-field MS by reassigning all rows to field 0 and updating
    the FIELD table accordingly.

    Parameters
    ----------
    ms_in :
        Input multi-field Measurement Set path
    ms_out :
        Output single-field Measurement Set path
    datacolumn :
        Data column to use ('DATA', 'CORRECTED_DATA', etc.)
    keepflags :
        Preserve flagging information

    Returns
    -------
        Path to output MS

    Raises
    ------
    FileNotFoundError
        If input MS doesn't exist
    RuntimeError
        If field merging fails

    """
    if not os.path.exists(ms_in):
        raise FileNotFoundError(f"Input MS not found: {ms_in}")

    # Remove existing output if present
    if os.path.isdir(ms_out):
        shutil.rmtree(ms_out, ignore_errors=True)

    # Copy the MS first
    shutil.copytree(ms_in, ms_out)

    try:
        # Read field information from original MS
        with table(f"{ms_in}::FIELD", readonly=True) as field_in:
            nfields = field_in.nrows()
            if nfields == 0:
                raise RuntimeError("Input MS has no fields")

            # Get phase center from first field
            phase_dirs = field_in.getcol("PHASE_DIR")  # Shape: (nfields, npoly, 2)
            ref_phase_dir = phase_dirs[0]  # Use first field's phase center
            ref_ra_rad = ref_phase_dir[0][0]  # Reference RA (radians)
            ref_dec_rad = ref_phase_dir[0][1]  # Reference Dec (radians)
            field_names = field_in.getcol("NAME")
            ref_name = field_names[0] if len(field_names) > 0 else "merged_field"

            # Validate that all fields share the same phase center (within tolerance)
            # This is critical: merging fields with different phase centers produces incorrect results
            tolerance_arcsec = 1.0  # 1 arcsecond tolerance
            tolerance_rad = np.deg2rad(tolerance_arcsec / 3600.0)

            max_separation_rad = 0.0
            mismatched_fields = []

            for i in range(1, nfields):
                ra_rad = phase_dirs[i, 0, 0]
                dec_rad = phase_dirs[i, 0, 1]

                # Calculate angular separation
                separation_rad = angular_separation(ref_ra_rad, ref_dec_rad, ra_rad, dec_rad)
                max_separation_rad = max(max_separation_rad, separation_rad)

                if separation_rad > tolerance_rad:
                    mismatched_fields.append(i)

            max_separation_arcsec = np.rad2deg(max_separation_rad) * 3600.0

            if mismatched_fields:
                raise RuntimeError(
                    f"Cannot merge fields with different phase centers. "
                    f"Fields {mismatched_fields} have phase centers that differ from field 0 "
                    f"by more than {tolerance_arcsec} arcsec (max separation: {max_separation_arcsec:.3f} arcsec). "
                    f"All fields must be phased to the same position before merging. "
                    f"Consider using phaseshift() to phase all fields to a common target first."
                )

            print(f"Input MS has {nfields} fields")
            if nfields > 1:
                print(
                    f":check: Validated: all {nfields} fields share the same phase center (max separation: {max_separation_arcsec:.3f} arcsec)"
                )
            print(f"Using phase center from field 0: {ref_name}")

        # Reassign all rows to field 0 in main table
        with table(ms_out, readonly=False) as tb:
            if "FIELD_ID" not in tb.colnames():
                raise RuntimeError("Main table has no FIELD_ID column")

            nrows = tb.nrows()
            print(f"Reassigning {nrows:,} rows to field 0...")

            # Set all FIELD_ID to 0
            field_ids = np.zeros(nrows, dtype=np.int32)
            tb.putcol("FIELD_ID", field_ids)

        # Update FIELD table to have only one field
        with table(ms_out + "/FIELD", readonly=False) as field_out:
            # Get all columns from original field table
            all_colnames = field_out.colnames()

            # Read first row's data as template
            field_data = {}
            for colname in all_colnames:
                field_data[colname] = field_out.getcol(colname, startrow=0, nrow=1)

            # Remove all rows
            field_out.removerows(list(range(nfields)))

            # Add single merged field row
            field_out.addrows(1)

            # Write merged field data
            for colname in all_colnames:
                field_out.putcol(colname, field_data[colname], startrow=0)

            # Update name to indicate it's merged
            if "NAME" in field_out.colnames():
                field_out.putcol("NAME", [ref_name + "_merged"], startrow=0)

        print(f":check: Successfully merged {nfields} fields into 1 field")

        # Verify the result
        with table(ms_out + "/FIELD", readonly=True) as field_check:
            nfields_out = field_check.nrows()
            if nfields_out != 1:
                raise RuntimeError(f"Expected 1 field in output, got {nfields_out}")

            print(f"Output MS has {nfields_out} field")

    except Exception as e:
        # Clean up on error
        if os.path.exists(ms_out):
            shutil.rmtree(ms_out, ignore_errors=True)
        raise RuntimeError(f"Field merging failed: {e}") from e

    return ms_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge multiple SPWs into a single SPW Measurement Set"
    )
    parser.add_argument("ms_in", help="Input multi-SPW MS path")
    parser.add_argument("ms_out", help="Output single-SPW MS path")
    parser.add_argument(
        "--datacolumn",
        default="DATA",
        choices=["DATA", "CORRECTED_DATA", "MODEL_DATA"],
        help="Data column to use",
    )
    parser.add_argument(
        "--no-regrid",
        action="store_true",
        help="Combine SPWs without regridding (faster but may have gaps)",
    )
    parser.add_argument(
        "--interpolation",
        default="linear",
        choices=["linear", "nearest", "cubic"],
        help="Interpolation method for regridding",
    )
    parser.add_argument(
        "--keep-sigma-spectrum",
        action="store_true",
        help="Keep SIGMA_SPECTRUM column (default: remove to save disk space)",
    )

    args = parser.parse_args()

    print(f"Input MS: {args.ms_in}")
    n_spw_in = get_spw_count(args.ms_in)
    if n_spw_in:
        print(f"Input SPWs: {n_spw_in}")

    print(f"Output MS: {args.ms_out}")
    print(f"Regridding: {not args.no_regrid}")
    print(f"Remove SIGMA_SPECTRUM: {not args.keep_sigma_spectrum}")

    merge_spws(
        ms_in=args.ms_in,
        ms_out=args.ms_out,
        datacolumn=args.datacolumn,
        regridms=not args.no_regrid,
        interpolation=args.interpolation,
        remove_sigma_spectrum=not args.keep_sigma_spectrum,
    )

    n_spw_out = get_spw_count(args.ms_out)
    if n_spw_out:
        print(f"Output SPWs: {n_spw_out}")
        if n_spw_out == 1:
            print(":check: Successfully merged SPWs into single SPW")
        else:
            print(f":warning: Warning: Expected 1 SPW, got {n_spw_out}")
