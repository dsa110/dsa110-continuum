#!/usr/bin/env python
"""Validation utility for generated synthetic UVH5 files.

Checks that generated files are readable and, when layout metadata is
provided, verifies key schema elements (npol, polarization array, nfreqs,
channel width, etc.) against the reference configuration.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from pyuvdata import UVData


def validate_uvh5_file(
    filepath: Path, *, layout_meta: dict | None = None
) -> tuple[bool, list[str]]:
    """Validate a single UVH5 file.

    Parameters
    ----------

    Returns
    -------
        (is_valid, error_messages)

    """
    errors = []

    try:
        uv = UVData()
        uv.read(
            str(filepath),
            file_type="uvh5",
            run_check=False,
            run_check_acceptability=False,
            strict_uvw_antpos_check=False,
        )
    except Exception as e:
        return False, [f"Failed to read file: {e}"]

    # Optional strict checks from layout metadata
    if layout_meta is not None:
        expected_nants = layout_meta.get("nants") or layout_meta.get("n_ants")
        if expected_nants is not None and uv.Nants_telescope != int(expected_nants):
            errors.append(f"Expected {expected_nants} antennas, got {uv.Nants_telescope}")

        expected_npol = layout_meta.get("npol")
        if expected_npol is not None and uv.Npols != int(expected_npol):
            errors.append(f"Expected {expected_npol} polarizations, got {uv.Npols}")

        expected_pol_array = layout_meta.get("polarization_array")
        if expected_pol_array is not None:
            if not np.array_equal(uv.polarization_array, np.array(expected_pol_array)):
                errors.append("polarization_array mismatch")

        expected_nfreqs = layout_meta.get("nfreqs")
        if expected_nfreqs is not None and uv.Nfreqs != int(expected_nfreqs):
            errors.append(f"Expected {expected_nfreqs} channels, got {uv.Nfreqs}")

        expected_width = layout_meta.get("channel_width_hz")
        if expected_width is not None:
            widths = np.abs(np.array(uv.channel_width).reshape(-1))
            if not np.allclose(widths, abs(float(expected_width)), rtol=1e-6, atol=1e-3):
                errors.append("channel width deviation detected")

    # Check integration time (DSA-110 typical: ~12-13 seconds)
    int_time = uv.integration_time[0]
    if not (10.0 < int_time < 20.0):
        errors.append(f"Integration time {int_time:.2f}s is unusual for DSA-110")

    # Check data array shape
    expected_shape = (uv.Nblts, uv.Nspws, uv.Nfreqs, uv.Npols)
    if uv.data_array.shape != expected_shape:
        # pyuvdata 3.x may squeeze Nspws dimension if it is 1
        if uv.Nspws == 1 and uv.data_array.shape == (uv.Nblts, uv.Nfreqs, uv.Npols):
            pass  # This is acceptable
        else:
            errors.append(f"Data array shape {uv.data_array.shape} != expected {expected_shape}")

    # Check for NaN or Inf in data
    if np.any(np.isnan(uv.data_array)):
        errors.append("Data array contains NaN values")
    if np.any(np.isinf(uv.data_array)):
        errors.append("Data array contains Inf values")

    # Check flag array
    if uv.flag_array.shape != uv.data_array.shape:
        errors.append(f"Flag array shape mismatch: {uv.flag_array.shape} vs {uv.data_array.shape}")

    return (len(errors) == 0), errors


def validate_subband_group(
    directory: Path, timestamp: str, *, layout_meta: dict | None = None
) -> tuple[bool, list[str]]:
    """Validate a complete subband group.

    Parameters
    ----------
    directory :
        Directory containing subband files
    timestamp :
        Expected timestamp string (e.g., "2025-10-06T12:00:00")

    Returns
    -------
        (is_valid, error_messages)

    """
    errors = []

    # Find all subbands for this timestamp
    pattern = f"{timestamp}_sb*.hdf5"
    subband_files = sorted(directory.glob(pattern))

    if len(subband_files) == 0:
        return False, [f"No subband files found matching {pattern}"]

    # Check subband count
    if len(subband_files) not in [4, 16]:  # 4 for minimal, 16 for full
        errors.append(f"Expected 16 subbands (or 4 for minimal), found {len(subband_files)}")

    # Validate each subband
    for sb_file in subband_files:
        is_valid, sb_errors = validate_uvh5_file(sb_file, layout_meta=layout_meta)
        if not is_valid:
            errors.append(f"{sb_file.name}: {'; '.join(sb_errors)}")

    # Check timestamps match
    timestamps = set()
    for sb_file in subband_files:
        # Extract timestamp from filename
        ts = sb_file.stem.rsplit("_sb", 1)[0]
        timestamps.add(ts)

    if len(timestamps) > 1:
        errors.append(f"Multiple timestamps found in group: {timestamps}")

    return (len(errors) == 0), errors


def print_summary(filepath: Path):
    """Print summary information about a UVH5 file.

    Parameters
    ----------
    """
    try:
        uv = UVData()
        uv.read(
            str(filepath),
            file_type="uvh5",
            run_check=False,
            run_check_acceptability=False,
            strict_uvw_antpos_check=False,
        )

        print(f"\n{filepath.name}:")
        print(f"  Antennas: {uv.Nants_telescope}")
        print(f"  Baselines: {uv.Nbls}")
        print(f"  Times: {uv.Ntimes}")
        print(f"  Frequencies: {uv.Nfreqs}")
        print(f"  Polarizations: {uv.Npols}")
        print(f"  Integration time: {uv.integration_time[0]:.2f} s")
        print(
            f"  Freq range: {uv.freq_array.min() / 1e6:.1f} - {uv.freq_array.max() / 1e6:.1f} MHz"
        )
        print(f"  Data shape: {uv.data_array.shape}")

    except Exception as e:
        print(f"\n{filepath.name}: ERROR - {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate synthetic UVH5 files for DSA-110",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate single file
  %(prog)s file.hdf5

  # Validate entire observation group
  %(prog)s --group /path/to/subbands --timestamp "2025-10-06T12:00:00"

  # Print summary of all files in directory
  %(prog)s --summary /path/to/subbands/*.hdf5
        """,
    )

    parser.add_argument("files", nargs="*", type=Path, help="UVH5 files to validate")
    parser.add_argument("--group", type=Path, help="Directory containing subband group")
    parser.add_argument("--timestamp", type=str, help="Timestamp for subband group validation")
    parser.add_argument(
        "--summary", action="store_true", help="Print summary instead of validation"
    )

    args = parser.parse_args()

    # Group validation mode
    if args.group and args.timestamp:
        print(f"Validating subband group: {args.timestamp}")
        is_valid, errors = validate_subband_group(args.group, args.timestamp)

        if is_valid:
            print(":check_mark: Subband group is valid")
            return 0
        else:
            print(":ballot_x: Subband group validation failed:")
            for error in errors:
                print(f"  - {error}")
            return 1

    # File validation mode
    if not args.files:
        parser.print_help()
        return 1

    if args.summary:
        # Summary mode
        for filepath in args.files:
            if filepath.exists():
                print_summary(filepath)
        return 0

    # Validation mode
    all_valid = True
    for filepath in args.files:
        if not filepath.exists():
            print(f":ballot_x: {filepath}: File not found")
            all_valid = False
            continue

        is_valid, errors = validate_uvh5_file(filepath)
        if is_valid:
            print(f":check_mark: {filepath.name}: Valid")
        else:
            print(f":ballot_x: {filepath.name}: Invalid")
            for error in errors:
                print(f"  - {error}")
            all_valid = False

    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
