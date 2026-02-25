#!/usr/bin/env python
"""Generate zero-filled placeholder HDF5 for missing DSA-110 subbands.

This module creates placeholder files that match the structure of real
subband files but contain only zeros and have all data flagged. This
allows incomplete observation groups to still be processed through the
pipeline, with the missing subbands properly flagged as bad data.
"""

import argparse
import logging
import os
import sqlite3
from pathlib import Path

import numpy as np
from pyuvdata import UVData

logger = logging.getLogger(__name__)


def create_placeholder_hdf5(
    reference_hdf5_path: str,
    output_path: str,
    target_subband_code: str,
    target_subband_num: int,
    compress: bool = True,
) -> tuple[bool, str]:
    """Create a zero-filled placeholder HDF5 matching reference file structure.

    Parameters
    ----------
    reference_hdf5_path :
        Path to existing subband file in same
        observation group
    output_path :
        Where to write the placeholder file
    target_subband_code :
        Subband code (e.g., "sb06")
    target_subband_num :
        Subband number 0-15
    compress :
        Whether to enable HDF5 compression (default True)

    Returns
    -------
    Tuple of (success
        bool, message: str)

    """
    try:
        # Load reference file to get structure
        logger.info("Loading reference file: %s", reference_hdf5_path)
        ref_uv = UVData()
        ref_uv.read(reference_hdf5_path, file_type="uvh5", run_check=False)

        # Extract reference metadata
        n_times = ref_uv.Ntimes
        n_blts = ref_uv.Nblts
        n_freqs = ref_uv.Nfreqs
        n_pols = ref_uv.Npols

        logger.info(
            "Reference dimensions: %d blts, %d times, %d freqs, %d pols",
            n_blts,
            n_times,
            n_freqs,
            n_pols,
        )

        # Calculate target subband frequency array
        # DSA-110 has 16 subbands from 1.28-1.53 GHz
        freq_min_hz = 1.28e9
        freq_max_hz = 1.53e9
        n_subbands = 16

        subband_bw = (freq_max_hz - freq_min_hz) / n_subbands
        freq_start = freq_min_hz + target_subband_num * subband_bw

        # Create frequency array matching reference spacing
        freq_array = ref_uv.freq_array
        if freq_array.ndim == 2:
            freq_spacing = np.median(np.diff(freq_array[0]))
        else:
            freq_spacing = np.median(np.diff(freq_array))
        target_freqs = freq_start + np.arange(n_freqs) * freq_spacing

        logger.info(
            "Target subband %s: %.4f-%.4f GHz",
            target_subband_code,
            target_freqs[0] / 1e9,
            target_freqs[-1] / 1e9,
        )

        # Create placeholder UVData object
        placeholder_uv = ref_uv.copy()

        # Update frequency array (match the shape of original)
        if placeholder_uv.freq_array.ndim == 2:
            placeholder_uv.freq_array = target_freqs.reshape(1, -1)
        else:
            placeholder_uv.freq_array = target_freqs

        # Zero out visibility data
        placeholder_uv.data_array = np.zeros_like(placeholder_uv.data_array)

        # Flag all data as bad
        placeholder_uv.flag_array = np.ones_like(placeholder_uv.flag_array, dtype=bool)

        # Set nsample to zero
        placeholder_uv.nsample_array = np.zeros_like(placeholder_uv.nsample_array)

        # Update metadata to indicate this is a placeholder
        placeholder_uv.extra_keywords = placeholder_uv.extra_keywords.copy()
        placeholder_uv.extra_keywords["IS_PLACEHOLDER"] = "True"
        placeholder_uv.extra_keywords["SUBBAND_CODE"] = target_subband_code
        placeholder_uv.extra_keywords["SUBBAND_NUM"] = str(target_subband_num)
        placeholder_uv.extra_keywords["PLACEHOLDER_REASON"] = "Missing subband hardware failure"

        # Write to HDF5
        logger.info("Writing placeholder to: %s", output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write (skip checks due to float32/float64 type mismatches in source)
        placeholder_uv.write_uvh5(output_path, clobber=True, run_check=False)

        # Check file size
        file_size_mb = Path(output_path).stat().st_size / (1024**2)
        logger.info("Placeholder created successfully: %.2f MB", file_size_mb)

        return True, f"Created placeholder: {file_size_mb:.2f} MB"

    except Exception as e:
        error_msg = f"Failed to create placeholder: {e}"
        logger.error("Failed to create placeholder: %s", e, exc_info=True)
        return False, error_msg


def find_reference_file_for_group(
    hdf5_db_path: Path, group_id: str, prefer_sb00: bool = True
) -> str | None:
    """Find a reference file from the same observation group.

    Parameters
    ----------
    hdf5_db_path :
        Path to HDF5 database
    group_id :
        Group ID (timestamp) for the observation
    prefer_sb00 :
        If True, prefer sb00 as reference (default True)

    Returns
    -------
        Path to reference file, or None if not found

    """
    conn = sqlite3.connect(hdf5_db_path)

    # Try to get sb00 first if preferred
    if prefer_sb00:
        result = conn.execute(
            """
            SELECT path
            FROM hdf5_files
            WHERE group_id = ? AND subband_code = 'sb00'
            LIMIT 1
        """,
            (group_id,),
        ).fetchone()

        if result:
            conn.close()
            return result[0]

    # Otherwise get any file from the group
    result = conn.execute(
        """
        SELECT path
        FROM hdf5_files
        WHERE group_id = ?
        LIMIT 1
    """,
        (group_id,),
    ).fetchone()

    conn.close()
    return result[0] if result else None


def generate_placeholders_for_incomplete_groups(
    hdf5_db_path: Path,
    output_dir: Path,
    tolerance_s: float = 120.0,
    dry_run: bool = True,
    max_placeholders: int | None = None,
) -> dict:
    """Scan database for incomplete groups and generate all needed placeholders.

    Parameters
    ----------
    hdf5_db_path :
        Path to HDF5 database
    output_dir :
        Directory to write placeholder files
    tolerance_s :
        Time tolerance for grouping (default 120.0 seconds)
    dry_run :
        If True, only report what would be done (default True)
    max_placeholders :
        Maximum number of placeholders to create (None = unlimited)

    Returns
    -------
    Dictionary with statistics

    Dictionary with statistics
        {
        'total_incomplete_groups': int,
        'total_placeholders_needed': int,
        'total_placeholders_created': int,
        'placeholders_by_subband': Dict[str, int],
        'storage_used_mb': float,
        'errors': List[str]
    Dictionary with statistics
        {
        'total_incomplete_groups': int,
        'total_placeholders_needed': int,
        'total_placeholders_created': int,
        'placeholders_by_subband': Dict[str, int],
        'storage_used_mb': float,
        'errors': List[str]
        }

    """
    logger.info("Scanning database for incomplete groups...")
    logger.info("Database: %s", hdf5_db_path)
    logger.info("Output directory: %s", output_dir)
    logger.info("Dry run: %s", dry_run)

    conn = sqlite3.connect(hdf5_db_path)

    # Get all files
    files = conn.execute(
        """
        SELECT subband_code, timestamp_mjd, group_id, path
        FROM hdf5_files
        ORDER BY timestamp_mjd
    """
    ).fetchall()

    conn.close()

    logger.info("Total files in database: %d", len(files))

    # Use proximity-based grouping to find incomplete groups
    sb_codes = np.array([f[0] for f in files])
    times_sec = np.array([f[1] * 86400.0 for f in files])
    group_ids = np.array([f[2] for f in files])
    paths = np.array([f[3] for f in files])

    expected_sb = set([f"sb{idx:02d}" for idx in range(16)])
    incomplete_groups = []
    used = np.zeros(len(times_sec), dtype=bool)

    logger.info("Identifying incomplete groups...")

    for i in range(len(times_sec)):
        if used[i]:
            continue

        close_indices = np.where(np.abs(times_sec - times_sec[i]) <= tolerance_s)[0]
        group_indices = [idx for idx in close_indices if not used[idx]]

        subband_map = {}
        for idx in group_indices:
            if sb_codes[idx] not in subband_map:
                subband_map[sb_codes[idx]] = {"index": idx, "path": paths[idx]}

        present_sbs = set(subband_map.keys())

        if present_sbs != expected_sb:
            missing_sbs = expected_sb - present_sbs

            # Get reference path (prefer sb00)
            ref_path = (
                subband_map["sb00"]["path"]
                if "sb00" in subband_map
                else list(subband_map.values())[0]["path"]
            )

            incomplete_groups.append(
                {
                    "group_id": group_ids[i],
                    "reference_path": ref_path,
                    "missing_sbs": missing_sbs,
                }
            )

            for idx in group_indices:
                used[idx] = True

    logger.info("Found %d incomplete groups", len(incomplete_groups))

    # Count placeholders needed
    placeholders_by_subband = {}
    for group in incomplete_groups:
        for sb in group["missing_sbs"]:
            placeholders_by_subband[sb] = placeholders_by_subband.get(sb, 0) + 1

    total_placeholders_needed = sum(placeholders_by_subband.values())

    logger.info("Total placeholders needed: %d", total_placeholders_needed)
    for sb in sorted(placeholders_by_subband.keys()):
        logger.info("  %s: %d", sb, placeholders_by_subband[sb])

    # Generate placeholders
    stats = {
        "total_incomplete_groups": len(incomplete_groups),
        "total_placeholders_needed": total_placeholders_needed,
        "total_placeholders_created": 0,
        "placeholders_by_subband": {sb: 0 for sb in placeholders_by_subband.keys()},
        "storage_used_mb": 0.0,
        "errors": [],
    }

    if dry_run:
        logger.info("DRY RUN - no files will be created")
        return stats

    # Create placeholders
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, group in enumerate(incomplete_groups):
        if max_placeholders and stats["total_placeholders_created"] >= max_placeholders:
            logger.info("Reached maximum placeholder limit: %d", max_placeholders)
            break

        group_id = group["group_id"]
        ref_path = group["reference_path"]

        for sb_code in sorted(group["missing_sbs"]):
            sb_num = int(sb_code[2:])

            # Generate output filename
            output_filename = f"{group_id}_{sb_code}.hdf5"
            output_path = output_dir / output_filename

            logger.info(
                "[%d/%d] Creating %s for %s",
                stats["total_placeholders_created"] + 1,
                total_placeholders_needed,
                sb_code,
                group_id,
            )

            success, message = create_placeholder_hdf5(
                ref_path, str(output_path), sb_code, sb_num, compress=True
            )

            if success:
                stats["total_placeholders_created"] += 1
                stats["placeholders_by_subband"][sb_code] += 1

                # Update storage stats
                if output_path.exists():
                    stats["storage_used_mb"] += output_path.stat().st_size / (1024**2)
            else:
                stats["errors"].append(f"{group_id}_{sb_code}: {message}")

    logger.info("\nPlaceholder generation complete!")
    logger.info(
        "Created: %d/%d",
        stats["total_placeholders_created"],
        total_placeholders_needed,
    )
    logger.info("Storage used: %.2f MB", stats["storage_used_mb"])

    if stats["errors"]:
        logger.warning("Errors: %d", len(stats["errors"]))
        for error in stats["errors"][:10]:  # Show first 10 errors
            logger.warning("  %s", error)

    return stats


def main():
    """Command-line interface for placeholder generation."""
    parser = argparse.ArgumentParser(
        description="Generate zero-filled placeholder HDF5 files for missing subbands"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Single file command
    single_parser = subparsers.add_parser("single", help="Create a single placeholder file")
    single_parser.add_argument("--reference", required=True, help="Reference HDF5 file path")
    single_parser.add_argument("--output", required=True, help="Output placeholder file path")
    single_parser.add_argument(
        "--subband-code", required=True, help="Target subband code (e.g., sb06)"
    )
    single_parser.add_argument(
        "--subband-num", type=int, required=True, help="Target subband number (0-15)"
    )
    single_parser.add_argument("--no-compress", action="store_true", help="Disable compression")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Generate all needed placeholders")
    batch_parser.add_argument(
        "--hdf5-db",
        type=Path,
        default=Path("/data/dsa110-contimg/state/db/pipeline.sqlite3"),
        help="HDF5 database path (unified pipeline database)",
    )
    batch_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/incoming/synthetic-vis"),
        help="Output directory for placeholders",
    )
    batch_parser.add_argument(
        "--tolerance",
        type=float,
        default=60.0,
        help="Time tolerance for grouping (seconds)",
    )
    batch_parser.add_argument(
        "--max-placeholders",
        type=int,
        help="Maximum number of placeholders to create (for testing)",
    )
    batch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be done without creating files",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if args.command == "single":
        success, message = create_placeholder_hdf5(
            args.reference,
            args.output,
            args.subband_code,
            args.subband_num,
            compress=not args.no_compress,
        )

        if success:
            print(f":check_mark: {message}")
            return 0
        else:
            print(f":ballot_x: {message}")
            return 1

    elif args.command == "batch":
        stats = generate_placeholders_for_incomplete_groups(
            args.hdf5_db,
            args.output_dir,
            tolerance_s=args.tolerance,
            dry_run=args.dry_run,
            max_placeholders=args.max_placeholders,
        )

        print("\n" + "=" * 70)
        print("PLACEHOLDER GENERATION SUMMARY")
        print("=" * 70)
        print(f"Incomplete groups found: {stats['total_incomplete_groups']}")
        print(f"Placeholders needed: {stats['total_placeholders_needed']}")
        print(f"Placeholders created: {stats['total_placeholders_created']}")
        print(f"Storage used: {stats['storage_used_mb']:.2f} MB")

        if stats["errors"]:
            print(f"\nErrors: {len(stats['errors'])}")

        return 0 if not stats["errors"] else 1

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())
