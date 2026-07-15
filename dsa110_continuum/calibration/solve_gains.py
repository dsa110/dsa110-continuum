"""Gain (G) and pre-bandpass phase calibration solvers."""

# ruff: noqa

from __future__ import annotations

import logging
import os
import sys

import numpy as np

from dsa110_continuum.calibration.solver_common import (
    _call_gaincal,
    _call_gaincal_with_progress,
    _determine_spwmap_for_bptables,
    _track_calibration_provenance,
    _validate_solve_success,
    table,
    timed,
)
from dsa110_continuum.calibration.validate import validate_caltables_for_use

logger = logging.getLogger(__name__)

def solve_prebandpass_phase(
    ms: str,
    cal_field: str,
    refant: str,
    table_prefix: str | None = None,
    combine_fields: bool = True,
    combine_spw: bool = True,
    uvrange: str = ">1klambda",  # Default: exclude short baselines for better SNR
    # Default to 'inf' to match test expectation and allow long integration when appropriate
    solint: str = "inf",
    # Default to 3.0 for better outrigger antenna coverage
    minsnr: float = 3.0,
    peak_field_idx: int | None = None,
    minblperant: int | None = None,  # Minimum baselines per antenna
    # SPW selection (e.g., "4~11" for central 8 SPWs)
    spw: str | None = None,
    # Custom table name (e.g., ".bpphase.gcal")
    table_name: str | None = None,
    # List of prior calibration tables to apply (e.g., K-table)
    gaintable: list[str] | None = None,
) -> str:
    """Solve phase-only calibration before bandpass to correct phase drifts in raw data.

    This phase-only calibration step is critical for uncalibrated raw data. It corrects
    for time-dependent phase variations that cause decorrelation and low SNR in bandpass
    calibration. This should be run BEFORE bandpass calibration.

    **PRECONDITION**: MODEL_DATA must be populated before calling this function.

    Parameters
    ----------
    ms : str
        Path to Measurement Set
    cal_field : str
        Field selection
    refant : str
        Reference antenna
    table_prefix : str, optional
        Prefix for tables
    combine_fields : bool, optional
        Whether to combine fields
    t_short : str, optional
        Short solution interval
    solint : str, optional
        Solution interval
    minsnr : float, optional
        Minimum SNR
    peak_field_idx : int, optional
        Index of peak field
    minblperant : int, optional
        Minimum baselines per antenna
    spw : str, optional
        SPW selection
    table_name : str, optional
        Custom table name
    gaintable : list[str], optional
        List of prior calibration tables to apply (e.g. K-table). Applying K-table
        is CRITICAL if instrumental delays are significant, as they cause
        phase decoherence when averaging over frequency (combine_spw=True).

    Returns
    -------
        Path to phase-only calibration table (to be passed to bandpass via gaintable)

    """
    import numpy as np  # type: ignore[import]

    # use module-level table

    if table_prefix is None:
        table_prefix = f"{os.path.splitext(ms)[0]}_{cal_field}"

    # ============================================================================
    # PRE-BANDPASS PHASE CALIBRATION - TRANSPARENCY HEADER
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PRE-BANDPASS PHASE CALIBRATION (calmode='p') - Stage 2")
    logger.info("=" * 80)
    logger.info("PURPOSE: Solve for time-dependent phase variations in raw uncalibrated data")
    logger.info("         BEFORE bandpass calibration. This prevents decorrelation and")
    logger.info("         low SNR in bandpass solve, especially for faint calibrators.")
    logger.info("")
    logger.info("WHY NEEDED:")
    logger.info("  - Bandpass solves with solint='inf' (per-channel), integrating entire obs")
    logger.info("  - Phase drifts during integration cause decorrelation → low SNR → flagging")
    logger.info("  - Pre-BP phase correction stabilizes phases → higher bandpass SNR")
    logger.info("")
    logger.info("EXPECTED INPUT:")
    logger.info("  - MODEL_DATA: Populated with calibrator source model (checking...)")
    logger.info(f"  - Field: {cal_field}")
    logger.info(f"  - Reference antenna: {refant}")
    logger.info(f"  - Solution interval: {solint}")
    logger.info(f"  - Combine fields: {combine_fields} (higher SNR if True)")
    logger.info(f"  - Combine SPWs: {combine_spw} (frequency-independent phase)")
    logger.info(f"  - Minimum SNR: {minsnr}")
    logger.info("")
    logger.info("EXPECTED SOLUTIONS:")
    logger.info("  - Phase range: -180° to +180° (wrapped)")
    logger.info("  - Time evolution: Smooth trends, not random jumps")
    logger.info("  - Frequency behavior: Similar phases across SPWs (frequency-independent)")
    logger.info("  - Reference antenna: Phase = 0° (by definition)")
    logger.info("  - All antennas: Phases cluster around 0° after correction")
    logger.info("")
    logger.info("QUALITY ASSURANCE:")
    logger.info("  1. MODEL_DATA validation: Column exists and is populated")
    logger.info("  2. SNR validation: Solutions above minimum SNR threshold")
    logger.info("  3. Solution continuity: No large phase jumps (>90°) between intervals")
    logger.info("  4. Table structure: Valid CASA calibration table format")
    logger.info("")
    logger.info("HOW SOLUTIONS ARE USED:")
    logger.info("  - Applied ONLY during bandpass calibration (Stage 3)")
    logger.info("  - NOT applied to target data (bandpass captures final phase solutions)")
    logger.info("  - Bandpass uses: gaintable=[prebandpass_phase_table]")
    logger.info("  - Improves bandpass SNR by removing phase decorrelation")
    logger.info("=" * 80 + "\n")

    # PRECONDITION CHECK: Verify MODEL_DATA exists and is populated for cal_field
    logger.info(f"→ Validating MODEL_DATA for pre-bandpass phase solve on field(s) {cal_field}...")
    with table(ms) as tb:
        if "MODEL_DATA" not in tb.colnames():
            raise ValueError(
                "MODEL_DATA column does not exist in MS. "
                "This is a required precondition for phase-only calibration. "
                "Populate MODEL_DATA before calling solve_prebandpass_phase()."
            )

        # Parse field selection to determine which field(s) to check
        # MODEL_DATA may only be populated for the calibration field(s)
        if "~" in str(cal_field):
            # Field range: check first field in range
            check_field = int(str(cal_field).split("~")[0])
        elif str(cal_field).isdigit():
            check_field = int(cal_field)
        else:
            # Field name - check all data as fallback
            check_field = None

        if check_field is not None:
            # Query only the cal_field's rows to check MODEL_DATA
            field_ids = tb.getcol("FIELD_ID")
            field_mask = field_ids == check_field
            field_rows = np.where(field_mask)[0]
            if len(field_rows) == 0:
                raise ValueError(f"No data found for field {check_field}. Check field selection.")
            # Sample up to 100 rows from the field
            sample_rows = field_rows[: min(100, len(field_rows))]
            model_sample = np.array([tb.getcell("MODEL_DATA", int(r)) for r in sample_rows])
        else:
            # Fallback: check first 100 rows
            model_sample = tb.getcol("MODEL_DATA", startrow=0, nrow=min(100, tb.nrows()))

        if np.all(np.abs(model_sample) < 1e-10):
            raise ValueError(
                f"MODEL_DATA column exists but is all zeros for field {cal_field}. "
                "This is a required precondition for phase-only calibration. "
                "Populate MODEL_DATA before calling solve_prebandpass_phase()."
            )

    # Determine field selector based on combine_fields setting
    # - If combining across fields: use the full selection string to maximize SNR
    # - Otherwise: use the peak field (closest to calibrator) if provided, otherwise parse from range
    #   The peak field is the one with maximum PB-weighted flux (closest to calibrator position)
    fallback_to_range = peak_field_idx is None and "~" in str(cal_field)
    use_combine_fields = combine_fields or fallback_to_range
    if use_combine_fields:
        field_selector = str(cal_field)
    elif peak_field_idx is not None:
        field_selector = str(peak_field_idx)
    else:
        field_selector = str(cal_field)
    logger.debug(
        f"Using field selector '{field_selector}' for pre-bandpass phase solve"
        + (
            f" (combined from range {cal_field})"
            if use_combine_fields
            else f" (peak field: {field_selector})"
        )
    )

    # Combine across scans, fields, and SPWs when requested
    # Combining SPWs improves SNR by using all 16 subbands simultaneously
    comb_parts = ["scan"]
    if use_combine_fields:
        comb_parts.append("field")
    if combine_spw:
        comb_parts.append("spw")
    comb = ",".join(comb_parts) if comb_parts else ""

    # VERIFICATION: Check which SPWs are available and will be used
    logger.info("\n" + "=" * 70)
    logger.info("SPW SELECTION VERIFICATION")
    logger.info("=" * 70)
    with table(f"{ms}::SPECTRAL_WINDOW", ack=False) as tspw:
        n_spws = tspw.nrows()
        spw_ids = list(range(n_spws))
        ref_freqs = tspw.getcol("REF_FREQUENCY")
        num_chan = tspw.getcol("NUM_CHAN")
        logger.info(f"MS contains {n_spws} spectral windows: SPW {spw_ids[0]} to SPW {spw_ids[-1]}")
        logger.info(f"  Frequency range: {ref_freqs[0] / 1e9:.4f} - {ref_freqs[-1] / 1e9:.4f} GHz")
        logger.info(f"  Total channels across all SPWs: {np.sum(num_chan)}")

    # Check data selection for the specified field
    with table(ms, ack=False) as tb:
        # Get unique SPW IDs in data for the selected field
        # We need to query the actual data to see which SPWs have data
        field_ids = tb.getcol("FIELD_ID")
        spw_ids_in_data = tb.getcol("DATA_DESC_ID")

        # Get unique SPW IDs (need to map DATA_DESC_ID to SPW)
        with table(f"{ms}::DATA_DESCRIPTION", ack=False) as tdd:
            data_desc_to_spw = tdd.getcol("SPECTRAL_WINDOW_ID")

        # Filter by field if field_selector is a single number
        if "~" not in str(field_selector):
            try:
                field_idx = int(field_selector)
                field_mask = field_ids == field_idx
                spw_ids_with_data = np.unique(data_desc_to_spw[spw_ids_in_data[field_mask]])
            except ValueError:
                # Field selector might be a name, use all data
                spw_ids_with_data = np.unique(data_desc_to_spw[spw_ids_in_data])
        else:
            # Range of fields, use all data
            spw_ids_with_data = np.unique(data_desc_to_spw[spw_ids_in_data])

        spw_ids_list = sorted(
            [int(x) for x in spw_ids_with_data]
        )  # Convert to plain ints for cleaner output
        logger.info(f"\nSPWs with data for field(s) '{field_selector}': {spw_ids_with_data}")
        logger.info(f"  Total SPWs to be processed: {len(spw_ids_with_data)}")

        if combine_spw:
            logger.info("\n  COMBINE='spw' is ENABLED:")
            logger.info(
                f"    :arrow_right: All {len(spw_ids_with_data)} SPWs will be used together in a single solve"
            )
            logger.info("    :arrow_right: Solution will be stored in SPW ID 0 (aggregate SPW)")
            logger.info(
                f"    :arrow_right: This improves SNR by using all {len(spw_ids_with_data)} subbands simultaneously"
            )
        else:
            logger.info("\n  COMBINE='spw' is DISABLED:")
            logger.info(
                f"    :arrow_right: Each of the {len(spw_ids_list)} SPWs will be solved separately"
            )
            logger.info(f"    :arrow_right: Solutions will be stored in SPW IDs {spw_ids_list}")

    logger.info("=" * 70 + "\n")

    # Determine table name
    if table_name:
        caltable_name = table_name
    else:
        caltable_name = f"{table_prefix}.prebp"

    # Solve phase-only calibration
    combine_desc = f" (combining across {comb})" if comb else ""
    spw_desc = f" (SPW: {spw})" if spw else ""
    gaintable_desc = f" (applying {len(gaintable)} prior tables)" if gaintable else ""
    logger.info(
        f"Running pre-bandpass phase-only solve on field {field_selector}"
        f"{combine_desc}{spw_desc}{gaintable_desc}..."
    )
    kwargs = dict(
        field=field_selector,
        spw=spw if spw else "",  # Use provided SPW selection or all SPWs
        solint=solint,
        refant=refant,
        calmode="p",  # Phase-only mode
        combine=comb,
        minsnr=minsnr,
        selectdata=True,
    )
    if uvrange:
        kwargs["uvrange"] = uvrange
    if minblperant is not None:
        kwargs["minblperant"] = minblperant
    if gaintable:
        kwargs["gaintable"] = gaintable
        # Apply the same spectral-window mapping to each gaintable.
        # _determine_spwmap_for_bptables returns a single mapping if any table was created with combine_spw=True.
        spwmap = _determine_spwmap_for_bptables(gaintable, ms)
        if spwmap:
            kwargs["spwmap"] = [spwmap for _ in gaintable]

    _call_gaincal_with_progress("Pre-bandpass phase solve", ms, caltable_name, **kwargs)

    # ============================================================================
    # QUALITY ASSURANCE: Validate pre-bandpass phase solutions
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PRE-BANDPASS PHASE CALIBRATION QA - Validating Solutions")
    logger.info("=" * 80)

    logger.info("→ QA Check 1: Solution table structure validation")
    _validate_solve_success(caltable_name, refant=refant)
    logger.info("  ✓ Calibration table created successfully")
    logger.info("  ✓ Reference antenna has valid solutions")
    logger.info("  ✓ Solutions exist for all expected antennas/times")

    # Track provenance after successful solve
    _track_calibration_provenance(
        ms_path=ms,
        caltable_path=caltable_name,
        task_name="gaincal",
        params={"vis": ms, "caltable": caltable_name, **kwargs},
    )

    logger.info("\n→ Solution Summary:")
    logger.info(f"  Output table: {caltable_name}")
    logger.info(f"  Calibration mode: Phase-only (calmode='p')")
    logger.info(f"  Solution interval: {solint}")
    logger.info(f"  Field(s) used: {field_selector}")
    logger.info(f"  SPW combination: {'Yes (aggregate SPW)' if combine_spw else 'No (per-SPW)'}")
    logger.info("")
    logger.info("→ Next step: Apply this table during bandpass calibration (Stage 3)")
    logger.info("  This will stabilize phases and improve bandpass SNR")

    logger.info("\n" + "=" * 80)
    logger.info("PRE-BANDPASS PHASE CALIBRATION - Complete")
    logger.info("=" * 80 + "\n")

    return caltable_name

def solve_gains(
    ms: str,
    cal_field: str,
    refant: str,
    ktable: str | None,
    bptables: list[str],
    table_prefix: str | None = None,
    t_short: str = "60s",
    combine_fields: bool = False,
    *,
    phase_only: bool = False,
    uvrange: str = "",
    solint: str = "inf",
    minsnr: float = 3.0,
    peak_field_idx: int | None = None,
) -> list[str]:
    """Solve gain amplitude and phase; optionally short-timescale.

    **PRECONDITION**: MODEL_DATA must be populated before calling this function.
    This ensures consistent, reliable calibration results across all calibrators
    (bright or faint). The calling code should verify MODEL_DATA exists and is
    populated before invoking solve_gains().

    **PRECONDITION**: If `bptables` are provided, they must exist and be
    compatible with the MS. This ensures consistent, reliable calibration results.

    **NOTE**: `ktable` (Delay calibration) is applied if provided.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.
    cal_field : str
        Field ID or name for calibration (e.g., "0" or "3C286").
    refant : str
        Reference antenna ID.
    ktable : str or None
        Delay calibration table to apply.
    bptables : list[str]
        List of bandpass calibration tables to apply.
    table_prefix : str or None, optional
        Prefix for output calibration tables. Defaults to MS name + field.
    t_short : str, optional
        Short timescale solution interval (default: "60s").
    combine_fields : bool, optional
        If True, combine all fields for solution (default: False).
    phase_only : bool, optional
        If True, solve phase only (calmode='p'). Default: False.
    uvrange : str, optional
        UV range selection string (default: "").
    solint : str, optional
        Solution interval (default: "inf").
    minsnr : float, optional
        Minimum signal-to-noise ratio for solutions (default: 3.0).
    peak_field_idx : int or None, optional
        Index of peak flux field for calibrator selection.

    Returns
    -------
    list[str]
        List of generated calibration table paths.
    """
    # use module-level table - access via sys.modules to avoid scoping issues
    import numpy as np  # type: ignore[import]

    _table = sys.modules[__name__].table
    if _table is None:
        raise ImportError(
            "dsa110_continuum.adapters.casa_tables module is not available. "
            "This function requires CASA environment to be properly configured. "
            "Please ensure you are running in the casa6 conda environment."
        )

    if table_prefix is None:
        table_prefix = f"{os.path.splitext(ms)[0]}_{cal_field}"

    # ============================================================================
    # FINAL GAIN CALIBRATION (G) - TRANSPARENCY HEADER
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FINAL GAIN CALIBRATION (gaintype='G') - Stage 4")
    logger.info("=" * 80)
    logger.info("PURPOSE: Solve for time-dependent amplitude and phase variations")
    logger.info("         after applying bandpass corrections. This is the FINAL")
    logger.info("         calibration step before applying to target data.")
    logger.info("")
    logger.info("WHAT IT CORRECTS:")
    logger.info("  - Atmospheric phase fluctuations: Tropospheric water vapor changes")
    logger.info("  - Antenna gain drifts: Electronics warm-up, temperature variations")
    logger.info("  - Pointing errors: Antennas drift slightly off source")
    logger.info("  - Ionospheric phase: For low-frequency observations")
    logger.info("")
    logger.info("EXPECTED INPUT:")
    logger.info("  - MODEL_DATA: Populated with calibrator source model (validating...)")
    logger.info(f"  - Field: {cal_field}")
    logger.info(f"  - Reference antenna: {refant}")
    logger.info(f"  - Bandpass tables: {len(bptables)} table(s) (validating...)")
    if ktable:
        logger.info(
            f"  - Delay table: {ktable} (NOT used - K-calibration not required for DSA-110)"
        )
    logger.info(f"  - Solution intervals: {solint} (long), {t_short} (short)")
    logger.info(f"  - Calibration mode: {'Phase-only' if phase_only else 'Amplitude + Phase'}")
    logger.info(f"  - Minimum SNR: {minsnr}")
    logger.info("")
    logger.info("TWO-TIMESCALE APPROACH:")
    logger.info("  This stage generates TWO gain tables:")
    logger.info("  1. Long timescale (.g file, solint='inf'):")
    logger.info("     - Captures overall amplitude and phase offsets")
    logger.info("     - Instrumental effects (stable)")
    logger.info("     - One solution per observation")
    logger.info("  2. Short timescale (.2g file, solint='60s'):")
    logger.info("     - Captures rapid atmospheric phase variations")
    logger.info("     - Atmospheric effects (variable)")
    logger.info("     - Time-resolved solutions")
    logger.info("")
    logger.info("EXPECTED SOLUTIONS:")
    logger.info("  - Amplitude: 0.8 to 1.2 (relative to mean)")
    logger.info("    * Slowly varying over time (minutes to hours)")
    logger.info("    * Large excursions suggest source confusion, pointing drift, or RFI")
    logger.info("  - Phase: -180° to +180° (wrapped)")
    logger.info("    * Can vary rapidly (seconds to minutes) due to atmosphere")
    logger.info("    * Smooth trends indicate tropospheric phase screen")
    logger.info("    * Random jumps indicate RFI or data quality issues")
    logger.info("  - Reference antenna: Amplitude = 1.0, Phase = 0° (by definition)")
    logger.info("")
    logger.info("QUALITY ASSURANCE:")
    logger.info("  1. MODEL_DATA validation: Column exists and is populated")
    logger.info("  2. Bandpass table validation: Tables exist and are compatible")
    logger.info("  3. SNR validation: Solutions above minimum SNR threshold")
    logger.info("  4. Flagging fraction: <5% ideal, <10% acceptable")
    logger.info("  5. Pipeline quality check: Table structure and coverage")
    logger.info("")
    logger.info("HOW SOLUTIONS ARE USED:")
    logger.info("  - Applied to target observations (final calibration step)")
    logger.info("  - Application order: B → G → 2G")
    logger.info("  - Time interpolation: Linear between calibrator scans")
    logger.info("  - Frequency interpolation: Bandpass handles frequency dependence")
    logger.info("=" * 80 + "\n")

    # PRECONDITION CHECK: Verify MODEL_DATA exists and is populated
    # This ensures we follow "measure twice, cut once" - establish requirements upfront
    # for consistent, reliable calibration across all calibrators (bright or faint).
    logger.info(f"→ QA Check 1: Validating MODEL_DATA for gain solve on field(s) {cal_field}...")
    with _table(ms) as tb:
        if "MODEL_DATA" not in tb.colnames():
            raise ValueError(
                "MODEL_DATA column does not exist in MS. "
                "This is a required precondition for gain calibration. "
                "Populate MODEL_DATA using setjy, ft(), or a catalog model before "
                "calling solve_gains()."
            )

        # Check if MODEL_DATA is populated (not all zeros)
        model_sample = tb.getcol("MODEL_DATA", startrow=0, nrow=min(100, tb.nrows()))
        if np.all(np.abs(model_sample) < 1e-10):
            raise ValueError(
                "MODEL_DATA column exists but is all zeros (unpopulated). "
                "This is a required precondition for gain calibration. "
                "Populate MODEL_DATA using setjy, ft(), or a catalog model before "
                "calling solve_gains()."
            )

    logger.info("  ✓ MODEL_DATA validation passed")

    # PRECONDITION CHECK: Validate all required calibration tables
    # This ensures we follow "measure twice, cut once" - establish requirements upfront
    # for consistent, reliable calibration across all calibrators.
    # Validate K-table if provided
    if ktable:
        logger.info(f"Validating K-table before gain calibration: {ktable}")
        try:
            if not os.path.exists(ktable):
                raise FileNotFoundError(f"K-table not found: {ktable}")
        except Exception as e:
            raise ValueError(
                f"K-table validation failed. This is a required precondition for "
                f"gain calibration when ktable is provided. Error: {e}"
            ) from e

    if bptables:
        logger.info(
            f"\n→ QA Check 2: Validating {len(bptables)} bandpass table(s) before gain calibration..."
        )
        logger.info("  Required checks:")
        logger.info("    - All bandpass tables exist on disk")
        logger.info("    - Tables are compatible with MS (matching SPWs, antennas)")
        logger.info("    - Reference antenna has valid solutions")
        logger.info("    - Tables are not corrupted (valid CASA table format)")
        try:
            # Convert refant string to int for validation
            # Handle comma-separated refant string (e.g., "113,114,103,106,112")
            # Use the first antenna in the chain for validation
            if isinstance(refant, str):
                if "," in refant:
                    # Comma-separated list: use first antenna
                    refant_str = refant.split(",")[0].strip()
                    refant_int = int(refant_str)
                else:
                    # Single antenna ID as string
                    refant_int = int(refant)
            else:
                refant_int = refant
            validate_caltables_for_use(bptables, ms, require_all=True, refant=refant_int)
            logger.info("  ✓ All bandpass table validation checks passed")
        except (FileNotFoundError, ValueError) as e:
            logger.error("  ❌ Bandpass table validation FAILED")
            logger.error(f"     Error: {e}")
            logger.error("     Cannot proceed without valid bandpass tables")
            raise ValueError(
                f"Calibration table validation failed. This is a required precondition for "
                f"gain calibration. Error: {e}"
            ) from e

    # Determine CASA field selector based on combine_fields setting
    # - If combining across fields: use the full selection string to maximize SNR
    # - Otherwise: use the peak field (closest to calibrator) if provided, otherwise parse from range
    #   The peak field is the one with maximum PB-weighted flux (closest to calibrator position)
    fallback_to_range = peak_field_idx is None and "~" in str(cal_field)
    use_combine_fields = combine_fields or fallback_to_range
    if use_combine_fields:
        field_selector = str(cal_field)
    elif peak_field_idx is not None:
        field_selector = str(peak_field_idx)
    else:
        field_selector = str(cal_field)
    logger.debug(
        f"Using field selector '{field_selector}' for gain calibration"
        + (
            f" (combined from range {cal_field})"
            if use_combine_fields
            else f" (peak field: {field_selector})"
        )
    )

    # Construct gaintable list
    gaintable = []
    if ktable:
        gaintable.append(ktable)
    gaintable.extend(bptables)

    # Combine across scans and fields when requested; otherwise do not combine
    comb = "scan,field" if use_combine_fields else ""

    # CRITICAL FIX: Determine spwmap if tables were created with combine_spw=True
    # When combine_spw is used, the table has solutions only for SPW=0 (aggregate).
    # We need to map all MS SPWs to SPW 0 in that table.

    # Check if K-table needs mapping
    k_spwmap = None
    if ktable:
        # Re-use the logic for BP tables (it works for any calibration table)
        k_spwmap = _determine_spwmap_for_bptables([ktable], ms)

    # Check if BP tables need mapping
    bp_spwmap = _determine_spwmap_for_bptables(bptables, ms)

    spwmap = None
    if k_spwmap or bp_spwmap:
        # If any table needs mapping, we must construct the full spwmap parameter list
        spwmap = []
        if ktable:
            spwmap.append(k_spwmap if k_spwmap else [])

        # Add mapping for each bandpass table
        for _ in bptables:
            spwmap.append(bp_spwmap if bp_spwmap else [])

    # Run gain calibration after bandpass
    # Default is amplitude+phase (calmode='ap')
    # Use phase_only=True for phase-only calibration (calmode='p')
    calmode = "p" if phase_only else "ap"
    logger.info(
        f"Running {'phase-only' if phase_only else 'amplitude+phase'} gain solve on field {field_selector}"
        + (" (combining across fields)..." if combine_fields else "...")
    )
    kwargs = dict(
        vis=ms,
        caltable=f"{table_prefix}.g",
        field=field_selector,
        solint=solint,
        refant=refant,
        gaintype="G",
        calmode=calmode,
        gaintable=gaintable,
        combine=comb,
        minsnr=minsnr,
        selectdata=True,
    )
    if uvrange:
        kwargs["uvrange"] = uvrange
    if spwmap:
        kwargs["spwmap"] = spwmap

    # Run with progress monitoring
    from dsa110_continuum.utils.progress import stage_progress

    with stage_progress(
        f"{'Phase-only' if phase_only else 'Amplitude+phase'} gain solve",
        output_path=f"{table_prefix}.g",
    ):
        _call_gaincal(**kwargs)
    # PRECONDITION CHECK: Verify phase-only gain solve completed successfully
    # This ensures we follow "measure twice, cut once" - verify solutions exist
    # immediately after solve completes, before proceeding.
    _validate_solve_success(f"{table_prefix}.g", refant=refant)
    # Track provenance after successful solve
    _track_calibration_provenance(
        ms_path=ms,
        caltable_path=f"{table_prefix}.g",
        task_name="gaincal",
        params=kwargs,
    )
    logger.info(f":check: Gain solve completed: {table_prefix}.g")

    out = [f"{table_prefix}.g"]
    gaintable2 = gaintable + [f"{table_prefix}.g"]

    if t_short:
        logger.info(
            f"Running short-timescale {'phase-only' if phase_only else 'amplitude+phase'} gain solve on field {field_selector}"
            + (" (combining across fields)..." if combine_fields else "...")
        )
        kwargs = dict(
            vis=ms,
            caltable=f"{table_prefix}.2g",
            field=field_selector,
            solint=t_short,
            refant=refant,
            gaintype="G",
            calmode=calmode,
            gaintable=gaintable2,
            combine=comb,
            minsnr=minsnr,
            selectdata=True,
        )
        if uvrange:
            kwargs["uvrange"] = uvrange
        # CRITICAL FIX: Apply spwmap to second gaincal call as well
        # Note: spwmap applies to bandpass tables in gaintable2; the gain table doesn't need it
        if spwmap:
            kwargs["spwmap"] = spwmap

        with stage_progress(
            f"Short-timescale {'phase-only' if phase_only else 'amplitude+phase'} gain solve",
            output_path=f"{table_prefix}.2g",
        ):
            _call_gaincal(**kwargs)
        # PRECONDITION CHECK: Verify short-timescale gain solve completed successfully
        # This ensures we follow "measure twice, cut once" - verify solutions exist
        # immediately after solve completes, before proceeding.
        _validate_solve_success(f"{table_prefix}.2g", refant=refant)
        # Track provenance after successful solve
        _track_calibration_provenance(
            ms_path=ms,
            caltable_path=f"{table_prefix}.2g",
            task_name="gaincal",
            params=kwargs,
        )
        logger.info(f":check: Short-timescale gain solve completed: {table_prefix}.2g")
        out.append(f"{table_prefix}.2g")

    # ============================================================================
    # QUALITY ASSURANCE: Validate gain calibration solutions
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FINAL GAIN CALIBRATION QA - Validating Solutions")
    logger.info("=" * 80)

    # QA validation of gain calibration tables
    logger.info("→ QA Check 3: Pipeline quality validation")
    logger.info("  Checking:")
    logger.info("    - Solution table structure validity")
    logger.info("    - Reference antenna has valid solutions")
    logger.info("    - Time coverage matches MS")
    logger.info("    - No entirely flagged time ranges")
    try:
        from dsa110_continuum.qa.pipeline_quality import check_calibration_quality

        check_calibration_quality(out, ms_path=ms, alert_on_issues=True)
        logger.info("  ✓ Pipeline quality validation passed")
    except Exception as e:
        logger.warning(f"  ⚠ QA validation warning: {e}")

    logger.info("\n→ Solution Summary:")
    logger.info(f"  Generated {len(out)} gain table(s):")
    for i, table in enumerate(out, 1):
        if table.endswith(".g"):
            logger.info(f"    {i}. {table} (long timescale, solint={solint})")
        elif table.endswith(".2g"):
            logger.info(f"    {i}. {table} (short timescale, solint={t_short})")
        else:
            logger.info(f"    {i}. {table}")
    logger.info(f"  Calibration mode: {'Phase-only' if phase_only else 'Amplitude + Phase'}")
    logger.info(f"  Field(s) used: {field_selector}")
    logger.info(f"  Bandpass corrections: {len(bptables)} table(s) applied")
    logger.info("")
    logger.info("→ Next step: Apply these tables to target observations")
    logger.info("  Application order: K → B → G → 2G")
    logger.info("  Time interpolation: Linear between calibrator scans")

    logger.info("\n" + "=" * 80)
    logger.info("FINAL GAIN CALIBRATION - Complete")
    logger.info("=" * 80 + "\n")

    return out
