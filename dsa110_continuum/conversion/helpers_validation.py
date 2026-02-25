"""Validation helper functions for conversion."""

import logging

import numpy as np

# Ensure CASAPATH is set before importing CASA modules
from dsa110_contimg.common.utils.casa_init import ensure_casa_path

ensure_casa_path()

# Use the shared patchable table symbol from conversion.helpers to make unit tests simpler
import dsa110_contimg.core.conversion.helpers as _helpers

# Use canonical angular_separation with numba→astropy fallback chain
from dsa110_contimg.core.conversion.helpers_coordinates import angular_separation

logger = logging.getLogger("dsa110_contimg.conversion.helpers")


def validate_ms_frequency_order(ms_path: str) -> None:
    """Verify MS has ascending frequency order across all spectral windows.

    This is critical for DSA-110 because subbands come in DESCENDING order
    (sb00=highest freq, sb15=lowest freq) but CASA imaging requires ASCENDING
    order. If frequencies are out of order, MFS imaging will produce fringes
    and bandpass calibration will fail.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set

    Raises
    ------
    RuntimeError
        If frequency order is incorrect

    """
    try:
        with _helpers.table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True) as spw:
            chan_freq = spw.getcol("CHAN_FREQ")  # Shape: (nspw, nchan)

            # Check each SPW has ascending frequency order
            for ispw in range(chan_freq.shape[0]):
                freqs = chan_freq[ispw, :]
                if freqs.size > 1 and not np.all(freqs[1:] >= freqs[:-1]):
                    raise RuntimeError(
                        f"frequencies are in DESCENDING order in {ms_path} (SPW {ispw}). "
                        f"Frequencies: {freqs[:3]}...{freqs[-3:]} Hz. "
                        f"This will cause MFS imaging artifacts and calibration failures."
                    )

            # If multiple SPWs, check they are in ascending order too
            if chan_freq.shape[0] > 1:
                spw_start_freqs = chan_freq[:, 0]  # First channel of each SPW
                if not np.all(spw_start_freqs[1:] >= spw_start_freqs[:-1]):
                    raise RuntimeError(
                        f"SPWs have incorrect frequency order in {ms_path}. "
                        f"SPW start frequencies: {spw_start_freqs} Hz. "
                        f"This will cause MFS imaging artifacts."
                    )

            logger.info(
                f":check: Frequency order validation passed: {chan_freq.shape[0]} SPW(s), "
                f"range {chan_freq.min() / 1e6:.1f}-{chan_freq.max() / 1e6:.1f} MHz"
            )
    except Exception as e:
        if "incorrect frequency order" in str(e) or "frequencies are in DESCENDING order" in str(e):
            raise  # Re-raise our validation errors
        else:
            logger.warning(f"Frequency order validation failed (non-fatal): {e}")


def validate_phase_center_coherence(ms_path: str, tolerance_arcsec: float = 1.0) -> None:
    """Verify all subbands in MS have coherent phase centers.

    This checks that all spectral windows (former subbands) have phase centers
    within tolerance of each other. Incoherent phase centers cause imaging
    artifacts and calibration failures.

    NOTE: With time-dependent phasing (phase centers tracking LST), multiple
    phase centers with large separations are EXPECTED and correct. This
    validation will skip the check if time-dependent phasing is detected.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    tolerance_arcsec :
        Maximum allowed separation between phase centers (arcsec)

    Raises
    ------
    RuntimeError
        If phase centers are incoherent beyond tolerance

    """
    try:
        with _helpers.table(f"{ms_path}::FIELD", readonly=True) as field_table:
            if field_table.nrows() == 0:
                logger.warning(f"No fields found in MS: {ms_path}")
                return

            phase_dirs = field_table.getcol("PHASE_DIR")  # Shape: (nfield, npoly, 2)

            if phase_dirs.shape[0] > 1:
                # Check if this looks like time-dependent phasing
                # Time-dependent phasing: phase centers should track LST (RA changes with time)
                # Get observation time range from main table
                try:
                    with _helpers.table(ms_path, readonly=True, ack=False) as main_table:
                        if main_table.nrows() > 0:
                            times = main_table.getcol("TIME")
                            if times.size > 0:
                                time_span_seconds = float(np.max(times) - np.min(times))
                                time_span_days = time_span_seconds / 86400.0

                                # Calculate expected LST change (15°/hour = 0.25°/min)
                                # Over time_span, LST changes by: 15° × time_span_hours
                                time_span_hours = time_span_days * 24.0
                                expected_lst_change_deg = 15.0 * time_span_hours
                                expected_lst_change_arcsec = expected_lst_change_deg * 3600.0

                                # Check if phase center separation matches expected LST tracking
                                ref_ra = phase_dirs[0, 0, 0]  # Reference RA (radians)
                                ref_dec = phase_dirs[0, 0, 1]  # Reference Dec (radians)

                                max_separation_rad = 0.0
                                for i in range(1, phase_dirs.shape[0]):
                                    ra = phase_dirs[i, 0, 0]
                                    dec = phase_dirs[i, 0, 1]

                                    # Calculate angular separation
                                    separation_rad = angular_separation(ref_ra, ref_dec, ra, dec)
                                    max_separation_rad = max(max_separation_rad, separation_rad)

                                max_separation_arcsec = np.rad2deg(max_separation_rad) * 3600

                                # If separation is close to expected LST change, this is time-dependent phasing
                                # Allow 20% tolerance for time-dependent phasing
                                if (
                                    max_separation_arcsec > 60.0  # More than 1 arcmin
                                    and max_separation_arcsec < expected_lst_change_arcsec * 1.2
                                    and max_separation_arcsec > expected_lst_change_arcsec * 0.8
                                ):
                                    logger.info(
                                        f":check: Time-dependent phase centers detected: "
                                        f"{phase_dirs.shape[0]} field(s), "
                                        f"max separation {max_separation_arcsec:.2f} arcsec "
                                        f"(expected LST change: {expected_lst_change_arcsec:.2f} arcsec). "
                                        f"This is correct for meridian-tracking phasing."
                                    )
                                    return  # Skip strict coherence check for time-dependent phasing
                except (KeyError, IndexError, TypeError, RuntimeError):
                    # If we can't determine time span, fall through to normal check
                    # KeyError: missing columns, IndexError: array access, RuntimeError: CASA errors
                    pass

                # Multiple fields - check they are coherent (for fixed phase centers)
                ref_ra = phase_dirs[0, 0, 0]  # Reference RA (radians)
                ref_dec = phase_dirs[0, 0, 1]  # Reference Dec (radians)

                max_separation_rad = 0.0
                for i in range(1, phase_dirs.shape[0]):
                    ra = phase_dirs[i, 0, 0]
                    dec = phase_dirs[i, 0, 1]

                    # Calculate angular separation
                    separation_rad = angular_separation(ref_ra, ref_dec, ra, dec)
                    max_separation_rad = max(max_separation_rad, separation_rad)

                max_separation_arcsec = np.rad2deg(max_separation_rad) * 3600

                if max_separation_arcsec > tolerance_arcsec:
                    # Check if this might be time-dependent phasing that wasn't detected
                    # If separation is large (> 60 arcsec), it's likely time-dependent phasing
                    if max_separation_arcsec > 60.0:
                        raise RuntimeError(
                            f"Phase centers are incoherent in {ms_path}. "
                            f"Maximum separation: {max_separation_arcsec:.2f} arcsec "
                            f"(tolerance: {tolerance_arcsec:.2f} arcsec). "
                            f"NOTE: Large separations (>60 arcsec) are EXPECTED for time-dependent phasing "
                            f"(meridian-tracking, RA=LST). If this is meridian-tracking phasing, this is correct. "
                            f"See conversion/README.md for details."
                        )
                    else:
                        raise RuntimeError(
                            f"Phase centers are incoherent in {ms_path}. "
                            f"Maximum separation: {max_separation_arcsec:.2f} arcsec "
                            f"(tolerance: {tolerance_arcsec:.2f} arcsec). "
                            f"This may cause imaging artifacts. "
                            f"If separation is large (>60 arcsec), this may be expected time-dependent phasing. "
                            f"See conversion/README.md for details."
                        )

                logger.info(
                    f":check: Phase center coherence validated: {phase_dirs.shape[0]} field(s), "
                    f"max separation {max_separation_arcsec:.2f} arcsec"
                )
            else:
                logger.info(":check: Single field MS - phase center coherence OK")

    except Exception as e:
        if "incoherent" in str(e):
            raise  # Re-raise our validation errors
        else:
            logger.warning(f"Phase center coherence validation failed (non-fatal): {e}")


def validate_uvw_precision(ms_path: str, tolerance_lambda: float = 0.1) -> None:
    """Validate UVW coordinate precision to prevent calibration decorrelation.

    This checks that UVW coordinates are accurate enough for calibration by
    comparing computed UVW values against expected values from antenna positions.
    Inaccurate UVW coordinates cause phase decorrelation and flagged solutions.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    tolerance_lambda :
        Maximum allowed UVW error in wavelengths (default: 0.1λ)

    Raises
    ------
    RuntimeError
        If UVW errors exceed tolerance

    """
    try:
        # Get observation parameters
        with _helpers.table(f"{ms_path}::OBSERVATION", readonly=True) as obs_table:
            if obs_table.nrows() == 0:
                logger.warning(f"No observation info in MS: {ms_path}")
                return

        # Get reference frequency for wavelength calculation
        with _helpers.table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True) as spw_table:
            ref_freqs = spw_table.getcol("REF_FREQUENCY")
            ref_freq_hz = float(np.median(ref_freqs))
            wavelength_m = 2.998e8 / ref_freq_hz  # c / freq

        # Sample UVW coordinates from main table
        with _helpers.table(ms_path, readonly=True) as tb:
            if tb.nrows() == 0:
                raise RuntimeError(f"MS has no data rows: {ms_path}")

            # Sample subset for performance (check every 100th row)
            n_rows = tb.nrows()
            sample_rows = list(range(0, n_rows, max(1, n_rows // 1000)))[:1000]

            uvw_data = tb.getcol("UVW", startrow=sample_rows[0], nrow=len(sample_rows))
            tb.getcol("ANTENNA1", startrow=sample_rows[0], nrow=len(sample_rows))
            tb.getcol("ANTENNA2", startrow=sample_rows[0], nrow=len(sample_rows))
            tb.getcol("TIME", startrow=sample_rows[0], nrow=len(sample_rows))

        # Check for obvious UVW coordinate problems
        uvw_data[:, 0]  # U coordinates
        uvw_data[:, 1]  # V coordinates
        uvw_data[:, 2]  # W coordinates

        # Detect unreasonably large UVW values (> 100km indicates error)
        max_reasonable_uvw_m = 100e3  # 100 km
        if np.any(np.abs(uvw_data) > max_reasonable_uvw_m):
            raise RuntimeError(
                f"UVW coordinates contain unreasonably large values (>{max_reasonable_uvw_m / 1000:.0f}km) "
                f"in {ms_path}. Max |UVW|: {np.max(np.abs(uvw_data)) / 1000:.1f}km. "
                f"This indicates UVW computation errors that will cause calibration failures."
            )

        # Check for all-zero UVW (indicates computation failure)
        if np.all(np.abs(uvw_data) < 1e-10):
            raise RuntimeError(
                f"All UVW coordinates are zero in {ms_path}. "
                f"This indicates UVW computation failed and will cause calibration failures."
            )

        # Statistical checks for UVW distribution
        uvw_magnitude = np.sqrt(np.sum(uvw_data**2, axis=1))
        median_uvw_m = float(np.median(uvw_magnitude))
        max_uvw_m = float(np.max(uvw_magnitude))

        # For DSA-110: expect baseline lengths from ~10m to ~2500m
        expected_min_baseline_m = 5.0  # Minimum expected baseline
        expected_max_baseline_m = 3210.0  # Maximum expected baseline

        if median_uvw_m < expected_min_baseline_m:
            logger.warning(
                f"UVW coordinates seem too small in {ms_path}. "
                f"Median baseline: {median_uvw_m:.1f}m (expected >{expected_min_baseline_m:.1f}m). "
                f"This may indicate UVW scaling errors."
            )

        if max_uvw_m > expected_max_baseline_m:
            logger.warning(
                f"UVW coordinates seem too large in {ms_path}. "
                f"Max baseline: {max_uvw_m:.1f}m (expected <{expected_max_baseline_m:.1f}m). "
                f"This may indicate UVW scaling errors."
            )

        # Convert tolerance to meters
        tolerance_m = tolerance_lambda * wavelength_m

        logger.info(
            f":check: UVW coordinate validation passed: "
            f"median baseline {median_uvw_m:.1f}m, max {max_uvw_m:.1f}m "
            f"(λ={wavelength_m:.2f}m, tolerance={tolerance_m:.3f}m)"
        )

    except Exception as e:
        if "UVW coordinates" in str(e):
            raise  # Re-raise our validation errors
        else:
            logger.warning(f"UVW coordinate validation failed (non-fatal): {e}")


def validate_antenna_positions(ms_path: str, position_tolerance_m: float = 0.05) -> None:
    """Validate antenna positions are accurate enough for calibration.

    This checks that antenna positions in the MS match expected DSA-110 positions
    within calibration tolerance. Position errors cause decorrelation and flagging.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    position_tolerance_m :
        Maximum allowed position error in meters (default: 5cm)

    Raises
    ------
    RuntimeError
        If antenna positions have excessive errors

    """
    try:
        # Get antenna positions from MS
        with _helpers.table(f"{ms_path}::ANTENNA", readonly=True) as ant_table:
            ms_positions = ant_table.getcol("POSITION")  # Shape: (nant, 3) ITRF meters
            ant_names = ant_table.getcol("NAME")
            n_antennas = len(ant_names)

        if n_antennas == 0:
            raise RuntimeError(f"No antennas found in MS: {ms_path}")

        # Load reference DSA-110 positions
        try:
            from dsa110_contimg.common.utils.antpos_local import get_itrf

            ref_df = get_itrf(latlon_center=None)

            # Convert reference positions to same format as MS
            ref_positions = np.array(
                [
                    ref_df["x_m"].values,
                    ref_df["y_m"].values,
                    ref_df["z_m"].values,
                ]
            ).T  # Shape: (nant, 3)

        except Exception as e:
            logger.warning(f"Could not load reference antenna positions: {e}")
            # Can't validate without reference - just check for obvious problems
            position_magnitudes = np.sqrt(np.sum(ms_positions**2, axis=1))

            # DSA-110 is near OVRO: expect positions around Earth radius from center
            earth_radius_m = 6.371e6
            expected_min_radius = earth_radius_m - 10e3  # 10km below Earth center
            expected_max_radius = earth_radius_m + 10e3  # 10km above Earth surface

            if np.any(position_magnitudes < expected_min_radius):
                raise RuntimeError(
                    f"Antenna positions too close to Earth center in {ms_path}. "
                    f"Min radius: {np.min(position_magnitudes) / 1000:.1f}km "
                    f"(expected >{expected_min_radius / 1000:.1f}km). "
                    f"This indicates position coordinate errors."
                )

            if np.any(position_magnitudes > expected_max_radius):
                raise RuntimeError(
                    f"Antenna positions too far from Earth center in {ms_path}. "
                    f"Max radius: {np.max(position_magnitudes) / 1000:.1f}km "
                    f"(expected <{expected_max_radius / 1000:.1f}km). "
                    f"This indicates position coordinate errors."
                )

            logger.info(f":check: Basic antenna position validation passed: {n_antennas} antennas")
            return

        # Compare MS positions with reference positions
        if ms_positions.shape[0] != ref_positions.shape[0]:
            logger.warning(
                f"Antenna count mismatch: MS has {ms_positions.shape[0]}, "
                f"reference has {ref_positions.shape[0]}. Using available antennas."
            )
            n_compare = min(ms_positions.shape[0], ref_positions.shape[0])
            ms_positions = ms_positions[:n_compare, :]
            ref_positions = ref_positions[:n_compare, :]

        # Calculate position differences
        position_errors = ms_positions - ref_positions
        position_error_magnitudes = np.sqrt(np.sum(position_errors**2, axis=1))

        max_error_m = float(np.max(position_error_magnitudes))
        float(np.median(position_error_magnitudes))
        rms_error_m = float(np.sqrt(np.mean(position_error_magnitudes**2)))

        # Check if errors exceed tolerance
        n_bad_antennas = np.sum(position_error_magnitudes > position_tolerance_m)

        if n_bad_antennas > 0:
            bad_indices = np.where(position_error_magnitudes > position_tolerance_m)[0]
            error_summary = ", ".join(
                [
                    f"ant{i}:{position_error_magnitudes[i] * 100:.1f}cm"
                    for i in bad_indices[:5]  # Show first 5
                ]
            )
            if len(bad_indices) > 5:
                error_summary += f" (and {len(bad_indices) - 5} more)"

            raise RuntimeError(
                f"Antenna position errors exceed tolerance in {ms_path}. "
                f"{n_bad_antennas}/{len(position_error_magnitudes)} antennas have errors "
                f">{position_tolerance_m * 100:.1f}cm (tolerance for calibration). "
                f"Errors: {error_summary}. Max error: {max_error_m * 100:.1f}cm. "
                f"This will cause decorrelation and flagged calibration solutions."
            )

        logger.info(
            f":check: Antenna position validation passed: {n_antennas} antennas, "
            f"max error {max_error_m * 100:.1f}cm, RMS {rms_error_m * 100:.1f}cm "
            f"(tolerance {position_tolerance_m * 100:.1f}cm)"
        )

    except Exception as e:
        if "position errors exceed tolerance" in str(e):
            raise  # Re-raise our validation errors
        else:
            logger.warning(f"Antenna position validation failed (non-fatal): {e}")


def validate_model_data_quality(
    ms_path: str,
    field_id: int | None = None,
    min_flux_jy: float = 0.1,
    max_flux_jy: float = 1000.0,
) -> None:
    """Validate MODEL_DATA quality for calibrator sources.

    This checks that MODEL_DATA contains reasonable flux values and structure
    for calibration. Poor calibrator models cause solution divergence and flagging.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    field_id :
        Optional field ID to check (if None, checks all fields)
    min_flux_jy :
        Minimum expected flux density in Jy
    max_flux_jy :
        Maximum expected flux density in Jy

    Raises
    ------
    RuntimeError
        If MODEL_DATA has quality issues

    """
    try:
        with _helpers.table(ms_path, readonly=True) as tb:
            if "MODEL_DATA" not in tb.colnames():
                raise RuntimeError(
                    f"MODEL_DATA column does not exist in {ms_path}. "
                    f"This is required for calibration and must be populated before solving."
                )

            # Get field selection
            field_col = tb.getcol("FIELD_ID")
            if field_id is not None:
                field_mask = field_col == field_id
                if not np.any(field_mask):
                    raise RuntimeError(f"Field ID {field_id} not found in MS: {ms_path}")
            else:
                field_mask = np.ones(len(field_col), dtype=bool)

            # Sample MODEL_DATA for the selected field(s)
            n_selected = np.sum(field_mask)
            sample_size = min(1000, n_selected)  # Sample for performance
            selected_indices = np.where(field_mask)[0]
            sample_indices = selected_indices[:: max(1, len(selected_indices) // sample_size)]

            model_sample = tb.getcol(
                "MODEL_DATA", startrow=int(sample_indices[0]), nrow=len(sample_indices)
            )

            # Check for all-zero model
            if np.all(np.abs(model_sample) < 1e-12):
                raise RuntimeError(
                    f"MODEL_DATA is all zeros in {ms_path}. "
                    f"Calibrator source models must be populated before calibration. "
                    f"Use setjy, ft(), or manual model assignment."
                )

            # Calculate flux statistics
            # MODEL_DATA shape: (nchan, npol, nrow) or similar
            model_amplitudes = np.abs(model_sample)

            # Get Stokes I equivalent (average polarizations for rough flux estimate)
            if model_amplitudes.ndim == 3:
                # Average across polarizations and frequencies for flux estimate
                stokes_i_approx = np.mean(model_amplitudes, axis=(0, 1))
            else:
                stokes_i_approx = np.mean(model_amplitudes, axis=0)

            median_flux = float(np.median(stokes_i_approx))
            max_flux = float(np.max(stokes_i_approx))

            # Check flux range
            if median_flux < min_flux_jy:
                logger.warning(
                    f"MODEL_DATA flux seems low in {ms_path}. "
                    f"Median flux: {median_flux:.3f} Jy (expected >{min_flux_jy:.1f} Jy). "
                    f"Weak calibrator models may cause flagged solutions."
                )

            if max_flux > max_flux_jy:
                raise RuntimeError(
                    f"MODEL_DATA flux unreasonably high in {ms_path}. "
                    f"Max flux: {max_flux:.1f} Jy (expected <{max_flux_jy:.1f} Jy). "
                    f"This indicates incorrect calibrator model scaling."
                )

            # Check for NaN or infinite values
            if not np.all(np.isfinite(model_sample)):
                raise RuntimeError(
                    f"MODEL_DATA contains NaN or infinite values in {ms_path}. "
                    f"This will cause calibration failures."
                )

            # Check model structure consistency across channels
            # For point sources, flux should be relatively flat across frequency
            # For resolved sources, may vary but should not have sharp discontinuities
            if model_amplitudes.ndim >= 2:
                channel_fluxes = np.mean(model_amplitudes, axis=-1)  # Average over baselines
                if channel_fluxes.size > 1:
                    # Look for sudden flux jumps between channels (>50% change)
                    flux_ratios = channel_fluxes[1:] / (channel_fluxes[:-1] + 1e-12)
                    large_jumps = np.sum((flux_ratios > 2.0) | (flux_ratios < 0.5))

                    if large_jumps > len(flux_ratios) * 0.1:  # >10% of channels have jumps
                        logger.warning(
                            f"MODEL_DATA has discontinuous flux structure in {ms_path}. "
                            f"{large_jumps}/{len(flux_ratios)} channel pairs have >50% flux changes. "
                            f"This may indicate incorrect calibrator model or frequency mapping."
                        )

            field_desc = f"field {field_id}" if field_id is not None else "all fields"
            logger.info(
                f":check: MODEL_DATA validation passed for {field_desc}: "
                f"median flux {median_flux:.3f} Jy, max {max_flux:.3f} Jy"
            )

    except Exception as e:
        if "MODEL_DATA" in str(e) and (
            "does not exist" in str(e)
            or "all zeros" in str(e)
            or "unreasonably high" in str(e)
            or "NaN" in str(e)
        ):
            raise  # Re-raise our validation errors
        else:
            logger.warning(f"MODEL_DATA validation failed (non-fatal): {e}")


def validate_reference_antenna_stability(ms_path: str, refant_list: list = None) -> str:
    """Validate reference antenna stability and suggest best refant.

    Unstable reference antennas cause calibration failures and flagged solutions.
    Checks for data availability, phase stability, and amplitude consistency.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    refant_list :
        List of preferred reference antennas (e.g., [15, 20, 24])
        If None, analyzes all antennas

    Returns
    -------
    str
        Best reference antenna name (e.g., 'ea15')

    Raises
    ------
    RuntimeError
        If no suitable reference antenna found

    """
    import logging
    import os

    logger = logging.getLogger(__name__)

    try:
        with _helpers.table(ms_path, readonly=True) as tb:
            # Get antenna information
            ant1 = tb.getcol("ANTENNA1")
            ant2 = tb.getcol("ANTENNA2")
            flags = tb.getcol("FLAG")  # Shape: (nrow, nchan, npol)
            data = tb.getcol("DATA")  # Shape: (nrow, nchan, npol)

            # Get antenna table for names
            ms_ant_path = os.path.join(ms_path, "ANTENNA")
            with _helpers.table(ms_ant_path, readonly=True) as ant_tb:
                ant_names = ant_tb.getcol("NAME")

            unique_ants = np.unique(np.concatenate([ant1, ant2]))

            # Score each antenna candidate
            ant_scores = {}
            for ant_id in unique_ants:
                # Find baselines with this antenna
                ant_mask = (ant1 == ant_id) | (ant2 == ant_id)

                # Count unflagged data
                unflagged = ~flags[ant_mask]
                data_availability = np.sum(unflagged)

                # Check phase stability (sample across channels)
                if data_availability > 0:
                    ant_data = data[ant_mask]
                    ant_flags = flags[ant_mask]

                    # Use first polarization for phase check
                    pol_data = ant_data[:, :, 0] if ant_data.shape[2] > 0 else ant_data[:, :, 0]
                    pol_flags = ant_flags[:, :, 0] if ant_flags.shape[2] > 0 else ant_flags[:, :, 0]

                    # Calculate phase stability (std of phase)
                    valid_data = pol_data[~pol_flags]
                    if len(valid_data) > 100:
                        phases = np.angle(valid_data)
                        phase_std = float(np.std(phases))

                        # Score: higher data availability and lower phase std = better
                        score = data_availability / (1.0 + phase_std * 100)
                        ant_scores[ant_id] = score
                    else:
                        ant_scores[ant_id] = 0.0
                else:
                    ant_scores[ant_id] = 0.0

            # Select best antenna (prefer refant_list if provided)
            if refant_list:
                # Score refant_list candidates
                refant_scores = {
                    ant_id: ant_scores.get(ant_id, 0.0)
                    for ant_id in refant_list
                    if ant_id in unique_ants
                }
                if refant_scores:
                    best_ant_id = max(refant_scores, key=refant_scores.get)
                    best_ant_name = ant_names[best_ant_id]
                    logger.info(
                        f"Selected reference antenna from provided list: {best_ant_name} (score: {ant_scores[best_ant_id]:.2f})"
                    )
                    return best_ant_name

            # Otherwise, select best overall
            if not ant_scores:
                raise RuntimeError(f"No valid antennas found in MS: {ms_path}")

            best_ant_id = max(ant_scores, key=ant_scores.get)
            best_ant_name = ant_names[best_ant_id]
            logger.info(
                f"Selected best reference antenna: {best_ant_name} (score: {ant_scores[best_ant_id]:.2f})"
            )
            return best_ant_name

    except Exception as e:
        logger.warning(f"Reference antenna validation failed (non-fatal): {e}")
        # Fallback: return first antenna
        try:
            with _helpers.table(f"{ms_path}::ANTENNA", readonly=True) as ant_tb:
                ant_names = ant_tb.getcol("NAME")
                if len(ant_names) > 0:
                    return ant_names[0]
        except (RuntimeError, OSError, KeyError):
            # RuntimeError: CASA errors, OSError: file issues, KeyError: missing columns
            pass
        raise RuntimeError(f"Could not select reference antenna for {ms_path}") from e
