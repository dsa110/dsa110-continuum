"""
Geometric validation of delay (K) calibration solutions for DSA-110.

This module validates that solved delays are physically plausible given the
interferometer geometry. Delays should not exceed the light-travel time
between antennas.

For DSA-110:
- Max baseline: ~2707 m
- Max geometric delay: ~9 μs (9030 ns)
- Typical instrumental delays: < 100 ns

If solved delays significantly exceed geometric limits, the calibration
has likely failed and should be rejected.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dsa110_contimg.common.utils.casa_init import ensure_casa_path

ensure_casa_path()

# Import casacore for reading calibration tables
try:
    import casacore.tables as casatables

    table = casatables.table
    HAVE_CASACORE = True
except ImportError:
    table = None
    HAVE_CASACORE = False


logger = logging.getLogger(__name__)

# Speed of light in m/s
C_LIGHT = 299792458.0


def _parse_refant(refant: int | str | None) -> int | None:
    """Parse refant value, handling comma-separated lists.

    CASA's gaincal accepts a comma-separated list of reference antennas
    (priority order). This function extracts the first/primary refant.

    Parameters
    ----------
    refant : int | str | None
        Reference antenna ID. Can be:
        - int: single antenna ID (returned as-is)
        - str: single ID ("103") or comma-separated ("104,105,106")
        - None: returns None

    Returns
    -------
    int | None
        Primary reference antenna ID, or None if input is None.
    """
    if refant is None:
        return None
    if isinstance(refant, int):
        return refant
    # String: may be comma-separated list
    refant_str = str(refant).strip()
    if "," in refant_str:
        # Take first antenna from priority list
        first_ant = refant_str.split(",")[0].strip()
        return int(first_ant)
    return int(refant_str)


@dataclass
class DelayValidationResult:
    """Result of delay solution validation against geometric constraints."""

    is_valid: bool
    """Whether all delays are within geometric bounds."""

    n_antennas: int
    """Number of antennas with solutions."""

    n_flagged: int
    """Number of flagged antenna solutions."""

    n_within_bounds: int
    """Number of antennas with delays within geometric limits."""

    n_outside_bounds: int
    """Number of antennas with delays exceeding geometric limits."""

    max_geometric_delay_ns: float
    """Maximum allowed geometric delay (light-travel time to farthest antenna)."""

    delays_ns: dict[int, float] = field(default_factory=dict)
    """Solved delays per antenna in nanoseconds."""

    geometric_limits_ns: dict[int, float] = field(default_factory=dict)
    """Maximum geometric delay per antenna (distance to refant / c)."""

    violations: list[str] = field(default_factory=list)
    """List of validation violations."""

    warnings: list[str] = field(default_factory=list)
    """List of validation warnings."""

    refant: int | None = None
    """Reference antenna ID used for calibration."""

    def __str__(self) -> str:
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        lines = [
            f"Delay Validation: {status}",
            f"  Antennas: {self.n_antennas} total, {self.n_flagged} flagged",
            f"  Within bounds: {self.n_within_bounds}/{self.n_antennas - self.n_flagged}",
            f"  Max geometric delay: {self.max_geometric_delay_ns:.1f} ns",
        ]
        if self.violations:
            lines.append("  Violations:")
            for v in self.violations[:5]:  # Show first 5
                lines.append(f"    - {v}")
            if len(self.violations) > 5:
                lines.append(f"    ... and {len(self.violations) - 5} more")
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings[:3]:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def get_antenna_positions() -> np.ndarray:
    """Get DSA-110 antenna positions in ITRF coordinates.

    Returns
    -------
    np.ndarray
        Array of shape (n_antennas, 3) containing [x, y, z] in meters.
    """
    from dsa110_contimg.common.utils.antpos_local import get_itrf

    df = get_itrf()
    return np.array([df["x_m"], df["y_m"], df["z_m"]]).T


def compute_geometric_delay_limits(
    refant_idx: int,
    antenna_positions: np.ndarray | None = None,
    safety_factor: float = 1.5,
) -> dict[int, float]:
    """Compute maximum geometric delay for each antenna relative to reference.

    The geometric delay limit for antenna i is:
        max_delay_i = distance(ant_i, refant) / c × safety_factor

    Parameters
    ----------
    refant_idx : int
        Index of reference antenna in position array.
    antenna_positions : np.ndarray, optional
        Antenna positions in ITRF coordinates, shape (n_ant, 3).
        If None, loads DSA-110 positions.
    safety_factor : float
        Multiply geometric limit by this factor to allow for some
        instrumental delays. Default 1.5.

    Returns
    -------
    dict[int, float]
        Maximum allowed delay in nanoseconds for each antenna index.
    """
    if antenna_positions is None:
        antenna_positions = get_antenna_positions()

    n_ant = len(antenna_positions)
    refant_pos = antenna_positions[refant_idx]

    limits = {}
    for i in range(n_ant):
        if i == refant_idx:
            # Reference antenna has zero delay by definition
            limits[i] = 10.0  # Allow small tolerance for numerical noise
        else:
            distance = np.linalg.norm(antenna_positions[i] - refant_pos)
            geo_delay_s = distance / C_LIGHT
            geo_delay_ns = geo_delay_s * 1e9 * safety_factor
            limits[i] = geo_delay_ns

    return limits


def read_delay_solutions(caltable_path: str) -> tuple[dict[int, float], int | None]:
    """Read delay solutions from a CASA K-calibration table.

    Parameters
    ----------
    caltable_path : str
        Path to K-calibration table.

    Returns
    -------
    dict[int, float]
        Delays per antenna ID in nanoseconds.
    int | None
        Reference antenna ID, if determinable.
    """
    if not HAVE_CASACORE:
        raise ImportError(
            "casacore is required to read calibration tables. "
            "Install with: pip install python-casacore"
        )

    delays = {}
    refant = None

    with table(caltable_path, readonly=True, ack=False) as tb:
        antenna_ids = tb.getcol("ANTENNA1")
        flags = tb.getcol("FLAG")

        # FPARAM contains delays (float values)
        if "FPARAM" not in tb.colnames():
            raise ValueError(f"Not a K-calibration table: {caltable_path}")

        fparam = tb.getcol("FPARAM")

        # Get unique antennas
        unique_ants = np.unique(antenna_ids)

        for ant in unique_ants:
            ant_mask = antenna_ids == ant
            ant_fparam = fparam[:, :, ant_mask]
            ant_flags = flags[:, :, ant_mask]

            # Average unflagged values
            unflagged = ant_fparam[~ant_flags]
            if len(unflagged) > 0:
                # FPARAM is typically in seconds
                delay_s = float(np.median(unflagged))
                delay_ns = delay_s * 1e9
                delays[int(ant)] = delay_ns

                # Reference antenna typically has near-zero delay
                if abs(delay_ns) < 1.0:
                    refant = int(ant)

    return delays, refant


def validate_delay_solutions(
    caltable_path: str,
    refant: int | str | None = None,
    antenna_positions: np.ndarray | None = None,
    safety_factor: float = 1.5,
    strict: bool = False,
) -> DelayValidationResult:
    """Validate delay solutions against geometric constraints.

    Checks that solved delays do not exceed the light-travel time
    between each antenna and the reference antenna.

    Parameters
    ----------
    caltable_path : str
        Path to K-calibration table.
    refant : int | str | None
        Reference antenna ID or name. If None, attempts to infer from
        the caltable (antenna with smallest delay).
    antenna_positions : np.ndarray, optional
        Antenna positions. If None, loads DSA-110 positions.
    safety_factor : float
        Allow delays up to this factor × geometric limit.
        Default 1.5 allows 50% margin for instrumental delays.
    strict : bool
        If True, any violation makes result invalid.
        If False, allows some violations with warnings.

    Returns
    -------
    DelayValidationResult
        Validation results with per-antenna details.
    """
    if not HAVE_CASACORE:
        raise ImportError("casacore required for delay validation")

    # Load antenna positions
    if antenna_positions is None:
        antenna_positions = get_antenna_positions()

    # Read delay solutions
    delays_ns, inferred_refant = read_delay_solutions(caltable_path)

    # Determine reference antenna
    if refant is not None:
        refant_idx = _parse_refant(refant)
    elif inferred_refant is not None:
        refant_idx = inferred_refant
    else:
        # Fall back to antenna with smallest delay
        if delays_ns:
            refant_idx = min(delays_ns.keys(), key=lambda k: abs(delays_ns[k]))
        else:
            refant_idx = 0

    # Compute geometric limits
    geo_limits = compute_geometric_delay_limits(
        refant_idx, antenna_positions, safety_factor
    )

    # Validate each antenna
    violations = []
    warnings = []
    n_within = 0
    n_outside = 0
    n_flagged = len(antenna_positions) - len(delays_ns)

    for ant_id, delay in delays_ns.items():
        if ant_id not in geo_limits:
            warnings.append(f"Antenna {ant_id} not in position array")
            continue

        limit = geo_limits[ant_id]
        if abs(delay) <= limit:
            n_within += 1
        else:
            n_outside += 1
            excess = abs(delay) - limit
            violations.append(
                f"Antenna {ant_id}: |{delay:.1f}| ns > {limit:.1f} ns limit "
                f"(excess: {excess:.1f} ns)"
            )

    # Additional checks
    if delays_ns:
        delays_arr = np.array(list(delays_ns.values()))
        median_delay = np.median(np.abs(delays_arr))
        delay_scatter = np.std(delays_arr)

        # Check for unreasonably large scatter
        # Good delay solutions should have scatter < 100 ns typically
        if delay_scatter > 500:
            warnings.append(f"High delay scatter: {delay_scatter:.1f} ns (expect <100 ns)")

        # Check for systematically large delays
        max_geo = max(geo_limits.values())
        if median_delay > max_geo:
            violations.append(
                f"Median delay {median_delay:.1f} ns exceeds max geometric "
                f"limit {max_geo:.1f} ns - likely calibration failure"
            )
    else:
        median_delay = 0.0

    # Determine validity
    if strict:
        is_valid = len(violations) == 0
    else:
        # Allow up to 10% antennas outside bounds
        n_checked = n_within + n_outside
        violation_frac = n_outside / n_checked if n_checked > 0 else 0
        is_valid = violation_frac < 0.10 and not any(
            "calibration failure" in v for v in violations
        )

    return DelayValidationResult(
        is_valid=is_valid,
        n_antennas=len(antenna_positions),
        n_flagged=n_flagged,
        n_within_bounds=n_within,
        n_outside_bounds=n_outside,
        max_geometric_delay_ns=max(geo_limits.values()) if geo_limits else 0.0,
        delays_ns=delays_ns,
        geometric_limits_ns=geo_limits,
        violations=violations,
        warnings=warnings,
        refant=refant_idx,
    )


def check_delay_solutions(
    caltable_path: str,
    refant: int | str | None = None,
    raise_on_failure: bool = True,
    **kwargs: Any,
) -> DelayValidationResult:
    """Validate delay solutions and optionally raise on failure.

    This is the main entry point for pipeline integration.

    Parameters
    ----------
    caltable_path : str
        Path to K-calibration table.
    refant : int | str | None
        Reference antenna.
    raise_on_failure : bool
        If True, raises CalibrationError on validation failure.
    **kwargs
        Additional arguments passed to validate_delay_solutions.

    Returns
    -------
    DelayValidationResult
        Validation results.

    Raises
    ------
    CalibrationError
        If validation fails and raise_on_failure is True.
    """
    result = validate_delay_solutions(caltable_path, refant=refant, **kwargs)

    logger.info(str(result))

    if not result.is_valid and raise_on_failure:
        from dsa110_contimg.common.utils.exceptions import CalibrationError

        msg = (
            f"Delay calibration validation failed for {caltable_path}:\n"
            f"  {len(result.violations)} violations, "
            f"{result.n_outside_bounds}/{result.n_antennas - result.n_flagged} "
            f"antennas outside geometric bounds.\n"
            f"  First violation: {result.violations[0] if result.violations else 'N/A'}"
        )
        raise CalibrationError(msg)

    return result
