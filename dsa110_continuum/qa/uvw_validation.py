"""
UVW geometry validation for DSA-110.

Validates that UVW values are physically plausible given the array geometry.

The DSA-110 array has a maximum baseline of ~2707 m. UVW values should never
exceed this physical limit.

Usage:
    from dsa110_contimg.core.qa.uvw_validation import validate_uvw_geometry

    result = validate_uvw_geometry("/path/to/ms")
    if not result.is_valid:
        print(f"UVW validation failed: {result.violations}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# DSA-110 array parameters
DSA110_MAX_BASELINE_M = 2707.0  # Maximum baseline length in meters
DSA110_MIN_BASELINE_M = 10.0  # Minimum baseline length (exclude autocorrs)


@dataclass
class UVWValidationResult:
    """Result of UVW geometry validation."""

    is_valid: bool
    """Whether all UVW values are within physical limits."""

    max_baseline_m: float
    """Maximum allowed baseline length."""

    max_uvw_distance_m: float
    """Maximum observed UVW distance in the data."""

    min_uvw_distance_m: float
    """Minimum observed UVW distance (excluding zeros)."""

    median_uvw_distance_m: float
    """Median UVW distance."""

    n_baselines: int
    """Total number of baselines checked."""

    n_violations: int
    """Number of baselines exceeding physical limits."""

    violation_fraction: float
    """Fraction of baselines with violations."""

    n_zeros: int
    """Number of zero-length baselines (autocorrelations or flagged)."""

    violations: list[str] = field(default_factory=list)
    """List of validation violation messages."""

    warnings: list[str] = field(default_factory=list)
    """List of validation warning messages."""

    statistics: dict[str, float] = field(default_factory=dict)
    """Additional UVW statistics."""

    def __str__(self) -> str:
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        lines = [
            f"UVW Validation: {status}",
            f"  Max baseline limit: {self.max_baseline_m:.1f} m",
            f"  Max UVW observed: {self.max_uvw_distance_m:.1f} m",
            f"  Median UVW: {self.median_uvw_distance_m:.1f} m",
            f"  Baselines: {self.n_baselines:,} total, {self.n_zeros:,} zeros",
        ]
        if self.n_violations > 0:
            lines.append(
                f"  Violations: {self.n_violations:,} ({self.violation_fraction * 100:.2f}%)"
            )
        if self.violations:
            lines.append("  Issues:")
            for v in self.violations[:5]:
                lines.append(f"    - {v}")
            if len(self.violations) > 5:
                lines.append(f"    ... and {len(self.violations) - 5} more")
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings[:3]:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def validate_uvw_geometry(
    ms_path: str,
    max_baseline_m: float = DSA110_MAX_BASELINE_M,
    tolerance_factor: float = 1.1,
    sample_size: int | None = None,
    check_distribution: bool = True,
) -> UVWValidationResult:
    """Verify UVW values are physically plausible.

    This function validates that UVW coordinates in a Measurement Set
    are consistent with the physical array geometry. It catches issues
    like the chgcentre bug where UVW values exceed physical baseline limits.

    Checks performed:
    1. Max UVW distance ≤ max_baseline_m × tolerance_factor
    2. UVW values are not all zeros (would indicate missing data)
    3. Distribution is unimodal (bimodal suggests convention error)
    4. No significant fraction of baselines exceed limits

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set.
    max_baseline_m : float, optional
        Maximum physical baseline length in meters.
        Default is DSA-110's 2707 m.
    tolerance_factor : float, optional
        Allow UVW distances up to this factor × max_baseline_m.
        Default 1.1 allows 10% margin for numerical precision.
    sample_size : int or None, optional
        If provided, randomly sample this many rows for validation.
        Useful for very large datasets. Default None uses all rows.
    check_distribution : bool, optional
        If True, check for bimodal distribution indicating convention errors.
        Default True.

    Returns
    -------
    UVWValidationResult
        Validation results with detailed statistics.

    Examples
    --------
    >>> result = validate_uvw_geometry("/path/to/ms")
    >>> if not result.is_valid:
    ...     print("UVW validation failed!")
    ...     for violation in result.violations:
    ...         print(f"  - {violation}")
    """
    from dsa110_contimg.common.utils.casa_init import ensure_casa_path

    ensure_casa_path()

    import casacore.tables as ct

    violations = []
    warnings = []
    statistics = {}

    # Read UVW data
    with ct.table(ms_path, readonly=True, ack=False) as tb:
        n_rows = tb.nrows()

        if sample_size is not None and sample_size < n_rows:
            # Random sampling for large datasets
            rng = np.random.default_rng(42)
            sample_indices = rng.choice(n_rows, size=sample_size, replace=False)
            sample_indices = np.sort(sample_indices)
            uvw_list = []
            for idx in sample_indices:
                uvw_list.append(tb.getcell("UVW", int(idx)))
            uvw = np.array(uvw_list)
        else:
            uvw = tb.getcol("UVW")  # Shape: (nrow, 3)

    n_baselines = len(uvw)

    # Compute baseline lengths from UVW
    uvw_distances = np.sqrt(np.sum(uvw**2, axis=1))

    # Separate zeros (autocorrelations or flagged data)
    zero_mask = uvw_distances < 1e-6  # Less than 1 micron
    n_zeros = int(np.sum(zero_mask))
    nonzero_distances = uvw_distances[~zero_mask]

    if len(nonzero_distances) == 0:
        violations.append("All UVW values are zero - no valid baseline data")
        return UVWValidationResult(
            is_valid=False,
            max_baseline_m=max_baseline_m,
            max_uvw_distance_m=0.0,
            min_uvw_distance_m=0.0,
            median_uvw_distance_m=0.0,
            n_baselines=n_baselines,
            n_violations=n_baselines,
            violation_fraction=1.0,
            n_zeros=n_zeros,
            violations=violations,
            warnings=warnings,
            statistics=statistics,
        )

    # Compute statistics
    max_uvw = float(np.max(nonzero_distances))
    min_uvw = float(np.min(nonzero_distances))
    median_uvw = float(np.median(nonzero_distances))
    mean_uvw = float(np.mean(nonzero_distances))
    std_uvw = float(np.std(nonzero_distances))

    statistics["max_uvw_m"] = max_uvw
    statistics["min_uvw_m"] = min_uvw
    statistics["median_uvw_m"] = median_uvw
    statistics["mean_uvw_m"] = mean_uvw
    statistics["std_uvw_m"] = std_uvw

    # Also compute U, V, W component statistics
    u_vals = uvw[~zero_mask, 0]
    v_vals = uvw[~zero_mask, 1]
    w_vals = uvw[~zero_mask, 2]

    statistics["max_u_m"] = float(np.max(np.abs(u_vals)))
    statistics["max_v_m"] = float(np.max(np.abs(v_vals)))
    statistics["max_w_m"] = float(np.max(np.abs(w_vals)))

    # Check 1: Maximum UVW distance within physical limits
    limit = max_baseline_m * tolerance_factor
    violation_mask = nonzero_distances > limit
    n_violations = int(np.sum(violation_mask))
    violation_fraction = n_violations / len(nonzero_distances)

    if max_uvw > limit:
        excess_factor = max_uvw / max_baseline_m
        violations.append(
            f"Max UVW distance {max_uvw:.1f} m exceeds limit {limit:.1f} m "
            f"({excess_factor:.2f}× max baseline of {max_baseline_m:.1f} m)"
        )

        # Check if this looks like the chgcentre 2x bug
        if 1.8 < excess_factor < 2.2:
            violations.append(
                "UVW values are approximately 2× physical limits - "
                "this is characteristic of the chgcentre UVW convention mismatch. "
                "Use CASA phaseshift instead: phaseshift_ms(..., use_chgcentre=False)"
            )

    if violation_fraction > 0.01:
        violations.append(
            f"{violation_fraction * 100:.1f}% of baselines ({n_violations:,}) exceed "
            f"physical limits - likely UVW recalculation error"
        )

    # Check 2: Minimum baseline sanity (should have some short baselines)
    if min_uvw > max_baseline_m * 0.5:
        warnings.append(
            f"Minimum baseline {min_uvw:.1f} m is unusually large - "
            "check if short baselines are missing or incorrectly flagged"
        )

    # Check 3: Distribution analysis for convention errors
    if check_distribution and len(nonzero_distances) > 100:
        # Check for bimodal distribution (sign of convention error)
        percentiles = np.percentile(nonzero_distances, [10, 50, 90])
        p10, p50, p90 = percentiles

        statistics["p10_uvw_m"] = float(p10)
        statistics["p50_uvw_m"] = float(p50)
        statistics["p90_uvw_m"] = float(p90)

        # Coefficient of variation
        cv = std_uvw / mean_uvw if mean_uvw > 0 else 0
        statistics["coefficient_of_variation"] = cv

        # Check for suspicious distribution shape
        # A healthy UVW distribution should be roughly uniform or normal
        # Bimodal distribution with peaks at x and 2x suggests convention error
        if cv > 0.8:
            warnings.append(
                f"High UVW distance variability (CV={cv:.2f}) - "
                "may indicate mixed UVW conventions in data"
            )

    # Check 4: W component should be consistent with source declination
    # For a source at declination δ, W should scale with sin(δ)
    # Large W values relative to UV indicate potential issues
    max_uv = np.sqrt(statistics["max_u_m"] ** 2 + statistics["max_v_m"] ** 2)
    max_w = statistics["max_w_m"]

    if max_w > max_uv * 2:
        warnings.append(
            f"W component ({max_w:.1f} m) is much larger than UV ({max_uv:.1f} m) - "
            "verify source declination and W-term handling"
        )

    statistics["max_uv_m"] = max_uv
    statistics["uv_w_ratio"] = max_uv / max_w if max_w > 0 else float("inf")

    # Determine overall validity
    is_valid = len(violations) == 0

    result = UVWValidationResult(
        is_valid=is_valid,
        max_baseline_m=max_baseline_m,
        max_uvw_distance_m=max_uvw,
        min_uvw_distance_m=min_uvw,
        median_uvw_distance_m=median_uvw,
        n_baselines=n_baselines,
        n_violations=n_violations,
        violation_fraction=violation_fraction,
        n_zeros=n_zeros,
        violations=violations,
        warnings=warnings,
        statistics=statistics,
    )

    logger.info(str(result))

    return result


def check_uvw_after_phaseshift(
    ms_path: str,
    raise_on_failure: bool = True,
    **kwargs: Any,
) -> UVWValidationResult:
    """Validate UVW geometry after phase shifting.

    This is the main entry point for pipeline integration.
    Should be called after phaseshift_ms() or chgcentre operations.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set.
    raise_on_failure : bool, optional
        If True, raises ValueError on validation failure. Default True.
    **kwargs
        Additional arguments passed to validate_uvw_geometry.

    Returns
    -------
    UVWValidationResult
        Validation results.

    Raises
    ------
    ValueError
        If validation fails and raise_on_failure is True.
    """
    result = validate_uvw_geometry(ms_path, **kwargs)

    if not result.is_valid and raise_on_failure:
        msg = (
            f"UVW validation failed for {ms_path}:\n"
            f"  Max UVW distance: {result.max_uvw_distance_m:.1f} m "
            f"(limit: {result.max_baseline_m * 1.1:.1f} m)\n"
            f"  Violations: {result.n_violations:,} "
            f"({result.violation_fraction * 100:.1f}%)\n"
        )
        if result.violations:
            msg += f"  First issue: {result.violations[0]}\n"
        msg += (
            "\nThis typically indicates incorrect UVW recalculation.\n"
            "If you used chgcentre, try CASA phaseshift instead:\n"
            "  phaseshift_ms(..., use_chgcentre=False)"
        )
        raise ValueError(msg)

    return result


def compare_uvw_before_after(
    ms_before: str,
    ms_after: str,
    tolerance_m: float = 1.0,
) -> dict[str, Any]:
    """Compare UVW values before and after a transformation.

    Useful for debugging phase shift operations.

    Parameters
    ----------
    ms_before : str
        Path to original Measurement Set.
    ms_after : str
        Path to transformed Measurement Set.
    tolerance_m : float, optional
        Tolerance for UVW differences in meters. Default 1.0 m.

    Returns
    -------
    dict
        Comparison statistics including max difference and correlation.
    """
    from dsa110_contimg.common.utils.casa_init import ensure_casa_path

    ensure_casa_path()

    import casacore.tables as ct

    with ct.table(ms_before, readonly=True, ack=False) as tb:
        uvw_before = tb.getcol("UVW")

    with ct.table(ms_after, readonly=True, ack=False) as tb:
        uvw_after = tb.getcol("UVW")

    if uvw_before.shape != uvw_after.shape:
        return {
            "compatible": False,
            "error": f"Shape mismatch: {uvw_before.shape} vs {uvw_after.shape}",
        }

    # Compute differences
    diff = uvw_after - uvw_before
    diff_magnitude = np.sqrt(np.sum(diff**2, axis=1))

    # Compute distance scaling
    dist_before = np.sqrt(np.sum(uvw_before**2, axis=1))
    dist_after = np.sqrt(np.sum(uvw_after**2, axis=1))

    # Avoid division by zero
    nonzero_mask = dist_before > 1e-6
    if np.sum(nonzero_mask) > 0:
        scale_ratio = dist_after[nonzero_mask] / dist_before[nonzero_mask]
        mean_scale = float(np.mean(scale_ratio))
        std_scale = float(np.std(scale_ratio))
    else:
        mean_scale = 1.0
        std_scale = 0.0

    result = {
        "compatible": True,
        "n_rows": len(uvw_before),
        "max_diff_m": float(np.max(diff_magnitude)),
        "mean_diff_m": float(np.mean(diff_magnitude)),
        "median_diff_m": float(np.median(diff_magnitude)),
        "n_changed": int(np.sum(diff_magnitude > tolerance_m)),
        "mean_scale_ratio": mean_scale,
        "std_scale_ratio": std_scale,
        "max_u_diff": float(np.max(np.abs(diff[:, 0]))),
        "max_v_diff": float(np.max(np.abs(diff[:, 1]))),
        "max_w_diff": float(np.max(np.abs(diff[:, 2]))),
    }

    # Check for the 2x scaling bug
    if 1.8 < mean_scale < 2.2 and std_scale < 0.1:
        result["warning"] = (
            f"UVW values scaled by ~{mean_scale:.2f}× - "
            "this is characteristic of the chgcentre convention mismatch"
        )
        result["likely_chgcentre_bug"] = True
    else:
        result["likely_chgcentre_bug"] = False

    return result
