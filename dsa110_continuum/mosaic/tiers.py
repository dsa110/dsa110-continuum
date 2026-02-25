"""
Mosaic tier definitions and selection logic.

Exactly three tiers with clear definitions:
- Quicklook: Real-time monitoring, fast, permissive
- Science: Publication-quality, nightly, quality-filtered
- Deep: Targeted integration, on-demand, best images only
"""

from dataclasses import dataclass
from enum import Enum


class MosaicTier(Enum):
    """The three mosaic quality tiers."""

    QUICKLOOK = "quicklook"
    SCIENCE = "science"
    DEEP = "deep"


@dataclass(frozen=True)
class TierConfig:
    """Immutable tier configuration.

    Tiers control quality settings (RMS filtering, alignment accuracy),
    not sky coverage. Coverage is determined by the time range requested.

    """

    tier: MosaicTier
    rms_threshold_jy: float
    alignment_order: int
    require_astrometry: bool
    timeout_minutes: int


# The only three tier configurations that exist
TIER_CONFIGS: dict[MosaicTier, TierConfig] = {
    MosaicTier.QUICKLOOK: TierConfig(
        tier=MosaicTier.QUICKLOOK,
        rms_threshold_jy=0.01,  # Permissive
        alignment_order=1,  # Fast nearest-neighbor
        require_astrometry=False,
        timeout_minutes=5,
    ),
    MosaicTier.SCIENCE: TierConfig(
        tier=MosaicTier.SCIENCE,
        rms_threshold_jy=0.001,  # Quality filter
        alignment_order=3,  # High-order SIP
        require_astrometry=True,
        timeout_minutes=30,
    ),
    MosaicTier.DEEP: TierConfig(
        tier=MosaicTier.DEEP,
        rms_threshold_jy=0.0005,  # Best images only
        alignment_order=5,  # Full astrometric solve
        require_astrometry=True,
        timeout_minutes=120,
    ),
}


def select_tier_for_request(
    time_range_hours: float,
    target_quality: str | None = None,
) -> MosaicTier:
    """Automatic tier selection based on request parameters.

    Simple, clear logic:
    - Recent data (< 1 hour) → Quicklook
    - Daily range + quality → Science
    - Multi-day + "deep" requested → Deep

    Parameters
    ----------
    time_range_hours : float
        Time span of data to mosaic in hours.
    target_quality : str or None, optional
        Optional quality hint ("quicklook", "science", "deep").
        Default is None.

    Returns
    -------
    MosaicTier
        Selected MosaicTier.

    Examples
    --------
    >>> select_tier_for_request(0.5)
    MosaicTier.QUICKLOOK

    >>> select_tier_for_request(24.0)
    MosaicTier.SCIENCE

    >>> select_tier_for_request(72.0, "deep")
    MosaicTier.DEEP
    """
    # Explicit tier request takes precedence
    if target_quality:
        quality_lower = target_quality.lower()
        if quality_lower == "quicklook":
            return MosaicTier.QUICKLOOK
        elif quality_lower == "deep":
            return MosaicTier.DEEP
        elif quality_lower == "science":
            return MosaicTier.SCIENCE

    # Automatic selection based on time range
    if time_range_hours < 1:
        return MosaicTier.QUICKLOOK
    elif time_range_hours > 48:
        return MosaicTier.DEEP
    else:
        return MosaicTier.SCIENCE


def get_tier_config(tier: MosaicTier | str) -> TierConfig:
    """Get configuration for a tier.

    Parameters
    ----------
    tier :
        MosaicTier enum or string name
    tier: MosaicTier | str :


    Returns
    -------
        TierConfig for the tier

    Raises
    ------
    ValueError
        If tier is not recognized

    """
    if isinstance(tier, str):
        try:
            tier = MosaicTier(tier.lower())
        except ValueError:
            raise ValueError(f"Unknown tier: '{tier}'. Valid tiers: quicklook, science, deep")

    return TIER_CONFIGS[tier]
