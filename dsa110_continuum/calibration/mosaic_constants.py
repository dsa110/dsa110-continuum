"""Constants for mosaic calibration scheduling.

This module defines constants used by the mosaic calibration scheduler for
determining when to trigger gain calibration and bandpass calibration based on
mosaic completion and calibrator transits.
"""

# Source query parameters for sky model construction
SOURCE_QUERY_RADIUS_DEG = 0.3  # Conservative, ensures >50% beam response
MIN_SKYMODEL_SOURCES = 3  # Minimum sources for sky model (fall back to brightest if fewer)
SKYMODEL_MIN_FLUX_MJY = 5.0  # SNR threshold for sources

# Mosaic parameters
MOSAIC_TILE_COUNT = 12  # Number of consecutive tiles at same declination

# Declination change threshold
DEC_CHANGE_THRESHOLD_DEG = 5.0  # Trigger BP calibrator reselection

# Field center prediction parameters
INTEGRATION_TIME_SEC = 12.884902  # Integration time per field in seconds
N_FIELDS = 24  # Number of fields per observation
EARTH_ROTATION_DEG_PER_SEC = 360.0 / 86164.0905  # Sidereal day in seconds

# Calibration validity windows
BP_VALIDITY_HOURS = 24  # Bandpass calibration validity
GAIN_VALIDITY_HOURS = 1  # Gain calibration validity

# Source deduplication parameters
SOURCE_MATCH_RADIUS_ARCSEC = 5.0  # Match radius for deduplication
