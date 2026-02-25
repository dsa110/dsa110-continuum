"""DSA-110 Tile Specification Constants and Validators.

This module provides centralized tile specification constants that define
the standard 5-minute observation tiles produced by the DSA-110 continuum
imaging pipeline.

A "tile" is one 16-subband group representing approximately 5 minutes (309 seconds)
of drift-scan observing data, which is the fundamental unit of the imaging pipeline.

Examples
--------
>>> from dsa110_contimg.core.simulation.tile_specs import STANDARD_TILE, TileSpecification
>>>
>>> # Use standard specification
>>> print(f"Tile duration: {STANDARD_TILE.total_duration_sec} seconds")
>>> print(f"Field of view: {STANDARD_TILE.fov_deg:.2f} degrees")
>>>
>>> # Validate a FITS header
>>> from astropy.io import fits
>>> hdu = fits.open("tile.fits")[0]
>>> errors = STANDARD_TILE.validate_fits_header(hdu.header)
>>> if errors:
...     print("Validation errors:", errors)
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TileSpecification:
    """Standard DSA-110 tile parameters.

    This dataclass encapsulates all the key parameters that define a standard
    DSA-110 observation tile, including temporal, spectral, imaging, and
    antenna configuration specifications.

    Attributes
    ----------
    integration_time_sec : float
        Integration time per visibility sample (seconds)
    num_integrations : int
        Number of time integrations in a tile
    total_duration_sec : float
        Total observation duration (seconds)
    num_subbands : int
        Number of spectral subbands (16 for DSA-110)
    channels_per_subband : int
        Number of frequency channels per subband
    channel_width_hz : float
        Width of each frequency channel (Hz)
    subband_width_hz : float
        Total bandwidth of one subband (Hz)
    total_bandwidth_hz : float
        Total bandwidth across all subbands (Hz)
    freq_min_hz : float
        Minimum frequency (Hz)
    freq_max_hz : float
        Maximum frequency (Hz)
    reference_freq_hz : float
        Reference frequency for the observation (Hz)
    image_size_px : int
        Image size in pixels (square images)
    pixel_scale_arcsec : float
        Pixel scale (arcseconds per pixel)
    fov_deg : float
        Field of view (degrees, square FOV)
    weighting : str
        Visibility weighting scheme
    robust : float
        Briggs robustness parameter
    typical_beam_fwhm_arcsec : float
        Typical synthesized beam FWHM (arcseconds)
    typical_noise_mjy : float
        Typical RMS noise level (mJy/beam)
    num_antennas : int
        Number of antennas in the array
    """

    # Temporal parameters (drift-scan observation)
    # Integration time from correlator (high precision)
    integration_time_sec: float = 12.884902000427246
    num_integrations: int = 24
    total_duration_sec: float = 309.12  # ~5 minutes

    # Spectral parameters (L-band, 1.3114-1.4986 GHz)
    # Values from dsa110_measured_parameters.yaml (authoritative source)
    num_subbands: int = 16
    channels_per_subband: int = 48
    channel_width_hz: float = 244140.625  # 15.625 MHz / 48 channels per raw subband
    subband_width_hz: float = 11718750.0  # 48 × 244140.625 Hz
    total_bandwidth_hz: float = 187500000.0  # 16 × 11.71875 MHz
    freq_min_hz: float = 1.3114e9  # Lower edge of sb15 (verified from correlator)
    freq_max_hz: float = 1.4986e9  # Upper edge of sb00 (verified from correlator)
    reference_freq_hz: float = 1.405e9  # Center of L-band

    # Imaging parameters (standard quality)
    image_size_px: int = 2048
    pixel_scale_arcsec: float = 6.0
    fov_deg: float = field(init=False)  # Computed from image_size_px and pixel_scale_arcsec
    weighting: str = "briggs"
    robust: float = 0.0
    typical_beam_fwhm_arcsec: float = 10.0
    typical_noise_mjy: float = 1.0

    # Antenna configuration (full DSA-110 array)
    num_antennas: int = 117

    def __post_init__(self):
        """Compute derived parameters."""
        # Field of view in degrees
        self.fov_deg = (self.image_size_px * self.pixel_scale_arcsec) / 3600.0

    def validate_fits_header(self, header) -> list[str]:
        """Validate FITS header matches tile specifications.

        Parameters
        ----------
        header : astropy.io.fits.Header
            FITS header to validate

        Returns
        -------
        errors : List[str]
            List of validation error messages (empty if valid)

        Examples
        --------
        >>> from astropy.io import fits
        >>> hdu = fits.open("tile.fits")[0]
        >>> spec = TileSpecification()
        >>> errors = spec.validate_fits_header(hdu.header)
        >>> if errors:
        ...     for error in errors:
        ...         print(f"ERROR: {error}")
        """
        errors = []

        # Check image dimensions
        if "NAXIS1" in header and header["NAXIS1"] != self.image_size_px:
            errors.append(
                f"Image width mismatch: expected {self.image_size_px}, got {header['NAXIS1']}"
            )
        if "NAXIS2" in header and header["NAXIS2"] != self.image_size_px:
            errors.append(
                f"Image height mismatch: expected {self.image_size_px}, got {header['NAXIS2']}"
            )

        # Check pixel scale (from CDELT keywords or CD matrix)
        if "CDELT1" in header:
            cdelt1_arcsec = abs(header["CDELT1"]) * 3600.0  # Convert deg to arcsec
            tolerance = 0.1  # 0.1 arcsec tolerance
            if abs(cdelt1_arcsec - self.pixel_scale_arcsec) > tolerance:
                errors.append(
                    f"Pixel scale mismatch: expected {self.pixel_scale_arcsec} arcsec, "
                    f"got {cdelt1_arcsec:.2f} arcsec"
                )

        # Check required keywords exist
        required_keywords = [
            "NAXIS",
            "NAXIS1",
            "NAXIS2",
            "CTYPE1",
            "CTYPE2",
            "CRPIX1",
            "CRPIX2",
            "CRVAL1",
            "CRVAL2",
        ]
        for keyword in required_keywords:
            if keyword not in header:
                errors.append(f"Missing required FITS keyword: {keyword}")

        # Check coordinate system
        if "CTYPE1" in header and not header["CTYPE1"].startswith("RA"):
            errors.append(f"Unexpected CTYPE1: {header['CTYPE1']} (expected RA---)")
        if "CTYPE2" in header and not header["CTYPE2"].startswith("DEC"):
            errors.append(f"Unexpected CTYPE2: {header['CTYPE2']} (expected DEC--)")

        return errors

    def get_frequency_array(self) -> np.ndarray:
        """Generate frequency array for all subbands.

        Returns
        -------
        frequencies : np.ndarray
            Array of frequencies (Hz) with shape (num_subbands * channels_per_subband,)

        Examples
        --------
        >>> spec = TileSpecification()
        >>> freqs = spec.get_frequency_array()
        >>> print(f"Total channels: {len(freqs)}")
        >>> print(f"Frequency range: {freqs.min()/1e9:.3f} - {freqs.max()/1e9:.3f} GHz")
        """
        frequencies = []
        channel_width = self.subband_width_hz / self.channels_per_subband
        for sb_idx in range(self.num_subbands):
            # Calculate subband start frequency
            sb_start_hz = self.freq_min_hz + sb_idx * self.subband_width_hz
            # Generate channel center frequencies for this subband
            sb_freqs = sb_start_hz + channel_width * (np.arange(self.channels_per_subband) + 0.5)
            frequencies.extend(sb_freqs)

        return np.array(frequencies)

    def get_time_array(self, start_time_mjd: float) -> np.ndarray:
        """Generate time array for integrations.

        Parameters
        ----------
        start_time_mjd : float
            Start time in Modified Julian Date (MJD)

        Returns
        -------
        times : np.ndarray
            Array of times (MJD) with shape (num_integrations,)

        Examples
        --------
        >>> from astropy.time import Time
        >>> spec = TileSpecification()
        >>> start = Time("2025-01-15T12:00:00").mjd
        >>> times = spec.get_time_array(start)
        >>> print(f"Duration: {(times[-1] - times[0]) * 24 * 60:.1f} minutes")
        """
        dt_days = self.integration_time_sec / 86400.0  # Convert seconds to days
        times = start_time_mjd + dt_days * np.arange(self.num_integrations)
        return times

    def estimate_beam_size(self, frequency_hz: float | None = None) -> float:
        """Estimate synthesized beam FWHM for a given frequency.

        Uses simplified formula: θ ≈ λ / D where D is effective baseline.
        For DSA-110, typical baseline ~3 km gives ~10 arcsec at 1.4 GHz.

        Parameters
        ----------
        frequency_hz : float, optional
            Frequency in Hz (default: reference_freq_hz)

        Returns
        -------
        beam_fwhm_arcsec : float
            Estimated beam FWHM in arcseconds

        Examples
        --------
        >>> spec = TileSpecification()
        >>> beam = spec.estimate_beam_size()
        >>> print(f"Beam size: {beam:.1f} arcsec")
        """
        if frequency_hz is None:
            frequency_hz = self.reference_freq_hz

        # Wavelength in meters
        c = 299792458.0  # Speed of light (m/s)
        wavelength_m = c / frequency_hz

        # Effective baseline (approximate for DSA-110)
        effective_baseline_m = 3210.0  # ~3 km

        # Beam size: θ = λ / D (in radians), convert to arcseconds
        beam_rad = wavelength_m / effective_baseline_m
        beam_arcsec = beam_rad * 206265.0  # Convert radians to arcseconds

        return beam_arcsec

    def estimate_thermal_noise(
        self,
        system_temp_k: float = 50.0,
        efficiency: float = 0.7,
        frequency_hz: float | None = None,
    ) -> float:
        """Estimate thermal noise RMS for a tile observation.

        Uses radiometer equation:
        σ = T_sys / (η * sqrt(N_ant * (N_ant - 1) * Δν * t))

        Parameters
        ----------
        system_temp_k : float
            System temperature in Kelvin (default: 50 K for DSA-110)
        efficiency : float
            System efficiency (default: 0.7)
        frequency_hz : float, optional
            Observation frequency (default: reference_freq_hz)

        Returns
        -------
        noise_rms_jy : float
            Estimated RMS noise in Jy/beam

        Examples
        --------
        >>> spec = TileSpecification()
        >>> noise = spec.estimate_thermal_noise()
        >>> print(f"Expected noise: {noise*1000:.2f} mJy/beam")
        """
        if frequency_hz is None:
            frequency_hz = self.reference_freq_hz

        # Number of baselines
        n_baselines = self.num_antennas * (self.num_antennas - 1) / 2

        # Total observing bandwidth and time
        delta_nu = self.total_bandwidth_hz
        t_obs = self.total_duration_sec

        # Radiometer equation (simplified)
        # Convert temperature to flux density using 2k T / A_eff
        # For DSA-110: A_eff ≈ 17 m^2 per antenna
        k_boltzmann = 1.380649e-23  # J/K
        a_eff = 17.0  # m^2

        # System equivalent flux density (SEFD)
        sefd_jy = (2 * k_boltzmann * system_temp_k) / (a_eff * 1e-26)  # 1e-26 for Jy conversion

        # RMS noise
        noise_jy = sefd_jy / (efficiency * np.sqrt(2 * n_baselines * delta_nu * t_obs))

        return noise_jy

    def to_dict(self) -> dict:
        """Convert specification to dictionary.

        Returns
        -------
        spec_dict : dict
            Dictionary representation of the specification
        """
        return {
            "integration_time_sec": self.integration_time_sec,
            "num_integrations": self.num_integrations,
            "total_duration_sec": self.total_duration_sec,
            "num_subbands": self.num_subbands,
            "channels_per_subband": self.channels_per_subband,
            "channel_width_hz": self.channel_width_hz,
            "subband_width_hz": self.subband_width_hz,
            "total_bandwidth_hz": self.total_bandwidth_hz,
            "freq_min_hz": self.freq_min_hz,
            "freq_max_hz": self.freq_max_hz,
            "reference_freq_hz": self.reference_freq_hz,
            "image_size_px": self.image_size_px,
            "pixel_scale_arcsec": self.pixel_scale_arcsec,
            "fov_deg": self.fov_deg,
            "weighting": self.weighting,
            "robust": self.robust,
            "typical_beam_fwhm_arcsec": self.typical_beam_fwhm_arcsec,
            "typical_noise_mjy": self.typical_noise_mjy,
            "num_antennas": self.num_antennas,
        }


# Standard tile specification (singleton instance)
STANDARD_TILE = TileSpecification()


# Quality-specific variants
QUICK_TILE = TileSpecification(
    image_size_px=512,
    pixel_scale_arcsec=12.0,
)

HIGH_PRECISION_TILE = TileSpecification(
    image_size_px=4096,
    pixel_scale_arcsec=3.0,
)
