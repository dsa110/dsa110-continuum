"""
Forced photometry utilities on FITS images (PB-corrected mosaics or tiles).

Enhanced implementation with features from VAST forced_phot:
- Cluster fitting for blended sources
- Chi-squared goodness-of-fit metrics
- Optional noise maps (separate FITS files)
- Source injection for testing
- Weighted convolution (Condon 1997) for accurate flux measurement
- Numba-accelerated kernel generation and convolution (~2-5x speedup)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits  # type: ignore[reportMissingTypeStubs]

# type: ignore[reportMissingTypeStubs]
from astropy.modeling import fitting, models
from astropy.stats import mad_std
from astropy.wcs import WCS  # type: ignore[reportMissingTypeStubs]

# type: ignore[reportMissingTypeStubs]
from astropy.wcs.utils import proj_plane_pixel_scales

from dsa110_contimg.common.unified_config import settings
from dsa110_contimg.common.utils.gpu_utils import get_array_module

try:
    import scipy.spatial  # type: ignore[reportMissingTypeStubs]

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# Try to import numba for acceleration
try:
    from numba import njit, prange  # type: ignore[reportMissingTypeStubs]

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

    # Provide no-op decorators when numba is not available
    def njit(*args, **kwargs):
        """No-op decorator when numba is not available.

        Parameters
        ----------
        *args :

        **kwargs :


        """

        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    def prange(*args, **kwargs):
        """Fallback to range when numba is not available.

        Parameters
        ----------
        *args :

        **kwargs :


        """
        return range(*args)


@dataclass
class ForcedPhotometryResult:
    """Result from forced photometry measurement."""

    ra_deg: float
    dec_deg: float
    peak_jyb: float
    peak_err_jyb: float
    pix_x: float
    pix_y: float
    box_size_pix: int
    chisq: float | None = None  # Chi-squared goodness-of-fit
    dof: int | None = None  # Degrees of freedom
    # Cluster ID if part of blended source group
    cluster_id: int | None = None


# Position angle offset: VAST uses E of N convention
PA_OFFSET = 90 * u.deg


# =============================================================================
# Numba-accelerated functions (VAST-style optimization)
# =============================================================================


@njit(cache=True)
def _numba_meshgrid(xmin: int, xmax: int, ymin: int, ymax: int) -> tuple[np.ndarray, np.ndarray]:
    """Numba-accelerated meshgrid generation.

        Creates coordinate grids for kernel evaluation.

    Parameters
    ----------
    xmin : int
        Minimum X coordinate
    xmax : int
        Maximum X coordinate
    ymin : int
        Minimum Y coordinate
    ymax : int
        Maximum Y coordinate

    Returns
    -------
        None
    """
    nx = xmax - xmin
    ny = ymax - ymin
    xx = np.empty((ny, nx), dtype=np.float64)
    yy = np.empty((ny, nx), dtype=np.float64)

    for i in range(ny):
        for j in range(nx):
            xx[i, j] = xmin + j
            yy[i, j] = ymin + i

    return xx, yy


@njit(cache=True)
def _numba_gaussian_kernel(
    xx: np.ndarray,
    yy: np.ndarray,
    x0: float,
    y0: float,
    fwhm_x_pix: float,
    fwhm_y_pix: float,
    pa_deg: float,
) -> np.ndarray:
    """Numba-accelerated 2D Gaussian kernel generation.

        Computes a 2D Gaussian kernel with specified FWHM and position angle.
        Based on VAST forced_phot implementation.

    Parameters
    ----------
    xx : np.ndarray
        Coordinate meshgrid in x (pixels)
    yy : np.ndarray
        Coordinate meshgrid in y (pixels)
    x0 : float
        Kernel center x-coordinate (pixels)
    y0 : float
        Kernel center y-coordinate (pixels)
    fwhm_x_pix : float
        FWHM in x direction (pixels)
    fwhm_y_pix : float
        FWHM in y direction (pixels)
    pa_deg : float
        Position angle (degrees, E of N)

    Returns
    -------
        None
    """
    # Convert FWHM to sigma
    sigma_x = fwhm_x_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_y = fwhm_y_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # Position angle offset (E of N convention)
    pa_rad = np.deg2rad(pa_deg - 90.0)

    # Pre-compute coefficients
    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)

    a = cos_pa**2 / (2.0 * sigma_x**2) + sin_pa**2 / (2.0 * sigma_y**2)
    b = np.sin(2.0 * pa_rad) / (2.0 * sigma_x**2) - np.sin(2.0 * pa_rad) / (2.0 * sigma_y**2)
    c = sin_pa**2 / (2.0 * sigma_x**2) + cos_pa**2 / (2.0 * sigma_y**2)

    # Compute kernel
    ny, nx = xx.shape
    kernel = np.empty((ny, nx), dtype=np.float64)

    for i in range(ny):
        for j in range(nx):
            dx = xx[i, j] - x0
            dy = yy[i, j] - y0
            kernel[i, j] = np.exp(-a * dx**2 - b * dx * dy - c * dy**2)

    return kernel


@njit(cache=True)
def _numba_convolution(
    data: np.ndarray,
    noise: np.ndarray,
    kernel: np.ndarray,
) -> tuple[float, float, float]:
    """Numba-accelerated weighted convolution (Condon 1997).

        Computes flux using optimal weighting by noise map.

    Parameters
    ----------
    data : np.ndarray
        Background-subtracted data (1D flattened or 2D)
    noise : np.ndarray
        Noise map (RMS, same shape as data)
    kernel : np.ndarray
        2D Gaussian kernel (same shape as data)

    Returns
    -------
        None
    """
    # Flatten arrays if needed
    d = data.ravel()
    n = noise.ravel()
    k = kernel.ravel()

    # Compute weighted sums
    flux_num = 0.0
    flux_denom = 0.0
    err_num = 0.0
    err_denom = 0.0

    for i in range(len(d)):
        n2 = n[i] * n[i]
        if n2 > 0:
            w = k[i] / n2
            flux_num += d[i] * w
            flux_denom += k[i] * w
            err_num += n[i] * w
            err_denom += w

    if flux_denom == 0:
        return np.nan, np.nan, np.nan

    flux = flux_num / flux_denom
    flux_err = err_num / err_denom

    # Compute chi-squared
    chisq = 0.0
    for i in range(len(d)):
        if n[i] > 0:
            residual = (d[i] - k[i] * flux) / n[i]
            chisq += residual * residual

    return flux, flux_err, chisq


def _world_to_pixel(
    wcs: WCS,
    ra_deg: float,
    dec_deg: float,
) -> tuple[float, float]:
    xy = wcs.world_to_pixel_values(ra_deg, dec_deg)
    # astropy WCS: returns (x, y) with 0-based pixel coordinates
    return float(xy[0]), float(xy[1])


class G2D:
    """2D Gaussian kernel for forced photometry.

    Generates a 2D Gaussian kernel with specified FWHM and position angle.
    Used for weighted convolution flux measurement (Condon 1997).

    """

    def __init__(
        self,
        x0: float,
        y0: float,
        fwhm_x: float,
        fwhm_y: float,
        pa: float | u.Quantity,
    ):
        """Initialize 2D Gaussian kernel.

        Parameters
        ----------
        x0 : float
            Mean x coordinate (pixels)
        y0 : float
            Mean y coordinate (pixels)
        fwhm_x : float
            FWHM in x direction (pixels)
        fwhm_y : float
            FWHM in y direction (pixels)
        pa : float or Quantity
            Position angle (East of North) in degrees or Quantity
        """
        self.x0 = x0
        self.y0 = y0
        self.fwhm_x = fwhm_x
        self.fwhm_y = fwhm_y
        # Convert PA to radians, adjust for E of N convention
        if isinstance(pa, u.Quantity):
            pa_rad = (pa - PA_OFFSET).to(u.rad).value
        else:
            pa_rad = np.deg2rad(pa - PA_OFFSET.value)
        self.pa = pa_rad

        # Convert FWHM to sigma
        self.sigma_x = self.fwhm_x / 2 / np.sqrt(2 * np.log(2))
        self.sigma_y = self.fwhm_y / 2 / np.sqrt(2 * np.log(2))

        # Pre-compute coefficients for efficiency
        self.a = (
            np.cos(self.pa) ** 2 / 2 / self.sigma_x**2 + np.sin(self.pa) ** 2 / 2 / self.sigma_y**2
        )
        self.b = (
            np.sin(2 * self.pa) / 2 / self.sigma_x**2 - np.sin(2 * self.pa) / 2 / self.sigma_y**2
        )
        self.c = (
            np.sin(self.pa) ** 2 / 2 / self.sigma_x**2 + np.cos(self.pa) ** 2 / 2 / self.sigma_y**2
        )

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate kernel at given pixel coordinates.

        Parameters
        ----------
        x : array_like
            X coordinates (pixels)
        y : array_like
            Y coordinates (pixels)

        Returns
        -------
            array_like
            Kernel values at (x, y)
        """
        return np.exp(
            -self.a * (x - self.x0) ** 2
            - self.b * (x - self.x0) * (y - self.y0)
            - self.c * (y - self.y0) ** 2
        )


def _weighted_convolution(
    data: np.ndarray,
    noise: np.ndarray,
    kernel: np.ndarray,
    *,
    use_numba: bool = True,
    prefer_gpu: bool | None = None,
    min_elements: int | None = None,
) -> tuple[float, float, float]:
    """Calculate flux using weighted convolution (Condon 1997).

        Uses numba-accelerated implementation when available for ~2-5x speedup.

    Parameters
    ----------
    data : np.ndarray
        Background-subtracted data
    noise : np.ndarray
        Noise map (RMS)
    kernel : np.ndarray
        2D Gaussian kernel
    use_numba : bool, optional
        If True and numba is available, use accelerated version (default True)
    prefer_gpu : bool or None, optional
        Prefer GPU acceleration if available (default None)
    min_elements : int or None, optional
        Minimum number of elements to trigger acceleration (default None)

    Returns
    -------
        None
    """
    prefer_gpu = settings.gpu.prefer_gpu if prefer_gpu is None else prefer_gpu
    min_elements = settings.gpu.min_array_size if min_elements is None else min_elements

    # Try GPU vectorized path when available
    xp, is_gpu = get_array_module(prefer_gpu=prefer_gpu, min_elements=min_elements)
    if is_gpu and data.size >= min_elements:
        data_xp = xp.asarray(data)
        noise_xp = xp.asarray(noise)
        kernel_xp = xp.asarray(kernel)
        kernel_n2 = kernel_xp / noise_xp**2
        flux = (data_xp * kernel_n2).sum() / ((kernel_xp**2 / noise_xp**2).sum())
        flux_err = (noise_xp * kernel_n2).sum() / kernel_n2.sum()
        chisq = (((data_xp - kernel_xp * flux) / noise_xp) ** 2).sum()
        to_host = xp.asnumpy if hasattr(xp, "asnumpy") else np.asarray
        return float(to_host(flux)), float(to_host(flux_err)), float(to_host(chisq))

    if use_numba and HAVE_NUMBA:
        # Numba requires native byte order - convert if needed (e.g., WSClean outputs big-endian >f4)
        if data.dtype.byteorder not in ("=", "|", "<" if np.little_endian else ">"):
            data = data.astype(data.dtype.newbyteorder("="), copy=False)
        if noise.dtype.byteorder not in ("=", "|", "<" if np.little_endian else ">"):
            noise = noise.astype(noise.dtype.newbyteorder("="), copy=False)
        if kernel.dtype.byteorder not in ("=", "|", "<" if np.little_endian else ">"):
            kernel = kernel.astype(kernel.dtype.newbyteorder("="), copy=False)
        # Use numba-accelerated version
        return _numba_convolution(data, noise, kernel)

    # Fallback to numpy implementation
    kernel_n2 = kernel / noise**2
    flux = ((data) * kernel_n2).sum() / (kernel**2 / noise**2).sum()
    flux_err = ((noise) * kernel_n2).sum() / kernel_n2.sum()
    chisq = (((data - kernel * flux) / noise) ** 2).sum()

    return float(flux), float(flux_err), float(chisq)


def generate_kernel(
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    x0: float,
    y0: float,
    fwhm_x_pix: float,
    fwhm_y_pix: float,
    pa_deg: float,
    *,
    use_numba: bool = True,
) -> np.ndarray:
    """Generate 2D Gaussian kernel with optional numba acceleration.

    Parameters
    ----------
    xmin : int
        X coordinate minimum
    xmax : int
        X coordinate maximum
    ymin : int
        Y coordinate minimum
    ymax : int
        Y coordinate maximum
    x0 : float
        Kernel center x-coordinate (pixels)
    y0 : float
        Kernel center y-coordinate (pixels)
    fwhm_x_pix : float
        FWHM in x direction (pixels)
    fwhm_y_pix : float
        FWHM in y direction (pixels)
    pa_deg : float
        Position angle in degrees
    use_numba : bool, optional
        If True and numba is available, use accelerated version (default is True)

    """
    if use_numba and HAVE_NUMBA:
        xx, yy = _numba_meshgrid(xmin, xmax, ymin, ymax)
        return _numba_gaussian_kernel(xx, yy, x0, y0, fwhm_x_pix, fwhm_y_pix, pa_deg)

    # Fallback to numpy implementation
    x_coords = np.arange(xmin, xmax)
    y_coords = np.arange(ymin, ymax)
    xx, yy = np.meshgrid(x_coords, y_coords)
    g = G2D(x0, y0, fwhm_x_pix, fwhm_y_pix, pa_deg)
    return g(xx, yy)


def _identify_clusters(
    X0: np.ndarray,
    Y0: np.ndarray,
    threshold_pixels: float,
) -> tuple[dict[int, set], list[int]]:
    """Identify clusters of sources using KDTree.

    Parameters
    ----------
    X0 : np.ndarray
        X pixel coordinates
    Y0 : np.ndarray
        Y pixel coordinates
    threshold_pixels : float
        Distance threshold in pixels

    """
    if not HAVE_SCIPY:
        return {}, []

    if threshold_pixels <= 0 or len(X0) == 0:
        return {}, []

    tree = scipy.spatial.KDTree(np.c_[X0, Y0])
    clusters: dict[int, set] = {}

    for i in range(len(X0)):
        dists, indices = tree.query(
            np.c_[X0[i], Y0[i]],
            k=min(10, len(X0)),
            distance_upper_bound=threshold_pixels,
        )
        indices = indices[~np.isinf(dists)]
        if len(indices) > 1:
            # Check if any indices are already in a cluster
            existing_cluster = None
            for idx in indices:
                for cluster_id, members in clusters.items():
                    if idx in members:
                        existing_cluster = cluster_id
                        break
                if existing_cluster is not None:
                    break

            if existing_cluster is not None:
                # Add all indices to existing cluster
                for idx in indices:
                    clusters[existing_cluster].add(idx)
            else:
                # Create new cluster
                clusters[i] = set(indices)

    in_cluster = sorted(list(chain.from_iterable(clusters.values())))
    return clusters, in_cluster


def measure_forced_peak(
    fits_path: str,
    ra_deg: float,
    dec_deg: float,
    *,
    box_size_pix: int = 5,
    annulus_pix: tuple[int, int] = (30, 50),
    noise_map_path: str | None = None,
    background_map_path: str | None = None,
    nbeam: float = 3.0,
    use_weighted_convolution: bool = True,
) -> ForcedPhotometryResult:
    """Measure flux using forced photometry with optional weighted convolution.

        Uses weighted convolution (Condon 1997) when beam information is available,
        otherwise falls back to simple peak measurement.

    Parameters
    ----------
    fits_path : str
        Path to FITS image
    ra_deg : float
        Right ascension in degrees
    dec_deg : float
        Declination in degrees
    box_size_pix : int, optional
        Size of measurement box in pixels (used for simple peak mode), default is 5
    annulus_pix : tuple of int, optional
        Annulus for RMS estimation (r_in, r_out) in pixels, default is (30, 50)
    noise_map_path : str or None, optional
        Path to noise map FITS file, default is None
    background_map_path : str or None, optional
        Path to background map FITS file, default is None
    nbeam : float, optional
        Size of cutout in units of beam major axis (for weighted convolution), default is 3.0
    use_weighted_convolution : bool, optional
        Use weighted convolution if beam info available, default is True

    """
    p = Path(fits_path)
    if not p.exists():
        return ForcedPhotometryResult(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            peak_jyb=float("nan"),
            peak_err_jyb=float("nan"),
            pix_x=float("nan"),
            pix_y=float("nan"),
            box_size_pix=box_size_pix,
        )

    # Load data/header
    hdr = fits.getheader(p)
    data = np.asarray(fits.getdata(p)).squeeze()

    # Load background if provided
    if background_map_path:
        bg_data = np.asarray(fits.getdata(background_map_path)).squeeze()
        if bg_data.shape != data.shape:
            raise ValueError(f"Background map shape {bg_data.shape} != image shape {data.shape}")
        data = data - bg_data

    # Load noise map if provided
    noise_map = None
    if noise_map_path:
        noise_path = Path(noise_map_path)
        if noise_path.exists():
            noise_map = np.asarray(fits.getdata(noise_path)).squeeze()
            if noise_map.shape != data.shape:
                raise ValueError(f"Noise map shape {noise_map.shape} != image shape {data.shape}")
            # Convert zero-valued noise pixels to NaN
            noise_map[noise_map == 0] = np.nan

    # Use celestial 2D WCS
    wcs = WCS(hdr).celestial
    x0, y0 = _world_to_pixel(wcs, ra_deg, dec_deg)

    # Check for invalid coordinates
    if not (np.isfinite(x0) and np.isfinite(y0)):
        return ForcedPhotometryResult(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            peak_jyb=float("nan"),
            peak_err_jyb=float("nan"),
            pix_x=x0,
            pix_y=y0,
            box_size_pix=box_size_pix,
        )

    # Check if we can use weighted convolution
    has_beam_info = "BMAJ" in hdr and "BMIN" in hdr and "BPA" in hdr and use_weighted_convolution

    if has_beam_info:
        # Use weighted convolution method
        pixelscale = (proj_plane_pixel_scales(wcs)[1] * u.deg).to(u.arcsec)

        # Some generators (tests) store BMAJ/BMIN in arcsec instead of degrees.
        # Heuristic: values > 2 likely given in arcsec; else assume degrees.
        def _to_arcsec(v: float) -> float:
            try:
                val = float(v)
            except (TypeError, ValueError):
                return float("nan")
            return val if val > 2.0 else val * 3600.0

        bmaj_arcsec = _to_arcsec(hdr["BMAJ"])  # Accept deg or arcsec
        bmin_arcsec = _to_arcsec(hdr["BMIN"])  # Accept deg or arcsec
        bpa_deg = hdr.get("BPA", 0.0)

        # Calculate cutout size in pixels
        npix = int(round((nbeam / 2.0) * bmaj_arcsec / pixelscale.value))
        cx, cy = int(round(x0)), int(round(y0))
        xmin = max(0, cx - npix)
        xmax = min(data.shape[-1], cx + npix + 1)
        ymin = max(0, cy - npix)
        ymax = min(data.shape[-2], cy + npix + 1)

        # Extract cutout
        sl = (slice(ymin, ymax), slice(xmin, xmax))
        cutout_data = data[sl]
        cutout_noise = noise_map[sl] if noise_map is not None else None

        # Generate kernel
        fwhm_x_pix = bmaj_arcsec / pixelscale.value
        fwhm_y_pix = bmin_arcsec / pixelscale.value
        x_coords = np.arange(xmin, xmax)
        y_coords = np.arange(ymin, ymax)
        xx, yy = np.meshgrid(x_coords, y_coords)
        g = G2D(x0, y0, fwhm_x_pix, fwhm_y_pix, bpa_deg)
        kernel = g(xx, yy)

        # Calculate noise if not provided
        if cutout_noise is None:
            # Use annulus-based RMS
            h, w = data.shape[-2], data.shape[-1]
            yy_full, xx_full = np.ogrid[0:h, 0:w]
            r = np.sqrt((xx_full - cx) ** 2 + (yy_full - cy) ** 2)
            rin, rout = annulus_pix
            ann = (r >= rin) & (r <= rout)
            vals = data[ann]
            finite_vals = vals[np.isfinite(vals)]
            if finite_vals.size == 0:
                rms = float("nan")
            else:
                m = np.median(finite_vals)
                s = 1.4826 * np.median(np.abs(finite_vals - m))
                mask = (finite_vals > (m - 3 * s)) & (finite_vals < (m + 3 * s))
                rms = float(np.std(finite_vals[mask])) if np.any(mask) else float("nan")
            cutout_noise = np.full_like(cutout_data, rms)

        # Filter NaN pixels
        good = np.isfinite(cutout_data) & np.isfinite(cutout_noise) & np.isfinite(kernel)
        if good.sum() == 0:
            return ForcedPhotometryResult(
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                peak_jyb=float("nan"),
                peak_err_jyb=float("nan"),
                pix_x=x0,
                pix_y=y0,
                box_size_pix=box_size_pix,
            )

        cutout_data_good = cutout_data[good]
        cutout_noise_good = cutout_noise[good]
        kernel_good = kernel[good]

        # Weighted convolution
        flux, flux_err, chisq = _weighted_convolution(
            cutout_data_good, cutout_noise_good, kernel_good
        )
        dof = int(good.sum() - 1)

        return ForcedPhotometryResult(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            peak_jyb=flux,
            peak_err_jyb=flux_err,
            pix_x=x0,
            pix_y=y0,
            box_size_pix=box_size_pix,
            chisq=chisq,
            dof=dof,
        )

    else:
        # Fall back to simple peak measurement (original method)
        cx, cy = int(round(x0)), int(round(y0))
        half = max(1, box_size_pix // 2)
        x1, x2 = cx - half, cx + half
        y1, y2 = cy - half, cy + half
        h, w = data.shape[-2], data.shape[-1]
        x1c, x2c = max(0, x1), min(w - 1, x2)
        y1c, y2c = max(0, y1), min(h - 1, y2)
        cut = data[y1c : y2c + 1, x1c : x2c + 1]
        finite_cut = cut[np.isfinite(cut)]
        peak = float(np.max(finite_cut)) if finite_cut.size > 0 else float("nan")

        # Local RMS in annulus
        rin, rout = annulus_pix
        yy, xx = np.ogrid[0:h, 0:w]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        ann = (r >= rin) & (r <= rout)
        vals = data[ann]
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size == 0:
            rms = float("nan")
        else:
            m = np.median(finite_vals)
            # REFACTOR: Use astropy.stats.mad_std
            s = mad_std(finite_vals, ignore_nan=True)
            mask = (finite_vals > (m - 3 * s)) & (finite_vals < (m + 3 * s))
            rms = float(np.std(finite_vals[mask])) if np.any(mask) else float("nan")

        return ForcedPhotometryResult(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            peak_jyb=peak,
            peak_err_jyb=rms,
            pix_x=x0,
            pix_y=y0,
            box_size_pix=box_size_pix,
        )


def _measure_cluster(
    fits_path: str,
    positions: list[tuple[float, float]],
    wcs: WCS,
    data: np.ndarray,
    noise_map: np.ndarray | None,
    hdr: fits.Header,
    nbeam: float = 3.0,
    annulus_pix: tuple[int, int] = (30, 50),
) -> list[ForcedPhotometryResult]:
    """Measure flux for a cluster of blended sources using simultaneous fitting.

    Parameters
    ----------
    fits_path : str
        Path to FITS image (for error messages)
    positions : list of tuple of float
        List of (ra_deg, dec_deg) tuples
    wcs : WCS
        WCS object
    data : np.ndarray
        Image data array
    noise_map : np.ndarray or None
        Optional noise map array
    hdr : fits.Header
        FITS header
    nbeam : float, optional
        Size of cutout in units of beam major axis, default is 3.0
    annulus_pix : tuple of int, optional
        Annulus for RMS estimation (r_in, r_out) in pixels, default is (30, 50)

    """
    if not ("BMAJ" in hdr and "BMIN" in hdr and "BPA" in hdr):
        # Fall back to individual measurements
        return [
            measure_forced_peak(fits_path, ra, dec, nbeam=nbeam, annulus_pix=annulus_pix)
            for ra, dec in positions
        ]

    pixelscale = (proj_plane_pixel_scales(wcs)[1] * u.deg).to(u.arcsec)
    bmaj_arcsec = hdr["BMAJ"] * 3600.0
    bmin_arcsec = hdr["BMIN"] * 3600.0
    bpa_deg = hdr.get("BPA", 0.0)

    # Convert positions to pixels
    X0 = []
    Y0 = []
    for ra, dec in positions:
        x, y = _world_to_pixel(wcs, ra, dec)
        X0.append(x)
        Y0.append(y)
    X0 = np.array(X0)
    Y0 = np.array(Y0)

    # Calculate cutout bounds
    npix = int(round((nbeam / 2.0) * bmaj_arcsec / pixelscale.value))
    xmin = max(0, int(round((X0 - npix).min())))
    xmax = min(data.shape[-1], int(round((X0 + npix).max())) + 1)
    ymin = max(0, int(round((Y0 - npix).min())))
    ymax = min(data.shape[-2], int(round((Y0 + npix).max())) + 1)

    # Extract cutout
    sl = (slice(ymin, ymax), slice(xmin, xmax))
    cutout_data = data[sl]
    cutout_noise = noise_map[sl] if noise_map is not None else None

    # Calculate noise if not provided
    if cutout_noise is None:
        cx, cy = int(round(X0.mean())), int(round(Y0.mean()))
        h, w = data.shape[-2], data.shape[-1]
        yy_full, xx_full = np.ogrid[0:h, 0:w]
        r = np.sqrt((xx_full - cx) ** 2 + (yy_full - cy) ** 2)
        rin, rout = annulus_pix
        ann = (r >= rin) & (r <= rout)
        vals = data[ann]
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size == 0:
            rms = float("nan")
        else:
            m = np.median(finite_vals)
            # REFACTOR: Use astropy.stats.mad_std
            s = mad_std(finite_vals, ignore_nan=True)
            mask = (finite_vals > (m - 3 * s)) & (finite_vals < (m + 3 * s))
            rms = float(np.std(finite_vals[mask])) if np.any(mask) else float("nan")
        cutout_noise = np.full_like(cutout_data, rms)

    # Create meshgrid for cutout
    x_coords = np.arange(xmin, xmax)
    y_coords = np.arange(ymin, ymax)
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Build composite model with fixed positions/shapes
    fwhm_x_pix = bmaj_arcsec / pixelscale.value
    fwhm_y_pix = bmin_arcsec / pixelscale.value
    sigma_x = fwhm_x_pix / 2 / np.sqrt(2 * np.log(2))
    sigma_y = fwhm_y_pix / 2 / np.sqrt(2 * np.log(2))
    pa_rad = np.deg2rad(bpa_deg - PA_OFFSET.value)

    composite_model = None
    for i, (x0, y0) in enumerate(zip(X0, Y0)):
        g = models.Gaussian2D(
            amplitude=1.0,  # Will be fitted
            x_mean=x0,
            y_mean=y0,
            x_stddev=sigma_x,
            y_stddev=sigma_y,
            theta=pa_rad,
            fixed={
                "x_mean": True,
                "y_mean": True,
                "x_stddev": True,
                "y_stddev": True,
                "theta": True,
            },
        )
        if composite_model is None:
            composite_model = g
        else:
            composite_model = composite_model + g

    # Filter NaN pixels
    good = np.isfinite(cutout_data) & np.isfinite(cutout_noise)
    if good.sum() == 0:
        return [
            ForcedPhotometryResult(
                ra_deg=ra,
                dec_deg=dec,
                peak_jyb=float("nan"),
                peak_err_jyb=float("nan"),
                pix_x=x,
                pix_y=y,
                box_size_pix=int(npix * 2),
            )
            for (ra, dec), x, y in zip(positions, X0, Y0)
        ]

    # Fit model
    fitter = fitting.LevMarLSQFitter()
    try:
        fitted_model = fitter(
            composite_model,
            xx[good],
            yy[good],
            cutout_data[good],
            weights=1.0 / cutout_noise[good] ** 2,
        )
        model = fitted_model(xx, yy)
        chisq_total = (((cutout_data[good] - model[good]) / cutout_noise[good]) ** 2).sum()
        dof_total = int(good.sum() - len(positions))
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        # Fit failed, return NaN results
        return [
            ForcedPhotometryResult(
                ra_deg=ra,
                dec_deg=dec,
                peak_jyb=float("nan"),
                peak_err_jyb=float("nan"),
                pix_x=x,
                pix_y=y,
                box_size_pix=int(npix * 2),
            )
            for (ra, dec), x, y in zip(positions, X0, Y0)
        ]

    # Extract fluxes and errors
    results = []
    for i, ((ra, dec), x, y) in enumerate(zip(positions, X0, Y0)):
        if i == 0:
            flux = fitted_model.amplitude_0.value
        else:
            flux = getattr(fitted_model, f"amplitude_{i}").value

        # Error estimated from noise map at source position
        cy_idx = int(round(y - ymin))
        cx_idx = int(round(x - xmin))
        if 0 <= cy_idx < cutout_noise.shape[0] and 0 <= cx_idx < cutout_noise.shape[1]:
            flux_err = float(cutout_noise[cy_idx, cx_idx])
        else:
            flux_err = float("nan")

        results.append(
            ForcedPhotometryResult(
                ra_deg=ra,
                dec_deg=dec,
                peak_jyb=float(flux),
                peak_err_jyb=flux_err,
                pix_x=x,
                pix_y=y,
                box_size_pix=int(npix * 2),
                chisq=chisq_total,
                dof=dof_total,
                cluster_id=0,  # All sources in same cluster
            )
        )

    return results


def measure_many(
    fits_path: str,
    coords: list[tuple[float, float]],
    *,
    box_size_pix: int = 5,
    annulus_pix: tuple[int, int] = (30, 50),
    noise_map_path: str | None = None,
    background_map_path: str | None = None,
    use_cluster_fitting: bool = False,
    cluster_threshold: float = 1.5,
    nbeam: float = 3.0,
) -> list[ForcedPhotometryResult]:
    """Measure flux for multiple sources with optional cluster fitting.

    Parameters
    ----------
    fits_path : str
        Path to FITS image
    coords : list of tuple of float
        List of (ra_deg, dec_deg) tuples
    box_size_pix : int, optional
        Size of measurement box for simple peak mode, default is 5
    annulus_pix : tuple of int, optional
        Annulus for RMS estimation (r_in, r_out) in pixels, default is (30, 50)
    noise_map_path : str or None, optional
        Path to noise map FITS file, default is None
    background_map_path : str or None, optional
        Path to background map FITS file, default is None
    use_cluster_fitting : bool, optional
        Enable cluster fitting for blended sources, default is False
    cluster_threshold : float, optional
        Cluster threshold in units of BMAJ, default is 1.5
    nbeam : float, optional
        Size of cutout in units of beam major axis, default is 3.0

    """
    if len(coords) == 0:
        return []

    # Load data once
    p = Path(fits_path)
    if not p.exists():
        return [
            ForcedPhotometryResult(
                ra_deg=ra,
                dec_deg=dec,
                peak_jyb=float("nan"),
                peak_err_jyb=float("nan"),
                pix_x=float("nan"),
                pix_y=float("nan"),
                box_size_pix=box_size_pix,
            )
            for ra, dec in coords
        ]

    hdr = fits.getheader(p)
    data = np.asarray(fits.getdata(p)).squeeze()

    # Load background if provided
    if background_map_path:
        bg_data = np.asarray(fits.getdata(background_map_path)).squeeze()
        data = data - bg_data

    # Load noise map if provided
    noise_map = None
    if noise_map_path:
        noise_path = Path(noise_map_path)
        if noise_path.exists():
            noise_map = np.asarray(fits.getdata(noise_path)).squeeze()
            noise_map[noise_map == 0] = np.nan

    wcs = WCS(hdr).celestial

    # Check if cluster fitting is enabled and beam info available
    if use_cluster_fitting and HAVE_SCIPY and "BMAJ" in hdr:
        # Identify clusters
        X0 = []
        Y0 = []
        for ra, dec in coords:
            x, y = _world_to_pixel(wcs, ra, dec)
            X0.append(x)
            Y0.append(y)
        X0 = np.array(X0)
        Y0 = np.array(Y0)

        pixelscale = (proj_plane_pixel_scales(wcs)[1] * u.deg).to(u.arcsec)
        bmaj_arcsec = hdr["BMAJ"] * 3600.0
        threshold_pixels = cluster_threshold * (bmaj_arcsec / pixelscale.value)

        clusters, in_cluster = _identify_clusters(X0, Y0, threshold_pixels)

        # Measure individual sources (not in clusters)
        results: list[ForcedPhotometryResult] = []
        cluster_results: dict[int, list[ForcedPhotometryResult]] = {}

        for i, (ra, dec) in enumerate(coords):
            if i not in in_cluster:
                # Individual measurement
                result = measure_forced_peak(
                    fits_path,
                    ra,
                    dec,
                    box_size_pix=box_size_pix,
                    annulus_pix=annulus_pix,
                    noise_map_path=noise_map_path,
                    background_map_path=background_map_path,
                    nbeam=nbeam,
                )
                results.append(result)

        # Measure clusters
        for cluster_id, members in clusters.items():
            cluster_positions = [coords[i] for i in members]
            cluster_result = _measure_cluster(
                fits_path,
                cluster_positions,
                wcs,
                data,
                noise_map,
                hdr,
                nbeam=nbeam,
                annulus_pix=annulus_pix,
            )
            # Assign cluster IDs
            for j, member_idx in enumerate(members):
                cluster_result[j].cluster_id = cluster_id
            cluster_results[cluster_id] = cluster_result

        # Combine results in original order
        final_results: list[ForcedPhotometryResult] = []
        {cluster_id: 0 for cluster_id in clusters.keys()}
        for i, (ra, dec) in enumerate(coords):
            if i not in in_cluster:
                # Find individual result
                for r in results:
                    if abs(r.ra_deg - ra) < 1e-6 and abs(r.dec_deg - dec) < 1e-6:
                        final_results.append(r)
                        break
            else:
                # Find cluster result
                for cluster_id, members in clusters.items():
                    if i in members:
                        idx = list(members).index(i)
                        final_results.append(cluster_results[cluster_id][idx])
                        break

        return final_results

    else:
        # Simple individual measurements
        return [
            measure_forced_peak(
                fits_path,
                ra,
                dec,
                box_size_pix=box_size_pix,
                annulus_pix=annulus_pix,
                noise_map_path=noise_map_path,
                background_map_path=background_map_path,
                nbeam=nbeam,
            )
            for ra, dec in coords
        ]


def inject_source(
    fits_path: str,
    ra_deg: float,
    dec_deg: float,
    flux_jy: float,
    *,
    output_path: str | None = None,
    nbeam: float = 15.0,
) -> str:
    """Inject a fake source into a FITS image for testing.

    Parameters
    ----------
    fits_path : str
        Path to input FITS image
    ra_deg : float
        Right ascension in degrees
    dec_deg : float
        Declination in degrees
    flux_jy : float
        Flux to inject in Jy/beam
    output_path : str or None, optional
        Optional output path; if None, overwrites input file (default is None)
    nbeam : float, optional
        Size of injection region in units of beam major axis, default is 15.0

    """
    p = Path(fits_path)
    if not p.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    # Load data/header
    hdul = fits.open(fits_path, mode="update" if output_path is None else "readonly")
    hdr = hdul[0].header
    data = hdul[0].data.squeeze()

    # Check for beam info
    if not ("BMAJ" in hdr and "BMIN" in hdr and "BPA" in hdr):
        raise ValueError("FITS header missing BMAJ, BMIN, or BPA keywords")

    wcs = WCS(hdr).celestial
    x0, y0 = _world_to_pixel(wcs, ra_deg, dec_deg)

    if not (np.isfinite(x0) and np.isfinite(y0)):
        raise ValueError(f"Invalid coordinates: ({ra_deg}, {dec_deg})")

    pixelscale = (proj_plane_pixel_scales(wcs)[1] * u.deg).to(u.arcsec)
    bmaj_arcsec = hdr["BMAJ"] * 3600.0
    bmin_arcsec = hdr["BMIN"] * 3600.0
    bpa_deg = hdr.get("BPA", 0.0)

    # Calculate cutout bounds
    npix = int(round((nbeam / 2.0) * bmaj_arcsec / pixelscale.value))
    xmin = max(0, int(round(x0 - npix)))
    xmax = min(data.shape[-1], int(round(x0 + npix)) + 1)
    ymin = max(0, int(round(y0 - npix)))
    ymax = min(data.shape[-2], int(round(y0 + npix)) + 1)

    # Generate kernel
    fwhm_x_pix = bmaj_arcsec / pixelscale.value
    fwhm_y_pix = bmin_arcsec / pixelscale.value
    x_coords = np.arange(xmin, xmax)
    y_coords = np.arange(ymin, ymax)
    xx, yy = np.meshgrid(x_coords, y_coords)
    g = G2D(x0, y0, fwhm_x_pix, fwhm_y_pix, bpa_deg)
    kernel = g(xx, yy)

    # Inject source
    sl = (slice(ymin, ymax), slice(xmin, xmax))
    data[sl] = data[sl] + kernel * flux_jy

    # Update HDU
    hdul[0].data = data.reshape(hdul[0].data.shape)

    # Write output
    if output_path:
        hdul.writeto(output_path, overwrite=True)
        hdul.close()
        return output_path
    else:
        hdul.flush()
        hdul.close()
        return fits_path
