"""
DSA-110 Simulation Harness
==========================

Cloud-safe simulation harness for generating realistic DSA-110 UVH5 subband
data without access to H17 infrastructure or real HDF5 files.

The harness produces all 16 subbands of a single ~5-minute drift-scan tile:
  - 16 subbands × 48 channels (768 total channels, 1311–1498 MHz)
  - 24 integrations × 12.885 s ≈ 5.08 min
  - N antennas (configurable; default 8 for fast tests, 64 for realistic)
  - Sky model: compact point sources drawn from pyradiosky, flux ~NVSS-like

Typical usage
-------------
>>> from dsa110_continuum.simulation.harness import SimulationHarness
>>> h = SimulationHarness(n_antennas=8, n_integrations=4, use_real_positions=False)  # fast unit-test mode
>>> paths = h.generate_subbands(output_dir="/tmp/sim_tile", n_subbands=2)
>>> assert len(paths) == 2

The SimulationHarness is intentionally minimal: it uses pyuvdata, pyradiosky,
and numpy only (no CASA, no matvis, no MPI, no GPU).  Accurate beam-weighted
visibility computation is the job of pyuvsim (used when available via the
`use_pyuvsim` flag), but the default path uses a simple geometric model that
produces self-consistent closure relations and is adequate for import, QA,
and integration testing.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u
from pyuvdata import UVData

logger = logging.getLogger(__name__)

# ── DSA-110 instrument constants (from docs/pipeline-specs.md) ────────────────
# OVRO location
_OVRO_LAT_DEG = 37.2339      # °N
_OVRO_LON_DEG = -118.2825    # °E
_OVRO_ALT_M   = 1222.0       # metres above sea level

_DSA110_LOCATION = EarthLocation(
    lat=_OVRO_LAT_DEG * u.deg,
    lon=_OVRO_LON_DEG * u.deg,
    height=_OVRO_ALT_M * u.m,
)

# Frequency axis: 1311.25 – 1498.75 MHz, 16 subbands × 48 channels
_N_SUBBANDS       = 16
_N_CHAN_PER_SB    = 48
_CHAN_WIDTH_HZ    = 244140.625   # 244.14 kHz = (250e6 / 1024)
# 1311.25 MHz is the *lower edge* of the first channel (from docs/pipeline-specs.md).
# The first channel *centre* is therefore _FREQ_MIN_HZ + 0.5 * _CHAN_WIDTH_HZ.
_FREQ_LOWER_EDGE_HZ = 1_311_250_000.0  # lower edge of band (Hz)
_FREQ_MIN_HZ        = _FREQ_LOWER_EDGE_HZ + 0.5 * _CHAN_WIDTH_HZ  # first channel centre
# Subband bandwidth and first channel centre for subband i:
_SB_BW_HZ         = _N_CHAN_PER_SB * _CHAN_WIDTH_HZ  # 11.71875 MHz per subband

# Temporal parameters
_INTEGRATION_SEC  = 12.884902   # seconds per integration
_N_INTEGRATIONS   = 24          # integrations per tile (~5.08 min)

# Dish diameter (for UVData telescope info only)
_DISH_DIAM_M      = 4.65

# ── Antenna positions ─────────────────────────────────────────────────────────

# Path to the canonical antenna position CSV (relative to this file).
# The CSV has 117 rows (all allocated station slots); 96 are active in
# the operational array (47 E-W + 35 N-S + 14 outriggers; Connor et al. 2025).
_DEFAULT_ANT_CSV = Path(__file__).parent / "pyuvsim" / "antennas.csv"
_DEFAULT_GEO_CSV = Path(__file__).parent / "pyuvsim" / "DSA110_Station_Coordinates.csv"


def _load_antenna_enu_from_csv(
    n_antennas: int,
    csv_path: Path | str | None = None,
) -> np.ndarray:
    """Load real DSA-110 ENU antenna positions from the station coordinates CSV.

    The CSV (``dsa110_continuum/simulation/pyuvsim/antennas.csv``) contains
    117 rows with columns ``antenna_name``, ``antenna_number``, ``east_m``,
    ``north_m``, ``up_m``.  The coordinates are projected ENU offsets from a
    common reference origin (large absolute values; NOT raw ECEF geocentric).

    We subtract the first row (DSA001) so that all positions are relative
    to DSA001, giving a local ENU frame centred on the western end of the
    E-W arm.  This is sufficient for computing baselines; the overall array
    location is handled separately by ``_enu_to_ecef``.

    Parameters
    ----------
    n_antennas:
        Number of antennas to return.  Must be ≤ 117 (the number of rows
        in the CSV).  Use 96 for the operational DSA-110 configuration.
        Use 117 to include all allocated station slots.
    csv_path:
        Path to the CSV file.  Defaults to the bundled ``antennas.csv``.

    Returns
    -------
    np.ndarray of shape (n_antennas, 3) with columns [east_m, north_m, up_m]
    relative to DSA001 (the first row in the CSV).
    """
    if csv_path is None:
        csv_path = _DEFAULT_ANT_CSV
    csv_path = Path(csv_path)

    rows: list[tuple[float, float, float]] = []
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append((
                float(row["east_m"]),
                float(row["north_m"]),
                float(row["up_m"]),
            ))

    if n_antennas > len(rows):
        raise ValueError(
            f"Requested n_antennas={n_antennas} but CSV has only {len(rows)} rows."
        )

    arr = np.array(rows[:n_antennas], dtype=float)  # shape (n_antennas, 3)

    # Subtract DSA001 so positions are local-ENU relative to the first antenna.
    arr -= arr[0]

    return arr  # columns: east_m, north_m, up_m


def load_geodetic_enu(
    n_antennas: int = 96,
    geo_csv_path: Path | str | None = None,
    ref_lat_deg: float = _OVRO_LAT_DEG,
    ref_lon_deg: float = _OVRO_LON_DEG,
    ref_alt_m: float = _OVRO_ALT_M,
) -> np.ndarray:
    """Load DSA-110 antenna positions as local ENU from geodetic coordinates.

    Reads ``DSA110_Station_Coordinates.csv`` (lat/lon/elevation per antenna in
    WGS-84), converts each antenna to ECEF via the WGS-84 ellipsoid, then
    rotates to the local ENU frame centred on the first antenna (DSA-001).

    This produces genuine local-ENU coordinates: East increases to the right,
    North increases upward in the array plane, Up is away from Earth centre.
    W-term values are correctly small (< 5 % of max baseline) for sources
    near transit, enabling WSClean to form a real synthesised beam.

    Parameters
    ----------
    n_antennas:
        Number of antennas to return (default 96, the operational set).
        Must be <= 117 (all allocated slots).
    geo_csv_path:
        Path to the geodetic CSV.  Defaults to the bundled
        ``DSA110_Station_Coordinates.csv``.
    ref_lat_deg, ref_lon_deg, ref_alt_m:
        Geodetic reference point for the local ENU frame (default: OVRO).
        The first antenna (DSA-001) is used as the origin of the ENU frame;
        these parameters define the rotation from ECEF to ENU.

    Returns
    -------
    np.ndarray of shape (n_antennas, 3)
        Columns are [east_m, north_m, up_m] in the local ENU frame, with
        DSA-001 at the origin (0, 0, 0).
    """
    if geo_csv_path is None:
        geo_csv_path = _DEFAULT_GEO_CSV
    geo_csv_path = Path(geo_csv_path)

    # WGS-84 constants
    a  = 6_378_137.0
    f  = 1.0 / 298.257_223_563
    e2 = 2 * f - f ** 2

    def geodetic_to_ecef(lat_d: float, lon_d: float, alt_m_val: float) -> np.ndarray:
        lat = np.radians(lat_d)
        lon = np.radians(lon_d)
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        x = (N + alt_m_val) * np.cos(lat) * np.cos(lon)
        y = (N + alt_m_val) * np.cos(lat) * np.sin(lon)
        z = (N * (1 - e2) + alt_m_val) * np.sin(lat)
        return np.array([x, y, z])

    # Parse CSV: skip header rows until we find lines with 'DSA-'
    stations: list[tuple[float, float, float]] = []
    with open(geo_csv_path, newline="") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            name_field = parts[1].strip()
            if not name_field.startswith("DSA-"):
                continue
            try:
                lat_v = float(parts[2])
                lon_v = float(parts[3])
                alt_v = float(parts[4]) if parts[4].strip() else ref_alt_m
            except ValueError:
                continue
            stations.append((lat_v, lon_v, alt_v))

    if n_antennas > len(stations):
        raise ValueError(
            f"Requested n_antennas={n_antennas} but CSV has only {len(stations)} rows."
        )

    # Convert all antennas to ECEF
    ecef_all = np.array(
        [geodetic_to_ecef(lat_v, lon_v, alt_v) for lat_v, lon_v, alt_v in stations[:n_antennas]],
        dtype=float,
    )  # shape (n_antennas, 3)

    # ENU rotation matrix at DSA-001 (first antenna = local origin)
    # Rotation R maps ECEF offset vector to local ENU: enu = R @ (ecef - ecef_ref)
    lat0 = np.radians(stations[0][0])
    lon0 = np.radians(stations[0][1])
    R = np.array([
        [-np.sin(lon0),                np.cos(lon0),              0.0          ],
        [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
        [ np.cos(lat0) * np.cos(lon0),  np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
    ])  # (3, 3): rows = [East, North, Up]

    ecef_ref = ecef_all[0]  # DSA-001 as ENU origin
    offsets  = ecef_all - ecef_ref  # shape (n_antennas, 3)
    enu      = offsets @ R.T  # shape (n_antennas, 3); columns = [E, N, U]

    return enu


def _enu_to_ecef(enu: np.ndarray, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Convert ENU offsets (metres) relative to OVRO to ECEF coordinates."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    # ECEF of reference point
    a = 6_378_137.0           # WGS-84 semi-major axis (m)
    f = 1.0 / 298.257_223_563 # WGS-84 flattening
    e2 = 2 * f - f ** 2
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)

    x0 = (N + alt_m) * np.cos(lat) * np.cos(lon)
    y0 = (N + alt_m) * np.cos(lat) * np.sin(lon)
    z0 = (N * (1 - e2) + alt_m) * np.sin(lat)

    # Rotation matrix ENU → ECEF
    R = np.array([
        [-np.sin(lon),              np.cos(lon),             0],
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [ np.cos(lat) * np.cos(lon),  np.cos(lat) * np.sin(lon), np.sin(lat)],
    ])
    ecef_offsets = enu @ R.T
    ecef = ecef_offsets + np.array([x0, y0, z0])
    return ecef


# ── Sky model (pyradiosky) ────────────────────────────────────────────────────

def _make_sky_model(
    n_sources: int,
    ra_center_deg: float,
    dec_center_deg: float,
    fov_deg: float,
    freq_hz: float,
    rng: np.random.Generator,
) -> "pyradiosky.SkyModel":
    """Generate a synthetic sky model with n_sources compact point sources.

    Flux densities follow a power-law distribution (consistent with NVSS
    1.4 GHz counts at mJy–Jy level).  All sources have Stokes I only.
    """
    import pyradiosky
    from astropy.coordinates import Longitude, Latitude

    # Random positions within FoV
    d_ra  = rng.uniform(-fov_deg / 2, fov_deg / 2, n_sources)
    d_dec = rng.uniform(-fov_deg / 2, fov_deg / 2, n_sources)
    ras  = (ra_center_deg  + d_ra  / np.cos(np.radians(dec_center_deg))) % 360.0
    decs = np.clip(dec_center_deg + d_dec, -90.0, 90.0)

    # Flux ~ power-law N(S) ∝ S^{-1.7} at 1.4 GHz (NVSS counts, ~mJy–Jy)
    # Sample via inverse CDF of truncated power law with index -0.7
    smin, smax = 0.005, 5.0  # Jy
    alpha = -0.7
    u_samp = rng.uniform(0.0, 1.0, n_sources)
    fluxes = (smin ** (alpha + 1) + u_samp * (smax ** (alpha + 1) - smin ** (alpha + 1))) ** (
        1.0 / (alpha + 1)
    )

    # Spectral index α ≈ -0.75 (typical extragalactic source)
    spectral_indices = rng.normal(-0.75, 0.2, n_sources)

    # Build pyradiosky SkyModel — Stokes I only, reference freq = freq_hz
    # stokes shape must be (4, 1, n_components) per pyradiosky API
    stokes_array = np.zeros((4, 1, n_sources), dtype=float)
    stokes_array[0, 0, :] = fluxes   # Stokes I

    sky = pyradiosky.SkyModel(
        name=[f"src_{i:04d}" for i in range(n_sources)],
        ra=Longitude(ras * u.deg),
        dec=Latitude(decs * u.deg),
        stokes=stokes_array * u.Jy,
        spectral_type="spectral_index",
        reference_frequency=np.full(n_sources, freq_hz) * u.Hz,
        spectral_index=spectral_indices,
        frame="icrs",
    )
    return sky


# ── Main harness ──────────────────────────────────────────────────────────────

@dataclass
class SimulationHarness:
    """Produces realistic DSA-110 UVH5 subband data without H17 or real data.

    Parameters
    ----------
    n_antennas:
        Number of antennas to simulate.  Default is **117** (all allocated
        station slots in ``antennas.csv``), which gives the correct DSA-110
        T-array geometry including all outriggers.  For fast unit tests use 8
        or 16 (with ``use_real_positions=False``).  Do **not** use 96 — the
        first 96 CSV rows cover only the E-W and N-S core and exclude all
        outriggers, which is geometrically incorrect.  Must be ≤ 117 when
        ``use_real_positions=True``.  See ``docs/GROUND_TRUTH.md`` §1.4.
    n_integrations:
        Number of time integrations per tile.  Default 24 ≈ 5-minute tile.
    pointing_ra_deg:
        Field centre RA in degrees.  Default 343.5° (near 3C454.3 at 22h 54m).
    pointing_dec_deg:
        Field centre Dec in degrees.  Default 16.15° (canary tile strip).
    n_sky_sources:
        Number of point sources injected into the sky model.
    noise_jy:
        RMS thermal noise per baseline per channel per integration in Jy.
        Typical DSA-110 value: ~1 Jy (per-baseline), mosaic ~10 mJy/beam.
    seed:
        Random seed for reproducibility.
    use_real_positions:
        If True (default), load antenna positions from the bundled
        ``antennas.csv`` (real DSA-110 T-array geometry: E-W arm, N-S arm,
        outriggers).  If False, generate synthetic 1-D east-west positions
        (legacy; only for tests that do not require realistic UV coverage).
    ant_csv_path:
        Override path to the antenna position CSV.  Only used when
        ``use_real_positions=True``.  Defaults to the bundled CSV.
    """
    n_antennas:         int   = 117
    n_integrations:     int   = 24
    pointing_ra_deg:    float = 343.5
    pointing_dec_deg:   float = 16.15
    n_sky_sources:      int   = 20
    noise_jy:           float = 1.0
    seed:               int   = 42
    use_real_positions: bool  = True
    ant_csv_path:       object = None  # Path | str | None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._location = _DSA110_LOCATION
        self._ant_enu: np.ndarray | None = None   # lazy
        self._ant_ecef: np.ndarray | None = None  # lazy

    # ── Antenna positions ────────────────────────────────────────────────────

    @property
    def antenna_enu(self) -> np.ndarray:
        """Local-ENU antenna positions (metres) relative to DSA001.

        By default (``use_real_positions=True``) these are loaded from the
        bundled ``antennas.csv``, which contains the real DSA-110 T-array
        geometry (E-W arm + N-S arm + outriggers).  Set
        ``use_real_positions=False`` to use synthetic 1-D east-west positions
        (legacy; adequate only for closure/format tests, not for imaging).
        """
        if self._ant_enu is None:
            if self.use_real_positions:
                try:
                    # Use geodetic CSV (lat/lon/el) for correct local-ENU
                    # conversion.  This avoids the large W-term problem that
                    # arose when using the projected-ECEF antennas.csv with a
                    # simple row subtraction.
                    self._ant_enu = load_geodetic_enu(
                        n_antennas=self.n_antennas,
                    )
                    logger.debug(
                        "Loaded %d real DSA-110 antenna positions from geodetic CSV",
                        self.n_antennas,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Could not load geodetic antenna positions (%s); "
                        "falling back to synthetic 1-D east-west layout.",
                        exc,
                    )
                    east  = np.sort(self._rng.uniform(0, 980.0, self.n_antennas))
                    north = self._rng.normal(0.0, 5.0, self.n_antennas)
                    up    = self._rng.normal(0.0, 0.5, self.n_antennas)
                    self._ant_enu = np.column_stack([east, north, up])
            else:
                # Explicitly requested synthetic 1-D east-west positions.
                east  = np.sort(self._rng.uniform(0, 980.0, self.n_antennas))
                north = self._rng.normal(0.0, 5.0, self.n_antennas)
                up    = self._rng.normal(0.0, 0.5, self.n_antennas)
                self._ant_enu = np.column_stack([east, north, up])
        return self._ant_enu

    @property
    def antenna_ecef(self) -> np.ndarray:
        if self._ant_ecef is None:
            self._ant_ecef = _enu_to_ecef(
                self.antenna_enu,
                _OVRO_LAT_DEG,
                _OVRO_LON_DEG,
                _OVRO_ALT_M,
            )
        return self._ant_ecef

    # ── Time axis ────────────────────────────────────────────────────────────

    def make_time_array(self, start_time: Time | None = None) -> np.ndarray:
        """Return JD times for all baseline-times (shape: n_baselines * n_integrations)."""
        if start_time is None:
            start_time = Time("2026-01-25T22:26:05", format="isot", scale="utc")
        dt = _INTEGRATION_SEC / 86400.0  # days
        unique_times = np.array([start_time.jd + i * dt for i in range(self.n_integrations)])
        n_baselines = self.n_antennas * (self.n_antennas - 1) // 2
        # Repeat each time for all baselines (time-major ordering)
        return np.repeat(unique_times, n_baselines)

    # ── Frequency axis ───────────────────────────────────────────────────────

    @staticmethod
    def subband_freqs(sb_index: int) -> np.ndarray:
        """Channel centre frequencies (Hz) for subband *sb_index* (0-based)."""
        f0 = _FREQ_MIN_HZ + sb_index * _SB_BW_HZ
        return f0 + np.arange(_N_CHAN_PER_SB) * _CHAN_WIDTH_HZ

    # ── UVW calculation ──────────────────────────────────────────────────────

    def _compute_uvw(
        self,
        ant1: np.ndarray,
        ant2: np.ndarray,
        times_jd: np.ndarray,
        phase_ra_deg: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute geometric UVW coordinates for all baselines and times.

        Uses the standard Earth rotation synthesis formula.  The phase
        centre per integration is given by *phase_ra_deg* (one value per
        integration).  For a drift-scan tile this is the LST at each
        timestamp; for a single fixed pointing it is a constant array.

        Parameters
        ----------
        ant1, ant2:
            Antenna index arrays for each baseline-time row.
        times_jd:
            Julian date for each baseline-time row.
        phase_ra_deg:
            Right ascension of the phase centre for each integration
            (shape: n_integrations).  If None, ``self.pointing_ra_deg``
            is broadcast to all integrations (legacy single-field mode).
        """
        n_blts = len(times_jd)
        uvw = np.zeros((n_blts, 3), dtype=float)

        n_baselines = self.n_antennas * (self.n_antennas - 1) // 2
        n_ints = self.n_integrations

        lsts_deg, _ = self._lst_and_ha(times_jd)
        if phase_ra_deg is None:
            phase_ra_deg = np.full(n_ints, self.pointing_ra_deg)

        dec_r  = np.radians(self.pointing_dec_deg)
        cos_d  = np.cos(dec_r)
        sin_d  = np.sin(dec_r)

        for t_idx in range(n_ints):
            # Hour angle = LST - phase_RA for this integration
            ha_deg = lsts_deg[t_idx] - phase_ra_deg[t_idx]
            ha_r   = np.radians(ha_deg)
            ch = np.cos(ha_r)
            sh = np.sin(ha_r)

            sl = slice(t_idx * n_baselines, (t_idx + 1) * n_baselines)
            a1 = ant1[sl]
            a2 = ant2[sl]
            # Baseline ECEF vector (metres)
            dx = self.antenna_ecef[a2, 0] - self.antenna_ecef[a1, 0]
            dy = self.antenna_ecef[a2, 1] - self.antenna_ecef[a1, 1]
            dz = self.antenna_ecef[a2, 2] - self.antenna_ecef[a1, 2]

            u_bl =  sh * dx + ch * dy
            v_bl = -sin_d * ch * dx + sin_d * sh * dy + cos_d * dz
            w_bl =  cos_d * ch * dx - cos_d * sh * dy + sin_d * dz
            uvw[sl, 0] = u_bl
            uvw[sl, 1] = v_bl
            uvw[sl, 2] = w_bl
        return uvw

    def _lst_and_ha(
        self, times_jd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (LST_deg, HA_deg) arrays, one entry per integration.

        HA is referenced to ``self.pointing_ra_deg`` for backward compatibility
        with code that still calls _hour_angle_deg indirectly.
        """
        n_baselines = self.n_antennas * (self.n_antennas - 1) // 2
        unique_times = times_jd[::n_baselines]   # one per integration
        t = Time(unique_times, format="jd", scale="utc")
        lst_deg = t.sidereal_time("apparent", longitude=_OVRO_LON_DEG * u.deg).deg
        ha_deg  = lst_deg - self.pointing_ra_deg
        return lst_deg, ha_deg

    def _hour_angle_deg(self, times_jd: np.ndarray) -> np.ndarray:
        """Compute hour angle (degrees) of the field centre for each unique time."""
        _, ha_deg = self._lst_and_ha(times_jd)
        return ha_deg

    # ── Visibility model ─────────────────────────────────────────────────────

    def _compute_visibilities(
        self,
        uvw: np.ndarray,
        freqs_hz: np.ndarray,
        sky: "pyradiosky.SkyModel",
        phase_ra_deg: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute visibilities from a point-source sky model.

        Returns complex64 array of shape (n_blts, n_freq, n_pol=2).
        The sky model contributes only to XX and YY (Stokes I only, so
        V_XX = V_YY = I/2).  Off-diagonal polarizations are zero.

        The visibilities are the van Cittert–Zernike theorem integral
        evaluated analytically for point sources.

        Parameters
        ----------
        uvw:
            UVW array (n_blts, 3) already computed with the per-integration
            phase centres.
        freqs_hz:
            Channel centre frequencies.
        sky:
            Point-source sky model.
        phase_ra_deg:
            Phase centre RA (degrees) per integration (shape: n_integrations).
            If None, ``self.pointing_ra_deg`` is broadcast to all integrations.
        """
        import astropy.units as u

        n_blts, _ = uvw.shape
        n_freq    = len(freqs_hz)
        n_baselines = self.n_antennas * (self.n_antennas - 1) // 2
        vis = np.zeros((n_blts, n_freq, 2), dtype=complex)  # XX, YY

        if phase_ra_deg is None:
            phase_ra_deg = np.full(self.n_integrations, self.pointing_ra_deg)

        # Source coordinates
        ra_src  = sky.ra.deg   # shape (n_src,)
        dec_src = sky.dec.deg
        dec_c   = np.radians(self.pointing_dec_deg)

        for s_idx in range(sky.Ncomponents):
            ra_s  = np.radians(ra_src[s_idx])
            dec_s = np.radians(dec_src[s_idx])

            # Source flux (with spectral index scaling)
            stokes_i = float(sky.stokes[0, 0, s_idx].to(u.Jy).value)
            ref_freq_arr = sky.reference_frequency

            for t_idx in range(self.n_integrations):
                sl   = slice(t_idx * n_baselines, (t_idx + 1) * n_baselines)
                ra_c = np.radians(phase_ra_deg[t_idx])

                # Direction cosines relative to THIS integration's phase centre
                l = np.cos(dec_s) * np.sin(ra_s - ra_c)
                m = (np.sin(dec_s) * np.cos(dec_c)
                     - np.cos(dec_s) * np.sin(dec_c) * np.cos(ra_s - ra_c))
                n_coord = np.sqrt(max(1.0 - l**2 - m**2, 0.0))

                # Phase per baseline for this integration
                uvw_sl = uvw[sl]   # shape (n_baselines, 3)
                phase_spatial = (uvw_sl[:, 0] * l
                                 + uvw_sl[:, 1] * m
                                 + uvw_sl[:, 2] * (n_coord - 1))  # (n_baselines,)

                for f_idx, freq in enumerate(freqs_hz):
                    wave = 2.998e8 / freq
                    phase = 2 * np.pi * phase_spatial / wave

                    if ref_freq_arr is not None:
                        ref_freq = float(ref_freq_arr[s_idx].to(u.Hz).value)
                        alpha    = float(sky.spectral_index[s_idx])
                        flux_jy  = stokes_i * (freq / ref_freq) ** alpha
                    else:
                        flux_jy = stokes_i

                    contrib = (flux_jy / 2.0) * np.exp(-1j * phase)  # XX = YY = I/2
                    vis[sl, f_idx, 0] += contrib
                    vis[sl, f_idx, 1] += contrib

        return vis.astype(np.complex64)

    # ── UVData builder ────────────────────────────────────────────────────────

    def _build_uvdata(
        self,
        subband_index: int,
        start_time: Time,
        sky: "pyradiosky.SkyModel",
        drift_scan: bool = True,
    ) -> UVData:
        """Build a complete UVData object for one subband (pyuvdata 3.x API).

        Strategy (two-pass):
        1. Build a skeleton UVData with unit visibilities so that pyuvdata can
           compute the UVW array from antenna positions via its own internal
           ``set_uvws_from_antenna_positions`` call (triggered automatically by
           ``UVData.new``).
        2. Read back the pyuvdata-computed UVW, feed it into the sky-model
           visibility engine, and write the result back to ``uv.data_array``.

        This avoids any disagreement between our UVW computation and pyuvdata's
        internal phase-rotation machinery.
        """
        from pyuvdata import Telescope

        freqs = self.subband_freqs(subband_index)
        n_baselines = self.n_antennas * (self.n_antennas - 1) // 2
        n_blts = n_baselines * self.n_integrations

        # Baseline pairs (lower triangle, i < j, time-major ordering)
        ants_a = np.array([i for i in range(self.n_antennas) for j in range(i + 1, self.n_antennas)])
        ants_b = np.array([j for i in range(self.n_antennas) for j in range(i + 1, self.n_antennas)])

        dt = _INTEGRATION_SEC / 86400.0
        unique_times_jd = np.array([start_time.jd + i * dt for i in range(self.n_integrations)])
        times_jd_blts = np.repeat(unique_times_jd, n_baselines)
        ant1_all      = np.tile(ants_a, self.n_integrations)
        ant2_all      = np.tile(ants_b, self.n_integrations)

        # ── Telescope object (pyuvdata 3.x) ──────────────────────────────────
        # antenna_positions must be ECEF *offsets* relative to the telescope
        # reference position (the EarthLocation ITRS geocentric metres).
        loc = self._location
        tel_ecef = np.array([
            loc.x.to(u.m).value,
            loc.y.to(u.m).value,
            loc.z.to(u.m).value,
        ])
        ant_positions_rel = self.antenna_ecef - tel_ecef  # shape (N, 3)

        tel = Telescope.new(
            name="DSA-110",
            location=self._location,
            antenna_positions=ant_positions_rel,
            antenna_names=[f"DSA{i:03d}" for i in range(self.n_antennas)],
            antenna_numbers=np.arange(self.n_antennas, dtype=int),
            instrument="DSA-110",
            x_orientation="north",
            update_from_known=False,
        )

        # ── Phase center catalog (pyuvdata 3.x) ───────────────────────────────
        # Phase centre catalogue:
        # - drift_scan=True  (default): one entry per integration at the LST
        #   for that moment, replicating the 24-field structure of real
        #   DSA-110 Measurement Sets (see docs/skills/ms-generation.md).
        # - drift_scan=False: single fixed phase centre at pointing_ra_deg
        #   (used for calibrator observations which track a known source).
        if drift_scan:
            lsts_deg = np.array([
                Time(unique_times_jd[i], format="jd", scale="utc")
                .sidereal_time("apparent", longitude=_OVRO_LON_DEG * u.deg)
                .deg
                for i in range(self.n_integrations)
            ])
            phase_cat = {
                i: {
                    "cat_name": f"FIELD_{i:02d}",
                    "cat_type": "sidereal",
                    "cat_lon": np.radians(lsts_deg[i]),
                    "cat_lat": np.radians(self.pointing_dec_deg),
                    "cat_frame": "icrs",
                    "cat_epoch": 2000.0,
                    "info_source": "user",
                }
                for i in range(self.n_integrations)
            }
            phase_id_array = np.repeat(
                np.arange(self.n_integrations, dtype=int), n_baselines
            )
        else:
            # Calibrator / fixed-pointing mode: single phase centre
            lsts_deg = np.full(self.n_integrations, self.pointing_ra_deg)
            phase_cat = {
                0: {
                    "cat_name": f"SIM_TILE_SB{subband_index:02d}",
                    "cat_type": "sidereal",
                    "cat_lon": np.radians(self.pointing_ra_deg),
                    "cat_lat": np.radians(self.pointing_dec_deg),
                    "cat_frame": "icrs",
                    "cat_epoch": 2000.0,
                    "info_source": "user",
                }
            }
            phase_id_array = np.zeros(n_blts, dtype=int)

        # ── Pass 1: skeleton UVData (unit visibilities) ───────────────────────
        # UVData.new() calls set_uvws_from_antenna_positions internally,
        # giving us a pyuvdata-consistent UVW array.
        antpairs = list(zip(ant1_all.tolist(), ant2_all.tolist()))

        uv = UVData.new(
            freq_array=freqs,
            polarization_array=np.array([-5, -6], dtype=int),  # XX, YY
            times=times_jd_blts,
            telescope=tel,
            antpairs=antpairs,
            integration_time=_INTEGRATION_SEC,
            channel_width=_CHAN_WIDTH_HZ,
            data_array=np.ones((n_blts, len(freqs), 2), dtype=np.complex64),
            flag_array=np.zeros((n_blts, len(freqs), 2), dtype=bool),
            nsample_array=np.ones((n_blts, len(freqs), 2), dtype=np.float32),
            phase_center_catalog=phase_cat,
            phase_center_id_array=phase_id_array,
            history=(
                f"Synthetic DSA-110 tile subband {subband_index} "
                f"generated by SimulationHarness (seed={self.seed})\n"
            ),
            vis_units="Jy",
            do_blt_outer=False,
            blts_are_rectangular=False,
        )

        # ── Pass 2: compute visibilities using per-integration drift-scan UVW ──
        # For each integration the phase centre is the LST at that moment
        # (drift-scan: telescope tracks the meridian, so HA=0 at every
        # timestamp and the phase centre advances in RA with the sky).
        # We pass lsts_deg both to _compute_uvw (so HA=LST-phase_RA≈0 at each
        # step) and to _compute_visibilities (so direction cosines are computed
        # relative to the correct per-integration phase centre).
        uvw = self._compute_uvw(
            ant1_all, ant2_all, times_jd_blts, phase_ra_deg=lsts_deg
        )  # shape (n_blts, 3)
        uv.uvw_array = uvw.astype(np.float64)  # overwrite with phased UVW

        vis = self._compute_visibilities(uvw, freqs, sky, phase_ra_deg=lsts_deg)

        # Add thermal noise
        if self.noise_jy > 0:
            noise = (
                self._rng.standard_normal((n_blts, len(freqs), 2))
                + 1j * self._rng.standard_normal((n_blts, len(freqs), 2))
            ).astype(np.complex64) * (self.noise_jy / np.sqrt(2.0))
            vis = vis + noise

        # Conjugate before storing: pyuvdata writes the MS UVW column as
        # -(our phased UVW) because its internal convention is ant1-ant2
        # while ours is ant2-ant1.  The MS standard requires the DATA column
        # to satisfy V(u,v,w) stored for UVW as written, so we store conj(vis)
        # here; when pyuvdata negates the UVW on MS write, the pair
        # (−UVW, conj(V)) is equivalent to (UVW, V) and WSClean images
        # correctly.  The UVH5 file stores the same conjugated visibilities
        # together with our phased UVW, so the UVH5 is internally consistent
        # with pyuvdata's read-back convention.
        uv.data_array = vis.conj().astype(np.complex64)

        # Store extra provenance in extra_keywords
        uv.extra_keywords = {
            "SIM_SEED":  self.seed,
            "SIM_NANTS": self.n_antennas,
            "SIM_NINT":  self.n_integrations,
            "SIM_NSRC":  self.n_sky_sources,
            "SIM_SBIDX": subband_index,
        }

        return uv

    # ── Public API ────────────────────────────────────────────────────────────

    def make_sky_model(
        self,
        fov_deg: float = 3.5,
        freq_hz: float | None = None,
    ) -> "pyradiosky.SkyModel":
        """Generate a synthetic sky model centred on the pointing direction.

        Parameters
        ----------
        fov_deg:
            Field of view diameter for source placement (degrees).
        freq_hz:
            Reference frequency for flux scaling (Hz).  Defaults to the
            centre of the DSA-110 band (1405 MHz).

        Returns
        -------
        pyradiosky.SkyModel
        """
        if freq_hz is None:
            freq_hz = 1_405_000_000.0  # L-band centre
        return _make_sky_model(
            n_sources=self.n_sky_sources,
            ra_center_deg=self.pointing_ra_deg,
            dec_center_deg=self.pointing_dec_deg,
            fov_deg=fov_deg,
            freq_hz=freq_hz,
            rng=self._rng,
        )

    def generate_subband(
        self,
        sb_index: int,
        output_path: Path | str,
        start_time: Time | None = None,
        sky: "pyradiosky.SkyModel | None" = None,
        drift_scan: bool = True,
    ) -> Path:
        """Generate one UVH5 subband file.

        Parameters
        ----------
        sb_index:
            Subband index (0–15).
        output_path:
            Destination path for the UVH5 file (will be created with parent dirs).
        start_time:
            Observation start time.  Defaults to the canary tile epoch
            ``2026-01-25T22:26:05`` (3C454.3 at ~12.5 Jy).
        sky:
            Sky model.  If None, one is generated automatically.
        drift_scan:
            If True (default), each integration uses its own LST as the phase
            centre — matching real DSA-110 multi-field drift-scan observations.
            If False, a single fixed phase centre at ``pointing_ra_deg`` is used
            (calibrator-observation mode).

        Returns
        -------
        Path to the written UVH5 file.
        """
        if start_time is None:
            start_time = Time("2026-01-25T22:26:05", format="isot", scale="utc")
        if sky is None:
            sky = self.make_sky_model(freq_hz=float(self.subband_freqs(sb_index).mean()))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        uv = self._build_uvdata(sb_index, start_time, sky, drift_scan=drift_scan)
        # Use pyuvdata's high-level write API
        uv.write_uvh5(str(output_path), clobber=True)
        logger.info("Wrote subband %d → %s", sb_index, output_path)
        return output_path

    def generate_subbands(
        self,
        output_dir: Path | str,
        n_subbands: int = _N_SUBBANDS,
        start_time: Time | None = None,
        filename_template: str = "sim_tile_sb{sb_index:02d}.uvh5",
        sky: "pyradiosky.SkyModel | None" = None,
        drift_scan: bool = True,
    ) -> list[Path]:
        """Generate multiple UVH5 subband files into *output_dir*.

        Parameters
        ----------
        output_dir:
            Directory to write files into (created if absent).
        n_subbands:
            How many subbands to generate (default: all 16).
        start_time:
            Observation start time.
        filename_template:
            Format string with ``{sb_index}`` placeholder.
        sky:
            Pre-built sky model to use for all subbands.  If *None* (default),
            ``make_sky_model()`` is called internally.  Pass an existing sky
            to ensure that the simulated visibilities match a ground-truth
            registry built from the same sky object — otherwise the random
            number generator will produce a different sky on the second call.
        drift_scan:
            Forwarded to :meth:`generate_subband`.  True (default) enables
            per-integration LST phase centres (drift-scan mode); False uses a
            single fixed phase centre (calibrator / fixed-pointing mode).

        Returns
        -------
        List of paths to the written UVH5 files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if start_time is None:
            start_time = Time("2026-01-25T22:26:05", format="isot", scale="utc")

        # Shared sky model across all subbands (same field).
        # Use the caller-supplied sky if provided to avoid consuming extra RNG state.
        if sky is None:
            sky = self.make_sky_model()

        paths: list[Path] = []
        for sb in range(n_subbands):
            out = output_dir / filename_template.format(sb_index=sb)
            paths.append(
                self.generate_subband(
                    sb, out, start_time=start_time, sky=sky, drift_scan=drift_scan
                )
            )
        return paths

    def generate_calibrator_subband(
        self,
        output_dir: "Path | str",
        *,
        flux_jy: float = 10.0,
        subband_index: int = 0,
    ) -> Path:
        """Generate a calibrator-observation UVH5 with one bright point source at phase centre.

        The calibrator sits exactly at (pointing_ra_deg, pointing_dec_deg) so
        all baseline phases are zero and the visibility amplitude equals
        flux_jy / 2 (XX = YY = I/2 convention).  Thermal noise is suppressed.
        This file is used by SimulatedPipeline._calibrate() to derive per-antenna
        gain solutions without requiring CASA.

        Parameters
        ----------
        output_dir:
            Directory in which to write the output file.
        flux_jy:
            Flux density of the calibrator source (Jy).  Default 10 Jy, typical
            for VLA calibrators used by DSA-110 (e.g. 3C 309.1 ≈ 9 Jy at 1.4 GHz).
        subband_index:
            Which subband frequency to simulate (0-indexed, default 0).

        Returns
        -------
        Path
            Path to the written calibrator UVH5.
        """
        import pyradiosky
        from astropy.coordinates import Longitude, Latitude
        from astropy import units

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"sim_cal_sb{subband_index:02d}.uvh5"

        # Build a single-source SkyModel exactly at phase centre
        freq_hz = float(self.subband_freqs(subband_index).mean())
        stokes = np.zeros((4, 1, 1), dtype=float)
        stokes[0, 0, 0] = flux_jy  # Stokes I only
        sky = pyradiosky.SkyModel(
            name=np.array(["SIM_CAL"]),
            ra=Longitude([self.pointing_ra_deg], unit="deg"),
            dec=Latitude([self.pointing_dec_deg], unit="deg"),
            stokes=stokes * units.Jy,
            spectral_type="spectral_index",
            reference_frequency=np.array([freq_hz]) * units.Hz,
            spectral_index=np.array([0.0]),
            frame="icrs",
        )

        # Suppress noise and background sources for a clean calibrator observation
        orig_noise = self.noise_jy
        orig_n_src = self.n_sky_sources
        self.noise_jy = 0.0
        self.n_sky_sources = 0

        try:
            self.generate_subband(subband_index, out_path, sky=sky, drift_scan=False)
        finally:
            # Always restore original state even if generate_subband raises
            self.noise_jy = orig_noise
            self.n_sky_sources = orig_n_src

        logger.info(
            "Wrote calibrator UVH5 (%.1f Jy at phase centre, sb=%d) -> %s",
            flux_jy, subband_index, out_path,
        )
        return out_path

    def load_subband(self, uvh5_path: Path | str) -> UVData:
        """Read a previously-written UVH5 subband file back into a UVData object."""
        uv = UVData()
        uv.read(str(uvh5_path))
        return uv

    # ── Validation helpers ────────────────────────────────────────────────────

    def check_closure(
        self,
        uvh5_path: Path | str,
        n_triangles: int = 5,
        freq_channel: int = 24,
    ) -> dict[str, Any]:
        """Verify closure relations for a synthetic UVH5 file.

        Closure phase is defined as::

            C_abc = arg( V_ab · V_bc · V_ca )

        where V_ij is the visibility on baseline (i→j).  All baselines are
        stored in the UVH5 file with ant1 < ant2 (lower-triangle convention),
        so V_ca = conj(V_ac) for the stored V_ac.

        For a *single* point source (which is what the SimulationHarness injects
        when ``n_sky_sources=1``), closure phase is exactly 0 at all times and
        frequencies — this is a fundamental property of the measurement equation
        for a point source.  For a multi-source sky the closure phase is in
        general non-zero.

        The harness always generates data from a sky with ``n_sky_sources``
        sources.  To run a meaningful closure-phase check, this method
        re-generates a *single-source* UVData from the stored UVH5 metadata
        and computes closure on that.

        Parameters
        ----------
        uvh5_path:
            Path to a UVH5 file previously written by the harness.
        n_triangles:
            Maximum number of antenna triangles to test.
        freq_channel:
            Which frequency channel index to evaluate closure on.

        Returns
        -------
        dict with keys ``'max_closure_phase_deg'`` and ``'passed'``.
        """
        import pyradiosky
        from astropy.coordinates import Longitude, Latitude

        uv = self.load_subband(uvh5_path)

        # Recover subband index from extra_keywords
        sb_idx = int(uv.extra_keywords.get("SIM_SBIDX", 0))

        # Build a fresh single-source sky at the phase centre
        stokes_arr = np.zeros((4, 1, 1), dtype=float)
        stokes_arr[0, 0, 0] = 1.0  # 1 Jy Stokes I
        sky_single = pyradiosky.SkyModel(
            name=["closure_test_src"],
            ra=Longitude([self.pointing_ra_deg] * u.deg),
            dec=Latitude([self.pointing_dec_deg] * u.deg),
            stokes=stokes_arr * u.Jy,
            spectral_type="spectral_index",
            spectral_index=np.array([0.0]),
            reference_frequency=np.array([self.subband_freqs(sb_idx).mean()]) * u.Hz,
            frame="icrs",
        )

        from astropy.time import Time
        # Re-derive start time from the UVData
        start_jd = float(uv.time_array.min())
        start_time = Time(start_jd, format="jd", scale="utc")

        # Build noiseless UVData for closure test: noise would break the
        # closure relation even for a single point source.
        saved_noise = self.noise_jy
        self.noise_jy = 0.0
        try:
            uv_single = self._build_uvdata(sb_idx, start_time, sky_single)
        finally:
            self.noise_jy = saved_noise

        # Pick antenna triples
        n = min(self.n_antennas, 6)
        ants = list(range(n))
        triples = [
            (ants[ai], ants[bi], ants[ci])
            for ai in range(n)
            for bi in range(ai + 1, n)
            for ci in range(bi + 1, n)
        ][:n_triangles]

        max_closure = 0.0
        for a, b, c in triples:
            # Retrieve V_ab, V_bc, V_ac (all stored with lower-index first)
            def _gv(i: int, j: int) -> complex | None:
                lo, hi = min(i, j), max(i, j)
                bl = uv_single.antnums_to_baseline(lo, hi)
                idx = np.where(uv_single.baseline_array == bl)[0]
                if len(idx) == 0:
                    return None
                v = complex(uv_single.data_array[idx[0], freq_channel, 0])
                return np.conj(v) if i > j else v

            v_ab = _gv(a, b)   # V(a→b)
            v_bc = _gv(b, c)   # V(b→c)
            v_ca = _gv(c, a)   # V(c→a) = conj(V(a→c))
            if v_ab is None or v_bc is None or v_ca is None:
                continue

            # Closure phase: arg(V_ab · V_bc · V_ca)
            closure_deg = float(np.degrees(np.angle(v_ab * v_bc * v_ca)))
            max_closure = max(max_closure, abs(closure_deg))

        passed = max_closure < 1e-3   # exact 0 for noiseless single-source data
        return {"max_closure_phase_deg": max_closure, "passed": passed}
