"""Hardening tests for conversion-side coordinate / UVW helpers.

Targets (all pure Python + numpy/astropy/pyuvdata; no MS, no casacore):

- ``conversion.helpers_coordinates.get_meridian_coords`` — phase-centre
  computation used by ``phase_to_meridian`` and ``ms_utils`` in production
  conversion.
- ``conversion.helpers_coordinates.angular_separation`` (and the numba kernel
  ``utils.numba_accel.angular_separation_jit``) — used by ``merge_spws`` and
  ``helpers_validation`` phase-centre coherence checks.
- ``utils.numba_accel.approx_lst_jit`` — the ``fast=True`` path of
  ``get_meridian_coords``.
- ``utils.numba_accel.compute_uvw_rotation_matrix`` /
  ``rotate_xyz_to_uvw_jit`` — exported pure-math UVW rotation.
- ``conversion.helpers_coordinates.compute_and_set_uvw`` — the production UVW
  reconstruction (via a minimal duck-typed UVData; exercises real pyuvdata).
- ``utils.numba_accel.compute_phase_corrections_jit`` — w-offset phase
  rotation sign convention.

Correctness criteria and their analytic bases are stated in each test-class
docstring. All expected values are derived from first principles or from an
independent reference implementation (astropy); nothing here is an
unverified regression pin.

Reference epoch MJD 58300 (2018-07-01) is used so astropy's *bundled* IERS-B
table covers all test times (hermetic: IERS auto-download is disabled for
the module).
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import ITRS, SkyCoord
from astropy.coordinates import angular_separation as astropy_angular_separation
from astropy.time import Time
from astropy.utils import iers
from dsa110_continuum.conversion import helpers_coordinates
from dsa110_continuum.conversion.helpers_coordinates import (
    angular_separation,
    compute_and_set_uvw,
    get_meridian_coords,
)
from dsa110_continuum.utils.constants import DSA110_LOCATION
from dsa110_continuum.utils.numba_accel import (
    angular_separation_jit,
    approx_lst_jit,
    compute_phase_corrections_jit,
    compute_uvw_rotation_matrix,
    rotate_xyz_to_uvw_jit,
)

MJD0 = 58300.0  # 2018-07-01, inside astropy's bundled IERS-B table
PT_DEC_RAD = np.deg2rad(16.1)  # DSA-110 Dec strip
SIDEREAL_RATE_RAD_PER_HR = 2.0 * np.pi * 1.00273790935 / 24.0
C_LIGHT_M_S = 299792458.0  # SI definition, matches numba_accel.compute_phase_corrections_jit

LAT_RAD = float(DSA110_LOCATION.lat.to_value(u.rad))
LON_RAD = float(DSA110_LOCATION.lon.to_value(u.rad))
ALT_M = float(DSA110_LOCATION.height.to_value(u.m))

# Local ENU unit vectors at DSA-110 expressed in the ECEF/ITRS frame.
EAST_ECEF = np.array([-np.sin(LON_RAD), np.cos(LON_RAD), 0.0])
NORTH_ECEF = np.array(
    [-np.sin(LAT_RAD) * np.cos(LON_RAD), -np.sin(LAT_RAD) * np.sin(LON_RAD), np.cos(LAT_RAD)]
)


@pytest.fixture(scope="module", autouse=True)
def _no_iers_download():
    with iers.conf.set_temp("auto_download", False):
        yield


def _apparent_lst_rad(jd_array: np.ndarray) -> np.ndarray:
    return (
        Time(np.atleast_1d(jd_array), format="jd")
        .sidereal_time("apparent", longitude=DSA110_LOCATION.lon)
        .to_value(u.rad)
    )


class _MockUVData:
    """Minimal duck-typed UVData carrying only what compute_and_set_uvw reads."""

    def __init__(self, ant_pos, ant_1_array, ant_2_array, time_array_jd):
        self.telescope_location_lat_lon_alt = (LAT_RAD, LON_RAD, ALT_M)
        self.antenna_positions = np.asarray(ant_pos, dtype=float)
        self.antenna_numbers = np.arange(len(ant_pos))
        self.ant_1_array = np.asarray(ant_1_array)
        self.ant_2_array = np.asarray(ant_2_array)
        self.time_array = np.asarray(time_array_jd, dtype=float)
        self.lst_array = _apparent_lst_rad(self.time_array)
        self.uvw_array = np.zeros((self.time_array.size, 3))


def _icrs_sky_basis_in_itrs(ra_rad: float, dec_rad: float, t: Time):
    """Independent astropy construction of the (east, north, source) UVW basis.

    The source unit vector is the astropy ICRS->ITRS transform; the on-sky
    east/north unit vectors (ICRS-north convention, matching pyuvdata's
    frame_pa) are central finite differences of that transform in RA/Dec.
    """
    eps = 1e-6

    def itrs_unit(ra, dec):
        sc = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs").transform_to(
            ITRS(obstime=t)
        )
        v = np.array([sc.x.value, sc.y.value, sc.z.value])
        return v / np.linalg.norm(v)

    s = itrs_unit(ra_rad, dec_rad)
    north = (itrs_unit(ra_rad, dec_rad + eps) - itrs_unit(ra_rad, dec_rad - eps)) / (2 * eps)
    east = (
        itrs_unit(ra_rad + eps / np.cos(dec_rad), dec_rad)
        - itrs_unit(ra_rad - eps / np.cos(dec_rad), dec_rad)
    ) / (2 * eps)
    return east, north, s


@pytest.fixture(scope="module")
def uvw_case():
    """One compute_and_set_uvw run on a random 5-antenna array, 2 times."""
    rng = np.random.default_rng(7)
    ant_pos = rng.uniform(-1500.0, 1500.0, (5, 3))
    ant_pos -= ant_pos.mean(axis=0)
    a1 = np.array([0, 0, 0, 0, 1, 1, 2, 3])
    a2 = np.array([1, 2, 3, 4, 2, 4, 3, 4])
    jd = Time(MJD0 + np.arange(2) * 12.0 / 86400.0, format="mjd").jd
    uv = _MockUVData(ant_pos, np.tile(a1, 2), np.tile(a2, 2), np.repeat(jd, a1.size))
    compute_and_set_uvw(uv, PT_DEC_RAD * u.rad)
    return uv, ant_pos, jd


class TestGetMeridianCoords:
    """Phase-centre computation for the drift-scan meridian.

    Criteria
    --------
    1. Round-trip inverse: the returned ICRS coordinate, transformed back to
       the HADec frame at the same time/location, must recover HA = 0 and
       Dec = pt_dec. Analytic basis: frame round-trips are exact inverses;
       ERFA forward/backward chains agree to sub-microarcsecond (measured
       residual <= 4e-13 rad). Tolerance 1e-10 rad (~20 uas) gives >100x
       headroom while staying ~6 orders below the pipeline's arcsecond-level
       astrometric budget.
    2. Sidereal rate: meridian ICRS RA must advance by ~1.00273791 * 15 deg/hr.
       The apparent<->ICRS RA offset itself varies across the 15 deg hourly
       sweep by up to ~30 arcsec (annual-precession n*dt*d(sin RA)*tan Dec at
       ~2 decades from J2000, Dec 16 deg), so the bound is 60 arcsec/hr
       (2.9e-4 rad). A solar-rate bug differs by 147 arcsec/hr and a sign bug
       by ~2 rad, both far outside the bound.
    3. Unit handling: a plain-float pt_dec (multiprocessing path) must give
       the identical result as the Quantity path (pure dispatch, no numerics).
    """

    @pytest.mark.parametrize("mjd", [MJD0, MJD0 + 0.3, MJD0 + 123.456])
    def test_round_trip_back_to_hadec_recovers_ha_zero_and_pt_dec(self, mjd):
        ra, dec = get_meridian_coords(PT_DEC_RAD * u.rad, mjd)
        hadec_frame = SkyCoord(
            ha=0 * u.hourangle,
            dec=PT_DEC_RAD * u.rad,
            frame="hadec",
            obstime=Time(mjd, format="mjd"),
            location=DSA110_LOCATION,
        ).frame
        back = SkyCoord(ra=ra, dec=dec, frame="icrs").transform_to(hadec_frame)
        ha_rad = (back.ha.to_value(u.rad) + np.pi) % (2 * np.pi) - np.pi
        np.testing.assert_allclose(ha_rad, 0.0, atol=1e-10)
        np.testing.assert_allclose(back.dec.to_value(u.rad), PT_DEC_RAD, rtol=0, atol=1e-10)

    def test_meridian_ra_advances_at_sidereal_rate(self):
        ra1, _ = get_meridian_coords(PT_DEC_RAD * u.rad, MJD0)
        ra2, _ = get_meridian_coords(PT_DEC_RAD * u.rad, MJD0 + 1.0 / 24.0)
        dra = (ra2 - ra1).to_value(u.rad) % (2 * np.pi)
        np.testing.assert_allclose(dra, SIDEREAL_RATE_RAD_PER_HR, rtol=0, atol=2.9e-4)

    def test_float_pt_dec_matches_quantity_pt_dec(self):
        ra_q, dec_q = get_meridian_coords(PT_DEC_RAD * u.rad, MJD0)
        ra_f, dec_f = get_meridian_coords(PT_DEC_RAD, MJD0)
        np.testing.assert_allclose(ra_f.to_value(u.rad), ra_q.to_value(u.rad), rtol=0, atol=1e-15)
        np.testing.assert_allclose(
            dec_f.to_value(u.rad), dec_q.to_value(u.rad), rtol=0, atol=1e-15
        )


class TestAngularSeparation:
    """Great-circle separation vs the independent astropy reference.

    astropy.coordinates.angular_separation implements the Vincenty formula,
    which is well-conditioned at all separations, so it is a valid reference
    everywhere. The kernel under test uses the arccos (spherical law of
    cosines) formula, whose conditioning floor near s=0 (and s=pi) is
    sqrt(2*eps) ~ 2.1e-8 rad in float64.

    Criteria
    --------
    1. Random sphere pairs (separations >~ 1e-3 rad): agreement within
       atol=1e-11 rad. Basis: arccos error ~ eps/sin(s) <= ~2e-13 for
       s >= 1e-3; 50x headroom for fastmath reassociation (measured 2.7e-14).
    2. Small separations (1e-9..1e-4 rad): agreement within atol=2.5e-8 rad
       — the arccos conditioning floor above, NOT a tunable bound (measured
       2.06e-8). This documents the kernel's precision floor.
    3. Invariants: symmetry, zero self-separation, range [0, pi].
    4. The pure-numpy dispatcher fallback must match the numba path within
       the same conditioning floor.
    """

    @staticmethod
    def _random_pairs(n=5000, seed=42):
        rng = np.random.default_rng(seed)
        ra_a = rng.uniform(0, 2 * np.pi, n)
        dec_a = np.arcsin(rng.uniform(-1, 1, n))
        ra_b = rng.uniform(0, 2 * np.pi, n)
        dec_b = np.arcsin(rng.uniform(-1, 1, n))
        return ra_a, dec_a, ra_b, dec_b

    def test_matches_astropy_on_random_sphere_pairs(self):
        ra_a, dec_a, ra_b, dec_b = self._random_pairs()
        ref = np.asarray(astropy_angular_separation(ra_a, dec_a, ra_b, dec_b))
        got = angular_separation_jit(ra_a, dec_a, ra_b, dec_b)
        keep = ref > 1e-3  # arccos conditioning valid regime; small seps tested separately
        np.testing.assert_allclose(got[keep], ref[keep], rtol=0, atol=1e-11)

    def test_small_separation_conditioning_floor(self):
        rng = np.random.default_rng(11)
        n = 5000
        ra_a = rng.uniform(0, 2 * np.pi, n)
        dec_a = np.arcsin(rng.uniform(-0.98, 0.98, n))
        offset = 10.0 ** rng.uniform(-9, -4, n)
        pa = rng.uniform(0, 2 * np.pi, n)
        ra_b = ra_a + offset * np.sin(pa) / np.cos(dec_a)
        dec_b = dec_a + offset * np.cos(pa)
        ref = np.asarray(astropy_angular_separation(ra_a, dec_a, ra_b, dec_b))
        got = angular_separation_jit(ra_a, dec_a, ra_b, dec_b)
        np.testing.assert_allclose(got, ref, rtol=0, atol=2.5e-8)

    def test_symmetry_identity_and_range(self):
        ra_a, dec_a, ra_b, dec_b = self._random_pairs(n=2000, seed=3)
        fwd = angular_separation_jit(ra_a, dec_a, ra_b, dec_b)
        rev = angular_separation_jit(ra_b, dec_b, ra_a, dec_a)
        np.testing.assert_allclose(fwd, rev, rtol=0, atol=1e-14)
        # Self-separation sits exactly at the arccos conditioning floor
        # sqrt(2*eps) ~ 2.1e-8 (rounding in sin^2+cos^2 makes cos_sep < 1).
        self_sep = angular_separation_jit(ra_a, dec_a, ra_a, dec_a)
        np.testing.assert_allclose(self_sep, 0.0, rtol=0, atol=2.5e-8)
        assert np.all(fwd >= 0.0) and np.all(fwd <= np.pi + 1e-14)

    def test_numpy_fallback_matches_numba_path(self, monkeypatch):
        ra_a, dec_a, ra_b, dec_b = self._random_pairs(n=500, seed=5)
        jit_result = angular_separation(ra_a, dec_a, ra_b, dec_b)
        monkeypatch.setattr(helpers_coordinates, "_USE_NUMBA_ANGULAR_SEP", False)
        numpy_result = angular_separation(ra_a, dec_a, ra_b, dec_b)
        np.testing.assert_allclose(
            np.asarray(numpy_result, dtype=float),
            np.asarray(jit_result, dtype=float),
            rtol=0,
            atol=2.5e-8,
        )


class TestApproxLst:
    """Fast LST approximation vs astropy mean sidereal time (reference).

    Criteria
    --------
    1. Agreement with astropy ``sidereal_time('mean')`` at DSA-110 within
       atol=1e-4 rad (~20.6 arcsec). Error budget: the Meeus linear GMST
       formula takes UTC as UT1 (|UT1-UTC| <= 0.9 s = 6.6e-5 rad of rotation)
       and truncates the T^2 GMST term (< 1e-6 rad within the test epoch
       range). Measured max residual over 2017-2020: 1.8e-5 rad. This bound
       is consistent with the documented use ("phase center tracking where
       sub-arcsecond precision is not required").
    2. Output normalized to [0, 2*pi).

    Differences are compared on the circle (via angle of the complex ratio)
    to avoid 2*pi wrap artifacts.
    """

    def test_matches_astropy_mean_sidereal_time(self):
        rng = np.random.default_rng(19)
        mjds = np.sort(rng.uniform(58000.0, 58800.0, 40))
        got = approx_lst_jit(mjds, LON_RAD)
        ref = (
            Time(mjds, format="mjd")
            .sidereal_time("mean", longitude=DSA110_LOCATION.lon)
            .to_value(u.rad)
        )
        circular_diff = np.angle(np.exp(1j * (got - ref)))
        np.testing.assert_allclose(circular_diff, 0.0, rtol=0, atol=1e-4)

    def test_output_in_principal_range(self):
        mjds = np.linspace(58000.0, 58800.0, 200)
        got = approx_lst_jit(mjds, LON_RAD)
        assert np.all(got >= 0.0) and np.all(got < 2 * np.pi)


class TestUvwRotation:
    """Pure-math XYZ->UVW rotation (numba_accel).

    Convention under test (Thompson-Moran-Swenson): XYZ equatorial with X at
    (HA=0, Dec=0), Y East (HA=-6h), Z at the NCP; u East on sky, v North on
    sky, w toward the source at (HA, Dec).

    Criteria
    --------
    1. Rotation-matrix orthonormality and det=+1 for random (HA, Dec):
       atol=1e-13 (pure trig products accumulate ~4 eps ~ 1e-15; measured
       3.3e-16).
    2. The batched JIT kernel must equal the matrix product R @ xyz
       (atol=1e-10 m on <=1.5 km baselines; fastmath reassociation ~10 eps *
       scale = 3e-13 m measured 4.5e-13) and preserve baseline norms
       (rtol=1e-12; rotations are isometries).
    3. Analytic transit cases (exact trig values, atol=1e-12*|b|):
       - East-West baseline (0, L, 0) at HA=0 maps to (L, 0, 0); in
         particular w=0 for ANY declination.
       - A baseline in the meridian plane (x, 0, z) has u=0 at HA=0.
       - The NCP direction (0, 0, 1) maps to (0, cos Dec, sin Dec).
    4. Zenith case: a horizontal baseline at latitude phi observing the
       zenith (HA=0, Dec=phi) has w=0 analytically (the baseline is
       perpendicular to the vertical): atol=1e-12*|b|.
    5. Independent basis cross-check: u = b . east, v = b . north, w = b . s
       where east/north are CENTRAL FINITE DIFFERENCES of the source unit
       vector s(HA, Dec) — independent of the coded trig identities, so a
       sign/row swap cannot cancel. atol=1e-6 m: central-difference
       truncation is h^2*|b|/6 ~ 2.5e-8 m at h=1e-5, |b|<=1.5 km (measured
       ~9e-9 m); 40x headroom.
    6. Linearity/antisymmetry: rotate(-b) = -rotate(b), atol=1e-12*|b|.
    """

    def test_rotation_matrix_is_orthonormal_with_unit_determinant(self):
        rng = np.random.default_rng(23)
        for _ in range(100):
            ha = rng.uniform(-np.pi, np.pi)
            dec = rng.uniform(-np.pi / 2, np.pi / 2)
            r_mat = compute_uvw_rotation_matrix(ha, dec)
            np.testing.assert_allclose(r_mat @ r_mat.T, np.eye(3), rtol=0, atol=1e-13)
            np.testing.assert_allclose(np.linalg.det(r_mat), 1.0, rtol=0, atol=1e-13)

    def test_jit_rotation_matches_matrix_and_preserves_norm(self):
        rng = np.random.default_rng(29)
        xyz = rng.uniform(-1500.0, 1500.0, (200, 3))
        ha_arr = rng.uniform(-np.pi, np.pi, 200)
        uvw = rotate_xyz_to_uvw_jit(xyz, ha_arr, PT_DEC_RAD)
        expected = np.einsum(
            "nij,nj->ni",
            np.stack([compute_uvw_rotation_matrix(h, PT_DEC_RAD) for h in ha_arr]),
            xyz,
        )
        np.testing.assert_allclose(uvw, expected, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            np.linalg.norm(uvw, axis=1), np.linalg.norm(xyz, axis=1), rtol=1e-12
        )

    @pytest.mark.parametrize("dec", [np.deg2rad(-20.0), PT_DEC_RAD, np.deg2rad(55.0)])
    def test_transit_analytic_cases(self, dec):
        length = 1000.0
        xyz = np.array(
            [
                [0.0, length, 0.0],  # East-West
                [300.0, 0.0, 700.0],  # in the meridian plane
                [0.0, 0.0, 1.0],  # NCP direction
            ]
        )
        uvw = rotate_xyz_to_uvw_jit(xyz, np.zeros(3), dec)
        np.testing.assert_allclose(
            uvw[0], [length, 0.0, 0.0], rtol=0, atol=1e-12 * length
        )
        np.testing.assert_allclose(uvw[1, 0], 0.0, rtol=0, atol=1e-12 * length)
        np.testing.assert_allclose(
            uvw[2], [0.0, np.cos(dec), np.sin(dec)], rtol=0, atol=1e-12
        )

    def test_zenith_horizontal_baseline_w_zero(self):
        length = 1000.0
        # ENU (e, n, 0) horizontal baseline at latitude phi in XYZ coords:
        # X = -n sin(phi), Y = e, Z = n cos(phi)
        e_comp, n_comp = 600.0, 800.0
        xyz = np.array(
            [[-n_comp * np.sin(LAT_RAD), e_comp, n_comp * np.cos(LAT_RAD)]]
        )
        uvw = rotate_xyz_to_uvw_jit(xyz, np.zeros(1), LAT_RAD)  # zenith: Dec = latitude
        np.testing.assert_allclose(uvw[0, 2], 0.0, rtol=0, atol=1e-12 * length)

    def test_uvw_basis_matches_finite_difference_of_source_direction(self):
        def s_hat(ha, dec):
            return np.array(
                [np.cos(dec) * np.cos(ha), -np.cos(dec) * np.sin(ha), np.sin(dec)]
            )

        rng = np.random.default_rng(31)
        h_step = 1e-5
        for dec in [np.deg2rad(-30.0), PT_DEC_RAD, np.deg2rad(60.0)]:
            xyz = rng.uniform(-1500.0, 1500.0, (20, 3))
            ha_arr = rng.uniform(-np.pi, np.pi, 20)
            uvw = rotate_xyz_to_uvw_jit(xyz, ha_arr, dec)
            for i in range(20):
                ha, b = ha_arr[i], xyz[i]
                s0 = s_hat(ha, dec)
                east = (s_hat(ha - h_step, dec) - s_hat(ha + h_step, dec)) / (
                    2 * h_step * np.cos(dec)
                )
                north = (s_hat(ha, dec + h_step) - s_hat(ha, dec - h_step)) / (2 * h_step)
                np.testing.assert_allclose(
                    uvw[i], [b @ east, b @ north, b @ s0], rtol=0, atol=1e-6
                )

    def test_baseline_antisymmetry(self):
        rng = np.random.default_rng(37)
        xyz = rng.uniform(-1500.0, 1500.0, (50, 3))
        ha_arr = rng.uniform(-np.pi, np.pi, 50)
        fwd = rotate_xyz_to_uvw_jit(xyz, ha_arr, PT_DEC_RAD)
        rev = rotate_xyz_to_uvw_jit(-xyz, ha_arr, PT_DEC_RAD)
        np.testing.assert_allclose(rev, -fwd, rtol=0, atol=1e-12 * 1500.0)


class TestComputeAndSetUvw:
    """Production UVW reconstruction against independent astropy geometry.

    The mock UVData carries ECEF-relative antenna positions at the real
    DSA-110 site with astropy apparent-LST time arrays; compute_and_set_uvw
    then runs the real pyuvdata phasing chain.

    Criteria
    --------
    1. Norm preservation: |uvw| = |antenna_j - antenna_i| for every
       baseline-time (rotations are isometries; rtol=1e-12, measured 8e-17
       relative).
    2. Antenna-pair antisymmetry: swapping ant_1_array/ant_2_array negates
       every UVW vector (atol=1e-9 m; measured exactly 0).
    3. w = b . s_hat where s_hat is the ASTROPY ICRS->ITRS unit vector of the
       phase centre: atol = |b|_max * 5e-6 (~1 arcsec). Basis: w is immune to
       the sky-frame position angle, so the only cross-implementation
       difference is apparent-place consistency between astropy and
       pyuvdata/erfa (sub-arcsec; measured 0.19 arcsec-equivalent).
    4. Full (u, v, w) vs the astropy finite-difference ICRS sky basis:
       atol = |b|_max * 1e-4 (~20.6 arcsec). Basis: the two implementations'
       sky-north (frame position angle) conventions agree only to ~13 arcsec
       (measured; differential-aberration-level definition differences).
       This still catches a dropped frame_pa (117 arcsec at the 2018 test
       epoch, growing ~6.5 arcsec/yr from J2000) and any sign/axis bug
       (O(|b|)).
    5. Transit geometry: for a locally East-West baseline the phase centre is
       on the meridian, so w = 0 analytically at ANY declination (w is
       frame-PA-immune; the residual measures apparent-HA consistency,
       measured ~3e-13 m). atol = |b| * 1e-6 (~0.2 arcsec) covers
       cross-library apparent-place differences. u must equal +|b| to within
       |b| * 1e-3 (frame-PA rotation reduces it only at second order).
    6. Fail-loud: a UVData object with no telescope location raises
       ValueError (guard added during hardening).
    """

    def test_norm_preserved(self, uvw_case):
        uv, ant_pos, _ = uvw_case
        baselines = ant_pos[uv.ant_2_array] - ant_pos[uv.ant_1_array]
        np.testing.assert_allclose(
            np.linalg.norm(uv.uvw_array, axis=1),
            np.linalg.norm(baselines, axis=1),
            rtol=1e-12,
        )

    def test_antisymmetry_under_antenna_swap(self, uvw_case):
        uv, ant_pos, _ = uvw_case
        swapped = _MockUVData(ant_pos, uv.ant_2_array, uv.ant_1_array, uv.time_array)
        compute_and_set_uvw(swapped, PT_DEC_RAD * u.rad)
        np.testing.assert_allclose(swapped.uvw_array, -uv.uvw_array, rtol=0, atol=1e-9)

    def test_w_matches_astropy_projection_onto_phase_centre(self, uvw_case):
        uv, ant_pos, jd = uvw_case
        baselines = ant_pos[uv.ant_2_array] - ant_pos[uv.ant_1_array]
        bl_max = np.linalg.norm(baselines, axis=1).max()
        for jdi in jd:
            t = Time(jdi, format="jd")
            ra, dec = get_meridian_coords(PT_DEC_RAD * u.rad, float(t.mjd))
            _, _, s = _icrs_sky_basis_in_itrs(ra.to_value(u.rad), dec.to_value(u.rad), t)
            rows = uv.time_array == jdi
            np.testing.assert_allclose(
                uv.uvw_array[rows, 2], baselines[rows] @ s, rtol=0, atol=bl_max * 5e-6
            )

    def test_uvw_matches_astropy_icrs_sky_basis(self, uvw_case):
        uv, ant_pos, jd = uvw_case
        baselines = ant_pos[uv.ant_2_array] - ant_pos[uv.ant_1_array]
        bl_max = np.linalg.norm(baselines, axis=1).max()
        for jdi in jd:
            t = Time(jdi, format="jd")
            ra, dec = get_meridian_coords(PT_DEC_RAD * u.rad, float(t.mjd))
            east, north, s = _icrs_sky_basis_in_itrs(
                ra.to_value(u.rad), dec.to_value(u.rad), t
            )
            rows = uv.time_array == jdi
            b = baselines[rows]
            expected = np.stack([b @ east, b @ north, b @ s], axis=1)
            np.testing.assert_allclose(
                uv.uvw_array[rows], expected, rtol=0, atol=bl_max * 1e-4
            )

    @pytest.mark.parametrize("dec_deg", [-20.0, 16.1, 55.0])
    def test_east_west_baseline_at_transit_has_zero_w(self, dec_deg):
        length = 1000.0
        ant_pos = np.array([[0.0, 0.0, 0.0], length * EAST_ECEF])
        jd = Time(MJD0, format="mjd").jd
        uv = _MockUVData(ant_pos, [0], [1], [jd])
        compute_and_set_uvw(uv, np.deg2rad(dec_deg) * u.rad)
        np.testing.assert_allclose(uv.uvw_array[0, 2], 0.0, rtol=0, atol=length * 1e-6)
        np.testing.assert_allclose(uv.uvw_array[0, 0], length, rtol=0, atol=length * 1e-3)

    def test_missing_telescope_location_raises(self):
        class _Bare:
            pass

        bare = _Bare()
        bare.time_array = np.array([Time(MJD0, format="mjd").jd])
        with pytest.raises(ValueError, match="telescope location"):
            compute_and_set_uvw(bare, PT_DEC_RAD * u.rad)

    def test_apparent_coord_failure_raises_instead_of_silent_frame_pa_zero(self, monkeypatch):
        """A failed apparent-coordinate transform must abort conversion: the old
        frame_pa=0 fallback silently rotated UVWs by the ICRS<->apparent sky
        rotation (~2 arcmin at 2026 epochs), which is wrong science with no
        exception."""
        pu_calc_uvw, _, calc_frame_pos_angle = (
            helpers_coordinates._load_pyuvdata_phasing_helpers()
        )

        def broken_calc_app_coords(*args, **kwargs):
            raise ValueError("synthetic apparent-coordinate failure")

        monkeypatch.setattr(
            helpers_coordinates,
            "_load_pyuvdata_phasing_helpers",
            lambda: (pu_calc_uvw, broken_calc_app_coords, calc_frame_pos_angle),
        )
        ant_pos = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        uv = _MockUVData(ant_pos, [0], [1], [Time(MJD0, format="mjd").jd])
        with pytest.raises(ValueError, match="apparent-coordinate"):
            compute_and_set_uvw(uv, PT_DEC_RAD * u.rad)


class TestPhaseCorrections:
    """w-offset phase rotation kernel.

    Criteria
    --------
    1. Unit modulus: |exp(i*phi)| = 1 exactly (atol=1e-14; cos^2+sin^2
       identity holds to ~eps, measured 2.2e-16).
    2. Reference cross-check: matches an independent numpy
       exp(-2j*pi*w*f/c) with the SI speed of light — verifies both the
       PHASE SIGN convention (negative for positive w offset) and the
       magnitude. atol=1e-9: sin/cos vs exp argument-reduction differences
       are ~eps*|phi| with |phi| up to 2*pi*2000m*1.53GHz/c ~ 6.4e4 rad,
       giving ~1.4e-11 (measured); ~70x headroom.
    3. Zero offset maps to exactly 1+0j; conjugate symmetry
       corr(-w) = conj(corr(w)) (atol=1e-14, pure parity of cos/sin).
    """

    FREQS_HZ = np.linspace(1.28e9, 1.53e9, 64)  # DSA-110 band

    def test_unit_modulus_and_matches_numpy_reference(self):
        rng = np.random.default_rng(41)
        w_off = rng.uniform(-2000.0, 2000.0, 50)
        corr = compute_phase_corrections_jit(np.zeros((50, 3)), self.FREQS_HZ, w_off)
        np.testing.assert_allclose(np.abs(corr), 1.0, rtol=0, atol=1e-14)
        reference = np.exp(-2j * np.pi * np.outer(w_off, self.FREQS_HZ) / C_LIGHT_M_S)
        np.testing.assert_allclose(corr, reference, rtol=0, atol=1e-9)

    def test_zero_offset_is_unity_and_conjugate_symmetry(self):
        w_off = np.array([0.0, 123.456])
        corr = compute_phase_corrections_jit(np.zeros((2, 3)), self.FREQS_HZ, w_off)
        np.testing.assert_allclose(corr[0], 1.0 + 0.0j, rtol=0, atol=1e-14)
        corr_neg = compute_phase_corrections_jit(np.zeros((2, 3)), self.FREQS_HZ, -w_off)
        np.testing.assert_allclose(corr_neg, np.conj(corr), rtol=0, atol=1e-14)
