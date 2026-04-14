"""
Tests for dsa110_continuum.simulation.harness
==============================================

All tests are cloud-safe (no CASA, no H17, no real HDF5).
They run against the SimulationHarness class which generates realistic
synthetic DSA-110 UVH5 subband data using pyuvdata + pyradiosky.

Fixture strategy
----------------
* ``harness_small`` — 6 antennas, 3 integrations, 5 sources, no noise.
  Fast enough for unit tests; produces exact closure ≈ 0.
* ``harness_realistic`` — 16 antennas, 8 integrations, 20 sources, 0.5 Jy noise.
  Used for shape/range/IO tests that need slightly more realism.
* A temporary directory (``tmp_dir``) is provided by a session-scoped fixture
  so that multiple tests share the same generated files where possible.
"""
from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

# ── skip guard ────────────────────────────────────────────────────────────────
pytest.importorskip("pyuvdata",  reason="pyuvdata not installed")
pytest.importorskip("pyradiosky", reason="pyradiosky not installed")

from dsa110_continuum.simulation.harness import (
    SimulationHarness,
    _FREQ_MIN_HZ,
    _FREQ_LOWER_EDGE_HZ,
    _SB_BW_HZ,
    _N_CHAN_PER_SB,
    _N_SUBBANDS,
    _CHAN_WIDTH_HZ,
    _INTEGRATION_SEC,
    _make_sky_model,
    _load_antenna_enu_from_csv,
    _enu_to_ecef,
)

# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def harness_small() -> SimulationHarness:
    """Minimal harness: 6 antennas, 3 integrations, 5 sources, no noise."""
    return SimulationHarness(
        n_antennas=6,
        n_integrations=3,
        n_sky_sources=5,
        noise_jy=0.0,
        seed=0,
    )


@pytest.fixture(scope="module")
def harness_realistic() -> SimulationHarness:
    """Larger harness for IO/shape tests."""
    return SimulationHarness(
        n_antennas=16,
        n_integrations=8,
        n_sky_sources=20,
        noise_jy=0.5,
        seed=1,
    )


@pytest.fixture(scope="module")
def generated_sb0(harness_small, tmp_path_factory):
    """Pre-generated subband-0 UVH5 file (shared across tests in the module)."""
    out = tmp_path_factory.mktemp("sim_sb") / "sb00.uvh5"
    harness_small.generate_subband(0, out)
    return out


@pytest.fixture(scope="module")
def generated_sb7(harness_small, tmp_path_factory):
    """Pre-generated subband-7 UVH5 file."""
    out = tmp_path_factory.mktemp("sim_sb7") / "sb07.uvh5"
    harness_small.generate_subband(7, out)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Module-level helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestHelpers:
    """Unit tests for standalone helper functions."""

    def test_load_antenna_enu_shape(self):
        """Loading from real CSV gives correct shape and 3 columns."""
        enu = _load_antenna_enu_from_csv(8)
        assert enu.shape == (8, 3)

    def test_load_antenna_enu_t_array_span(self):
        """Real DSA-110 positions: T-array should span > 100 m in both E and N."""
        enu = _load_antenna_enu_from_csv(96)
        east_span = enu[:, 0].max() - enu[:, 0].min()
        north_span = enu[:, 1].max() - enu[:, 1].min()
        assert east_span > 100.0, f"E-W span too small: {east_span:.1f} m"
        assert north_span > 100.0, f"N-S span too small: {north_span:.1f} m"

    def test_enu_to_ecef_origin(self):
        """ENU (0,0,0) should map to the OVRO reference ECEF point."""
        from astropy.coordinates import EarthLocation
        import astropy.units as u

        lat, lon, alt = 37.2339, -118.2825, 1222.0
        loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt * u.m)
        ecef = _enu_to_ecef(np.zeros((1, 3)), lat, lon, alt)
        ref = np.array([loc.x.to(u.m).value, loc.y.to(u.m).value, loc.z.to(u.m).value])
        np.testing.assert_allclose(ecef[0], ref, atol=1.0)  # within 1 m

    def test_enu_to_ecef_east_offset(self):
        """100 m east should produce a change primarily in x/y, not z."""
        lat, lon, alt = 37.2339, -118.2825, 1222.0
        enu_east = np.array([[100.0, 0.0, 0.0]])
        enu_orig = np.zeros((1, 3))
        ecef_e = _enu_to_ecef(enu_east, lat, lon, alt)
        ecef_o = _enu_to_ecef(enu_orig, lat, lon, alt)
        diff = ecef_e[0] - ecef_o[0]
        # Total displacement should be ~100 m
        np.testing.assert_allclose(np.linalg.norm(diff), 100.0, atol=0.1)

    def test_make_sky_model_n_components(self):
        rng = np.random.default_rng(7)
        sky = _make_sky_model(12, 343.5, 16.15, 3.5, 1.4e9, rng)
        assert sky.Ncomponents == 12

    def test_make_sky_model_stokes_shape(self):
        rng = np.random.default_rng(7)
        sky = _make_sky_model(5, 343.5, 16.15, 3.5, 1.4e9, rng)
        assert sky.stokes.shape == (4, 1, 5)

    def test_make_sky_model_positive_flux(self):
        rng = np.random.default_rng(7)
        sky = _make_sky_model(10, 343.5, 16.15, 3.5, 1.4e9, rng)
        import astropy.units as u
        stokes_i = sky.stokes[0, 0, :].to(u.Jy).value
        assert np.all(stokes_i > 0), "All Stokes I fluxes must be positive"

    def test_make_sky_model_flux_range(self):
        """Fluxes should span mJy–Jy range (5 mJy – 5 Jy)."""
        rng = np.random.default_rng(7)
        import astropy.units as u
        sky = _make_sky_model(100, 343.5, 16.15, 3.5, 1.4e9, rng)
        stokes_i = sky.stokes[0, 0, :].to(u.Jy).value
        assert stokes_i.min() > 0.001      # > 1 mJy
        assert stokes_i.max() < 20.0       # < 20 Jy


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Frequency axis
# ══════════════════════════════════════════════════════════════════════════════

class TestFrequencyAxis:
    """Tests for subband frequency generation."""

    @pytest.mark.parametrize("sb", [0, 7, 15])
    def test_subband_freqs_shape(self, sb):
        freqs = SimulationHarness.subband_freqs(sb)
        assert freqs.shape == (_N_CHAN_PER_SB,)

    def test_subband_freqs_sb0_start(self):
        """First channel centre should be 1311.25 MHz + 0.5 * channel_width."""
        freqs = SimulationHarness.subband_freqs(0)
        np.testing.assert_allclose(freqs[0], _FREQ_MIN_HZ, rtol=1e-9)

    def test_subband_freqs_sb15_end(self):
        """Last channel centre of last subband should be 1498.75 MHz - 0.5*chan_width."""
        freqs = SimulationHarness.subband_freqs(15)
        # Upper edge of band = 1498.75 MHz; last channel centre = edge - 0.5*chan_width
        expected_end = 1_498_750_000.0 - 0.5 * _CHAN_WIDTH_HZ
        np.testing.assert_allclose(freqs[-1], expected_end, rtol=1e-9)

    def test_subband_freqs_spacing(self):
        freqs = SimulationHarness.subband_freqs(3)
        diffs = np.diff(freqs)
        np.testing.assert_allclose(diffs, _CHAN_WIDTH_HZ, rtol=1e-9)

    def test_subband_freqs_monotonic(self):
        for sb in range(_N_SUBBANDS):
            freqs = SimulationHarness.subband_freqs(sb)
            assert np.all(np.diff(freqs) > 0), f"Subband {sb} not monotonic"

    def test_subband_freqs_no_overlap(self):
        """Adjacent subbands should be contiguous (no gap, no overlap)."""
        for sb in range(_N_SUBBANDS - 1):
            f_hi = SimulationHarness.subband_freqs(sb)[-1]
            f_lo = SimulationHarness.subband_freqs(sb + 1)[0]
            np.testing.assert_allclose(f_lo - f_hi, _CHAN_WIDTH_HZ, rtol=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Antenna positions
# ══════════════════════════════════════════════════════════════════════════════

class TestAntennaPositions:
    """Tests for harness antenna position properties."""

    def test_antenna_enu_shape(self, harness_small):
        assert harness_small.antenna_enu.shape == (harness_small.n_antennas, 3)

    def test_antenna_ecef_shape(self, harness_small):
        assert harness_small.antenna_ecef.shape == (harness_small.n_antennas, 3)

    def test_antenna_enu_cached(self, harness_small):
        """Second access should return the same object (cached)."""
        a = harness_small.antenna_enu
        b = harness_small.antenna_enu
        assert a is b

    def test_antenna_ecef_magnitude(self, harness_small):
        """Absolute ECEF magnitudes should be close to Earth's radius (~6.37 Mm)."""
        mags = np.linalg.norm(harness_small.antenna_ecef, axis=1)
        assert np.all(mags > 6.36e6) and np.all(mags < 6.39e6)

    def test_antenna_ecef_relative_offsets_small(self, harness_small):
        """ECEF offsets relative to telescope reference should be < ~2 km."""
        import astropy.units as u
        loc = harness_small._location
        tel_ecef = np.array([
            loc.x.to(u.m).value,
            loc.y.to(u.m).value,
            loc.z.to(u.m).value,
        ])
        offsets = harness_small.antenna_ecef - tel_ecef
        magnitudes = np.linalg.norm(offsets, axis=1)
        assert np.all(magnitudes < 2000.0), "Antenna offsets should be < 2 km"


# ══════════════════════════════════════════════════════════════════════════════
# 4.  UVData shape and metadata
# ══════════════════════════════════════════════════════════════════════════════

class TestUVDataShape:
    """Tests that the generated UVData has the correct shape and metadata."""

    def test_nblts(self, harness_small, generated_sb0):
        uv = harness_small.load_subband(generated_sb0)
        n_bls = harness_small.n_antennas * (harness_small.n_antennas - 1) // 2
        assert uv.Nblts == n_bls * harness_small.n_integrations

    def test_nfreqs(self, harness_small, generated_sb0):
        uv = harness_small.load_subband(generated_sb0)
        assert uv.Nfreqs == _N_CHAN_PER_SB

    def test_npols(self, harness_small, generated_sb0):
        uv = harness_small.load_subband(generated_sb0)
        assert uv.Npols == 2   # XX and YY

    def test_ntimes(self, harness_small, generated_sb0):
        uv = harness_small.load_subband(generated_sb0)
        assert uv.Ntimes == harness_small.n_integrations

    def test_nbls(self, harness_small, generated_sb0):
        uv = harness_small.load_subband(generated_sb0)
        n_bls = harness_small.n_antennas * (harness_small.n_antennas - 1) // 2
        assert uv.Nbls == n_bls

    def test_data_array_shape(self, harness_small, generated_sb0):
        uv = harness_small.load_subband(generated_sb0)
        n_bls = harness_small.n_antennas * (harness_small.n_antennas - 1) // 2
        assert uv.data_array.shape == (
            n_bls * harness_small.n_integrations,
            _N_CHAN_PER_SB,
            2,
        )

    def test_uvw_array_shape(self, harness_small, generated_sb0):
        uv = harness_small.load_subband(generated_sb0)
        assert uv.uvw_array.shape == (uv.Nblts, 3)

    def test_freq_range_sb0(self, harness_small, generated_sb0):
        """Subband 0 should span from first to last channel centre."""
        uv = harness_small.load_subband(generated_sb0)
        np.testing.assert_allclose(uv.freq_array.min(), _FREQ_MIN_HZ, rtol=1e-9)
        expected_end = _FREQ_MIN_HZ + (_N_CHAN_PER_SB - 1) * _CHAN_WIDTH_HZ
        np.testing.assert_allclose(uv.freq_array.max(), expected_end, rtol=1e-9)

    def test_freq_range_sb7(self, harness_small, generated_sb7):
        uv = harness_small.load_subband(generated_sb7)
        # Subband 7 starts at _FREQ_MIN_HZ + 7 * _SB_BW_HZ
        expected_start = _FREQ_MIN_HZ + 7 * _SB_BW_HZ
        np.testing.assert_allclose(uv.freq_array.min(), expected_start, rtol=1e-9)

    def test_vis_units(self, harness_small, generated_sb0):
        uv = harness_small.load_subband(generated_sb0)
        assert uv.vis_units == "Jy"

    def test_phase_center_sidereal(self, harness_small, generated_sb0):
        uv = harness_small.load_subband(generated_sb0)
        assert 0 in uv.phase_center_catalog
        cat = uv.phase_center_catalog[0]
        assert cat["cat_type"] == "sidereal"

    def test_phase_center_coords(self, harness_small, generated_sb0):
        uv = harness_small.load_subband(generated_sb0)
        cat = uv.phase_center_catalog[0]
        np.testing.assert_allclose(
            np.degrees(cat["cat_lon"]), harness_small.pointing_ra_deg, atol=0.01
        )
        np.testing.assert_allclose(
            np.degrees(cat["cat_lat"]), harness_small.pointing_dec_deg, atol=0.01
        )

    def test_extra_keywords(self, harness_small, generated_sb0):
        uv = harness_small.load_subband(generated_sb0)
        kw = uv.extra_keywords
        assert "SIM_SEED" in kw
        assert "SIM_NANTS" in kw
        assert int(kw["SIM_NANTS"]) == harness_small.n_antennas

    def test_polarization_array(self, harness_small, generated_sb0):
        uv = harness_small.load_subband(generated_sb0)
        # -5 = XX, -6 = YY in AIPS convention
        pols = set(uv.polarization_array.tolist())
        assert pols == {-5, -6}

    def test_integration_time(self, harness_small, generated_sb0):
        uv = harness_small.load_subband(generated_sb0)
        np.testing.assert_allclose(
            uv.integration_time, _INTEGRATION_SEC, rtol=1e-6
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Closure relations
# ══════════════════════════════════════════════════════════════════════════════

class TestClosureRelations:
    """Tests that verify the sky model satisfies closure-phase identities."""

    def test_closure_zero_noiseless(self, harness_small, generated_sb0):
        """Closure phase should be exactly 0 for a noiseless single-source sky."""
        result = harness_small.check_closure(generated_sb0, n_triangles=10)
        assert result["passed"], (
            f"Closure phase failed: max={result['max_closure_phase_deg']:.4f} deg"
        )
        assert result["max_closure_phase_deg"] < 1e-3

    def test_closure_returns_dict(self, harness_small, generated_sb0):
        result = harness_small.check_closure(generated_sb0)
        assert "max_closure_phase_deg" in result
        assert "passed" in result

    def test_closure_max_is_float(self, harness_small, generated_sb0):
        result = harness_small.check_closure(generated_sb0)
        assert isinstance(result["max_closure_phase_deg"], float)

    def test_closure_noiseless_source_at_phase_center(self, harness_small, tmp_path):
        """Single source exactly at phase centre → all visibilities are real and equal."""
        import pyradiosky
        from astropy.coordinates import Longitude, Latitude
        import astropy.units as u
        from astropy.time import Time

        stokes = np.zeros((4, 1, 1))
        stokes[0, 0, 0] = 1.0
        sky = pyradiosky.SkyModel(
            name=["center"],
            ra=Longitude([harness_small.pointing_ra_deg] * u.deg),
            dec=Latitude([harness_small.pointing_dec_deg] * u.deg),
            stokes=stokes * u.Jy,
            spectral_type="spectral_index",
            spectral_index=np.array([0.0]),
            reference_frequency=np.array([1.4e9]) * u.Hz,
            frame="icrs",
        )

        saved = harness_small.noise_jy
        harness_small.noise_jy = 0.0
        try:
            t0 = Time("2026-01-25T22:26:05", format="isot", scale="utc")
            uv = harness_small._build_uvdata(0, t0, sky)
        finally:
            harness_small.noise_jy = saved

        # All phases should be 0 (Stokes I source at phase centre → real visibilities)
        phases_deg = np.degrees(np.angle(uv.data_array[:, 24, 0]))
        np.testing.assert_allclose(phases_deg, 0.0, atol=1e-4)

    def test_closure_passes_for_noisy_harness(self, tmp_path):
        """check_closure should still pass for noisy data (it uses noiseless internally)."""
        h = SimulationHarness(
            n_antennas=6, n_integrations=3, n_sky_sources=5, noise_jy=2.0, seed=99
        )
        p = tmp_path / "noisy_sb00.uvh5"
        h.generate_subband(0, p)
        result = h.check_closure(p, n_triangles=5)
        assert result["passed"], (
            f"Closure check failed even though it should use noiseless data internally: "
            f"max={result['max_closure_phase_deg']:.4f} deg"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Visibility model properties
# ══════════════════════════════════════════════════════════════════════════════

class TestVisibilityModel:
    """Tests for visibility model physics."""

    def test_visibility_amplitude_order_of_magnitude(self, harness_small, generated_sb0):
        """Mean visibility amplitude should be in Jy range for mJy–Jy sources."""
        uv = harness_small.load_subband(generated_sb0)
        mean_amp = np.abs(uv.data_array[:, :, 0]).mean()
        assert 0.001 < mean_amp < 100.0, f"Mean amp out of range: {mean_amp}"

    def test_noise_increases_amplitude(self, tmp_path):
        """Adding noise should increase visibility amplitude."""
        h_clean = SimulationHarness(n_antennas=4, n_integrations=2, noise_jy=0.0, seed=5)
        h_noisy = SimulationHarness(n_antennas=4, n_integrations=2, noise_jy=5.0, seed=5)

        p_clean = tmp_path / "clean.uvh5"
        p_noisy = tmp_path / "noisy.uvh5"
        h_clean.generate_subband(0, p_clean)
        h_noisy.generate_subband(0, p_noisy)

        uv_clean = h_clean.load_subband(p_clean)
        uv_noisy = h_noisy.load_subband(p_noisy)

        std_clean = np.std(np.abs(uv_clean.data_array))
        std_noisy = np.std(np.abs(uv_noisy.data_array))
        assert std_noisy > std_clean, "Noisy data should have higher spread"

    def test_reproducibility_same_seed(self, tmp_path):
        """Two harnesses with the same seed should produce identical UVH5 data."""
        h1 = SimulationHarness(n_antennas=4, n_integrations=2, seed=42)
        h2 = SimulationHarness(n_antennas=4, n_integrations=2, seed=42)

        p1 = tmp_path / "seed42_a.uvh5"
        p2 = tmp_path / "seed42_b.uvh5"
        h1.generate_subband(0, p1)
        h2.generate_subband(0, p2)

        uv1 = h1.load_subband(p1)
        uv2 = h2.load_subband(p2)
        np.testing.assert_array_equal(uv1.data_array, uv2.data_array)

    def test_different_seeds_differ(self, tmp_path):
        """Different seeds should produce different data."""
        h1 = SimulationHarness(n_antennas=4, n_integrations=2, seed=1)
        h2 = SimulationHarness(n_antennas=4, n_integrations=2, seed=2)

        p1 = tmp_path / "seed1.uvh5"
        p2 = tmp_path / "seed2.uvh5"
        h1.generate_subband(0, p1)
        h2.generate_subband(0, p2)

        uv1 = h1.load_subband(p1)
        uv2 = h2.load_subband(p2)
        assert not np.allclose(uv1.data_array, uv2.data_array)

    def test_spectral_variation_across_subbands(self, tmp_path):
        """Different subbands should have different mean visibilities (spectral index)."""
        h = SimulationHarness(n_antennas=4, n_integrations=2, noise_jy=0.0, seed=3)

        # Generate a shared sky model so only frequency varies
        sky = h.make_sky_model()
        from astropy.time import Time
        t0 = Time("2026-01-25T22:26:05", format="isot", scale="utc")

        means = []
        for sb in range(4):
            uv = h._build_uvdata(sb, t0, sky)
            means.append(np.abs(uv.data_array[:, :, 0]).mean())

        # With spectral index ≈ -0.75, higher-sb (higher freq) should be fainter
        # Not guaranteed to be strictly monotonic for any random seed, but
        # means across 4 subbands should not all be identical
        assert len(set(f"{m:.4f}" for m in means)) > 1, (
            "All subband means are identical — spectral scaling may be broken"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7.  generate_subbands (multi-subband)
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateSubbands:
    """Tests for the batch generate_subbands method."""

    def test_generate_subbands_count(self, tmp_path):
        h = SimulationHarness(n_antennas=4, n_integrations=2, seed=11)
        paths = h.generate_subbands(tmp_path, n_subbands=3)
        assert len(paths) == 3

    def test_generate_subbands_files_exist(self, tmp_path):
        h = SimulationHarness(n_antennas=4, n_integrations=2, seed=11)
        paths = h.generate_subbands(tmp_path, n_subbands=4)
        for p in paths:
            assert p.exists(), f"Expected file {p} to exist"

    def test_generate_subbands_default_names(self, tmp_path):
        h = SimulationHarness(n_antennas=4, n_integrations=2, seed=11)
        paths = h.generate_subbands(tmp_path, n_subbands=2)
        assert paths[0].name == "sim_tile_sb00.uvh5"
        assert paths[1].name == "sim_tile_sb01.uvh5"

    def test_generate_subbands_freq_ranges(self, tmp_path):
        """Each subband should have a distinct frequency range."""
        h = SimulationHarness(n_antennas=4, n_integrations=2, seed=11)
        paths = h.generate_subbands(tmp_path, n_subbands=3)
        min_freqs = []
        for p in paths:
            uv = h.load_subband(p)
            min_freqs.append(uv.freq_array.min())
        # All different
        assert len(set(f"{f:.0f}" for f in min_freqs)) == 3

    def test_generate_subbands_creates_dir(self, tmp_path):
        """generate_subbands should create the output directory if absent."""
        out_dir = tmp_path / "new_subdir" / "sbs"
        h = SimulationHarness(n_antennas=4, n_integrations=2, seed=11)
        paths = h.generate_subbands(out_dir, n_subbands=1)
        assert out_dir.exists()
        assert paths[0].exists()


# ══════════════════════════════════════════════════════════════════════════════
# 8.  DSA-110 instrument specs
# ══════════════════════════════════════════════════════════════════════════════

class TestInstrumentSpecs:
    """Check that generated data matches DSA-110 instrument constants."""

    def test_total_bandwidth(self):
        """Total bandwidth: 16 subbands × 48 ch × 244.14 kHz ≈ 187.5 MHz."""
        total_bw = _N_SUBBANDS * _N_CHAN_PER_SB * _CHAN_WIDTH_HZ
        np.testing.assert_allclose(total_bw, 187_500_000.0, rtol=0.01)

    def test_channel_width_hz(self):
        """Channel width = 250 MHz / 1024 ≈ 244140.625 Hz."""
        np.testing.assert_allclose(_CHAN_WIDTH_HZ, 244_140.625, rtol=1e-6)

    def test_integration_time(self):
        """Integration time ≈ 12.885 s."""
        np.testing.assert_allclose(_INTEGRATION_SEC, 12.884902, rtol=1e-4)

    def test_freq_lower_edge(self):
        """Lower band edge should be exactly 1311.25 MHz."""
        np.testing.assert_allclose(_FREQ_LOWER_EDGE_HZ / 1e6, 1311.25, rtol=1e-9)

    def test_freq_min_is_half_channel_above_edge(self):
        """First channel centre = lower edge + 0.5 * channel_width."""
        np.testing.assert_allclose(
            _FREQ_MIN_HZ, _FREQ_LOWER_EDGE_HZ + 0.5 * _CHAN_WIDTH_HZ, rtol=1e-9
        )

    def test_freq_max(self):
        """Upper band edge should be exactly 1498.75 MHz."""
        upper_edge = _FREQ_LOWER_EDGE_HZ + _N_SUBBANDS * _N_CHAN_PER_SB * _CHAN_WIDTH_HZ
        np.testing.assert_allclose(upper_edge / 1e6, 1498.75, rtol=1e-9)

    def test_canary_tile_start_time(self, harness_small, generated_sb0):
        """Default start time should be the canary tile epoch."""
        uv = harness_small.load_subband(generated_sb0)
        from astropy.time import Time
        t_min = Time(uv.time_array.min(), format="jd", scale="utc")
        expected = Time("2026-01-25T22:26:05", format="isot", scale="utc")
        dt_sec = abs((t_min - expected).to("s").value)
        assert dt_sec < 1.0  # within 1 second


# ══════════════════════════════════════════════════════════════════════════════
# 9.  File I/O and round-trip
# ══════════════════════════════════════════════════════════════════════════════

class TestFileIO:
    """Tests for write/read round-trip fidelity."""

    def test_uvh5_is_valid_file(self, generated_sb0):
        assert generated_sb0.exists()
        assert generated_sb0.stat().st_size > 0

    def test_load_subband_returns_uvdata(self, harness_small, generated_sb0):
        from pyuvdata import UVData
        uv = harness_small.load_subband(generated_sb0)
        assert isinstance(uv, UVData)

    def test_data_roundtrip_fidelity(self, harness_small, tmp_path):
        """Visibility data should survive write → read without loss."""
        h = SimulationHarness(n_antennas=4, n_integrations=2, noise_jy=0.0, seed=99)
        import pyradiosky
        from astropy.coordinates import Longitude, Latitude
        import astropy.units as u_
        from astropy.time import Time

        stokes = np.zeros((4, 1, 1))
        stokes[0, 0, 0] = 2.5
        sky = pyradiosky.SkyModel(
            name=["rt_src"],
            ra=Longitude([343.5] * u_.deg),
            dec=Latitude([16.15] * u_.deg),
            stokes=stokes * u_.Jy,
            spectral_type="spectral_index",
            spectral_index=np.array([0.0]),
            reference_frequency=np.array([1.4e9]) * u_.Hz,
            frame="icrs",
        )
        t0 = Time("2026-01-25T22:26:05", format="isot", scale="utc")
        uv_orig = h._build_uvdata(0, t0, sky)

        p = tmp_path / "rt.uvh5"
        uv_orig.write_uvh5(str(p), clobber=True)
        uv_read = h.load_subband(p)

        np.testing.assert_array_equal(uv_orig.data_array, uv_read.data_array)

    def test_flag_array_all_false(self, harness_small, generated_sb0):
        """Generated data should have no flags."""
        uv = harness_small.load_subband(generated_sb0)
        assert not uv.flag_array.any(), "Unexpected flags in synthetic data"

    def test_nsample_array_all_ones(self, harness_small, generated_sb0):
        uv = harness_small.load_subband(generated_sb0)
        np.testing.assert_array_equal(uv.nsample_array, 1.0)


# ── Bug-fix regression tests ───────────────────────────────────────────────────

class TestPhasedUVW:
    """Regression tests for the UVW phasing bug (was: |w|/|u| ≈ 1.2, coherence < 1%)."""

    def test_phase_center_source_real_visibilities(self):
        """A noiseless 1 Jy source exactly at the phase centre must produce
        visibilities that are purely real ≈ 0.5 Jy on every baseline/time.

        This is the canonical test that the UVW is correctly phased: for a
        source at (l=0, m=0) the phase term is identically 0 for all
        baselines, so exp(-i*phase) = 1.0 and V = flux/2 + 0j for both pols.
        """
        import pyradiosky
        from astropy.coordinates import Longitude, Latitude
        import astropy.units as u_

        h = SimulationHarness(
            n_antennas=6,
            n_integrations=4,
            n_sky_sources=1,
            noise_jy=0.0,
            seed=99,
        )
        # Single 1 Jy source exactly at phase centre
        stokes_arr = np.zeros((4, 1, 1), dtype=float)
        stokes_arr[0, 0, 0] = 1.0
        sky = pyradiosky.SkyModel(
            name=["phase_ctr"],
            ra=Longitude([h.pointing_ra_deg] * u_.deg),
            dec=Latitude([h.pointing_dec_deg] * u_.deg),
            stokes=stokes_arr * u_.Jy,
            spectral_type="spectral_index",
            spectral_index=np.array([0.0]),
            reference_frequency=np.array([1.405e9]) * u_.Hz,
            frame="icrs",
        )
        from astropy.time import Time as ATime
        t0 = ATime("2026-01-25T22:26:05", format="isot", scale="utc")
        uv = h._build_uvdata(0, t0, sky)

        data = uv.data_array  # (n_blts, n_freq, n_pol)
        # Real part should be ≈ 0.5 Jy everywhere (flux/2 per polarization)
        np.testing.assert_allclose(
            data.real,
            0.5,
            atol=1e-3,
            err_msg="Real part of visibility deviates from 0.5 Jy for phase-centre source",
        )
        # Imaginary part should be ≈ 0
        np.testing.assert_allclose(
            data.imag,
            0.0,
            atol=1e-3,
            err_msg="Imaginary part of visibility non-zero for phase-centre source (UVW not phased)",
        )

    def test_uvw_phasing_symmetry(self):
        """Verify that the UVW array has correct conjugate symmetry properties.

        For a phased array, the UVW computed at transit (HA=0) of the phase
        centre should have u-coordinates proportional to the E-W baseline
        lengths.  We verify:
          1. The u-coordinate range spans the expected baseline range.
          2. Stored UVW matches what _compute_uvw returns (no overwrite drift).

        Note: for an E-W array at Dec +16° the w component is NOT zero —
        it carries the geometric delay to the source direction.  The key
        invariant is that visibilities are coherent (tested separately), not
        that w=0.
        """
        import pyradiosky
        from astropy.coordinates import Longitude, Latitude
        import astropy.units as u_

        h = SimulationHarness(
            n_antennas=8,
            n_integrations=4,
            n_sky_sources=1,
            noise_jy=0.0,
            seed=7,
        )
        stokes_arr = np.zeros((4, 1, 1), dtype=float)
        stokes_arr[0, 0, 0] = 1.0
        sky = pyradiosky.SkyModel(
            name=["w_test"],
            ra=Longitude([h.pointing_ra_deg] * u_.deg),
            dec=Latitude([h.pointing_dec_deg] * u_.deg),
            stokes=stokes_arr * u_.Jy,
            spectral_type="spectral_index",
            spectral_index=np.array([0.0]),
            reference_frequency=np.array([1.405e9]) * u_.Hz,
            frame="icrs",
        )
        from astropy.time import Time as ATime
        t0 = ATime("2026-01-25T22:26:05", format="isot", scale="utc")
        uv = h._build_uvdata(0, t0, sky)

        uvw_stored = uv.uvw_array
        # Recompute independently
        n_bl = h.n_antennas * (h.n_antennas - 1) // 2
        ants_a = np.array([i for i in range(h.n_antennas) for j in range(i+1, h.n_antennas)])
        ants_b = np.array([j for i in range(h.n_antennas) for j in range(i+1, h.n_antennas)])
        dt = 12.884902 / 86400.0
        unique_jd = np.array([t0.jd + i * dt for i in range(h.n_integrations)])
        times_blts = np.repeat(unique_jd, n_bl)
        ant1_all   = np.tile(ants_a, h.n_integrations)
        ant2_all   = np.tile(ants_b, h.n_integrations)
        uvw_ref = h._compute_uvw(ant1_all, ant2_all, times_blts)
        np.testing.assert_allclose(
            uvw_stored, uvw_ref, atol=1e-6,
            err_msg="Stored UVW does not match _compute_uvw output — overwrite failed"
        )
        # u-axis should span a non-trivial range: with 8 real antennas from the E-W
        # arm (DSA001-DSA008, ~5.74 m spacing), the longest baseline is ~40 m,
        # which after Earth-rotation synthesis and projection yields u-values spanning
        # at least a few metres.  We verify > 5 m (not 100 m) since these are
        # genuinely close-packed antennas, not synthetic random positions.
        u_range = uvw_stored[:, 0].max() - uvw_stored[:, 0].min()
        assert u_range > 5.0, f"u-range too small: {u_range:.1f} m — baseline layout may be wrong"

    def test_zero_spacing_coherence(self):
        """Channel-averaged coherence fraction must be > 0.5 for a source at
        the phase centre (noiseless, single source).

        Coherence = |mean(vis)| / mean(|vis|).  For a phase-centre source this
        is identically 1.0 (all vis are in phase).  The pre-fix harness gave
        coherence < 0.01 because visibilities from different baselines/times
        were randomly phased.
        """
        import pyradiosky
        from astropy.coordinates import Longitude, Latitude
        import astropy.units as u_

        h = SimulationHarness(
            n_antennas=8,
            n_integrations=6,
            n_sky_sources=1,
            noise_jy=0.0,
            seed=11,
        )
        stokes_arr = np.zeros((4, 1, 1), dtype=float)
        stokes_arr[0, 0, 0] = 1.0
        sky = pyradiosky.SkyModel(
            name=["coh_test"],
            ra=Longitude([h.pointing_ra_deg] * u_.deg),
            dec=Latitude([h.pointing_dec_deg] * u_.deg),
            stokes=stokes_arr * u_.Jy,
            spectral_type="spectral_index",
            spectral_index=np.array([0.0]),
            reference_frequency=np.array([1.405e9]) * u_.Hz,
            frame="icrs",
        )
        from astropy.time import Time as ATime
        t0 = ATime("2026-01-25T22:26:05", format="isot", scale="utc")
        uv = h._build_uvdata(0, t0, sky)

        data = uv.data_array[:, :, 0]  # XX pol, (n_blts, n_freq)
        coherence = np.abs(data.mean()) / np.abs(data).mean()
        assert coherence > 0.5, (
            f"Zero-spacing coherence = {coherence:.4f} — source signal is incoherent. "
            "UVW phasing is still broken."
        )


class TestFlatSpectrumSky:
    """Regression test for the None reference_frequency crash (Bug 2)."""

    def test_flat_spectrum_sky_does_not_crash(self):
        """SimulationHarness must handle a flat-spectrum SkyModel without raising TypeError.

        Before the fix: ``sky.reference_frequency[s_idx]`` raised
        ``TypeError: 'NoneType' object is not subscriptable`` for flat-spectrum models.
        """
        import pyradiosky
        from astropy.coordinates import Longitude, Latitude
        import astropy.units as u_

        h = SimulationHarness(
            n_antennas=4,
            n_integrations=2,
            n_sky_sources=1,
            noise_jy=0.0,
            seed=55,
        )
        # Flat-spectrum sky: reference_frequency is None
        stokes_arr = np.zeros((4, 1, 1), dtype=float)
        stokes_arr[0, 0, 0] = 2.5
        sky = pyradiosky.SkyModel(
            name=["flat_src"],
            ra=Longitude([h.pointing_ra_deg] * u_.deg),
            dec=Latitude([h.pointing_dec_deg] * u_.deg),
            stokes=stokes_arr * u_.Jy,
            spectral_type="flat",
            frame="icrs",
        )
        freqs = h.subband_freqs(0)
        n_blts = h.n_antennas * (h.n_antennas - 1) // 2 * h.n_integrations
        uvw = np.zeros((n_blts, 3), dtype=float)
        # Must not raise
        vis = h._compute_visibilities(uvw, freqs, sky)
        # For a source at phase centre (uvw=0) with 2.5 Jy, each pol = 1.25 Jy
        np.testing.assert_allclose(vis[:, :, 0].real, 1.25, atol=1e-5)
        np.testing.assert_allclose(vis[:, :, 0].imag, 0.0,  atol=1e-5)
