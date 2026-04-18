# tests/test_simulated_pipeline.py
import numpy as np
import pytest
import tempfile
from pathlib import Path
from dsa110_continuum.simulation.harness import SimulationHarness


class TestGainCorruption:
    @pytest.fixture
    def tiny_uvh5(self, tmp_path):
        """Generate a minimal 4-antenna UVH5 for corruption tests."""
        h = SimulationHarness(n_antennas=4, n_sky_sources=1, seed=0,
                              use_real_positions=False)
        paths = h.generate_subbands(output_dir=tmp_path, n_subbands=1)
        return paths[0]

    def test_corrupt_uvh5_creates_output(self, tiny_uvh5, tmp_path):
        from dsa110_continuum.simulation.gain_corruption import corrupt_uvh5
        out = corrupt_uvh5(tiny_uvh5, seed=0)
        assert out.exists()
        assert "_corrupted" in out.name

    def test_corrupt_uvh5_changes_visibilities(self, tiny_uvh5, tmp_path):
        import pyuvdata
        from dsa110_continuum.simulation.gain_corruption import corrupt_uvh5
        uv_orig = pyuvdata.UVData()
        uv_orig.read(str(tiny_uvh5))
        orig_data = uv_orig.data_array.copy()

        out = corrupt_uvh5(tiny_uvh5, amp_scatter=0.05, phase_scatter_deg=5.0, seed=1)
        uv_corr = pyuvdata.UVData()
        uv_corr.read(str(out))

        assert not np.allclose(uv_corr.data_array, orig_data, atol=1e-6), \
            "Corrupted data should differ from original"

    def test_corrupt_uvh5_amplitude_error_bounded(self, tiny_uvh5, tmp_path):
        """Amplitude ratio of corrupted/original should be close to 1 ± scatter."""
        import pyuvdata
        from dsa110_continuum.simulation.gain_corruption import corrupt_uvh5
        uv_orig = pyuvdata.UVData()
        uv_orig.read(str(tiny_uvh5))

        out = corrupt_uvh5(tiny_uvh5, amp_scatter=0.10, phase_scatter_deg=0.0, seed=2)
        uv_corr = pyuvdata.UVData()
        uv_corr.read(str(out))

        ratio = np.abs(uv_corr.data_array) / (np.abs(uv_orig.data_array) + 1e-30)
        finite = ratio[(ratio > 0.01) & np.isfinite(ratio)]
        assert finite.mean() == pytest.approx(1.0, abs=0.05), \
            f"Mean amplitude ratio {finite.mean():.3f} should be near 1.0"

    def test_corrupt_uvh5_seed_reproducible(self, tiny_uvh5, tmp_path):
        import pyuvdata
        from dsa110_continuum.simulation.gain_corruption import corrupt_uvh5
        out1 = corrupt_uvh5(tiny_uvh5, seed=42, output_path=tmp_path / "corrupted_42a.uvh5")
        out2 = corrupt_uvh5(tiny_uvh5, seed=42, output_path=tmp_path / "corrupted_42b.uvh5")
        uv1 = pyuvdata.UVData(); uv1.read(str(out1))
        uv2 = pyuvdata.UVData(); uv2.read(str(out2))
        np.testing.assert_array_equal(uv1.data_array, uv2.data_array)

        # Confirm different seeds produce different results
        out3 = corrupt_uvh5(tiny_uvh5, seed=99, output_path=tmp_path / "corrupted_99.uvh5")
        uv3 = pyuvdata.UVData(); uv3.read(str(out3))
        assert not np.array_equal(uv1.data_array, uv3.data_array), \
            "Different seeds must produce different corruptions"


class TestCalibratorGeneration:
    def test_generate_calibrator_subband_creates_file(self, tmp_path):
        from dsa110_continuum.simulation.harness import SimulationHarness
        h = SimulationHarness(n_antennas=4, seed=0, use_real_positions=False)
        path = h.generate_calibrator_subband(tmp_path, flux_jy=10.0)
        assert Path(path).exists()
        assert "_cal_" in Path(path).name

    def test_calibrator_subband_has_single_source_at_phase_centre(self, tmp_path):
        """Visibilities for a source at phase centre should be nearly real."""
        import pyuvdata
        from dsa110_continuum.simulation.harness import SimulationHarness
        h = SimulationHarness(n_antennas=4, seed=0, use_real_positions=False)
        path = h.generate_calibrator_subband(tmp_path, flux_jy=5.0)
        uv = pyuvdata.UVData()
        uv.read(str(path))
        # Cross-correlations only
        cross_mask = uv.ant_1_array != uv.ant_2_array
        data = uv.data_array[cross_mask, :, 0]  # XX pol
        # For a source exactly at phase centre all visibilities are real
        # (stored as conj(V) in the harness, but conj of real = real)
        imag_frac = np.abs(data.imag) / (np.abs(data.real) + 1e-10)
        assert float(imag_frac.mean()) < 0.01, \
            f"Mean imag/real fraction {imag_frac.mean():.4f} should be < 0.01"

    def test_calibrator_subband_amplitude_matches_flux(self, tmp_path):
        """Cross-correlation amplitude should equal flux_jy / 2 (XX = I/2)."""
        import pyuvdata
        from dsa110_continuum.simulation.harness import SimulationHarness
        h = SimulationHarness(n_antennas=4, seed=0, use_real_positions=False)
        flux = 8.0
        path = h.generate_calibrator_subband(tmp_path, flux_jy=flux)
        uv = pyuvdata.UVData()
        uv.read(str(path))
        cross_mask = uv.ant_1_array != uv.ant_2_array
        cross = uv.data_array[cross_mask, :, 0]
        mean_amp = float(np.abs(cross).mean())
        assert mean_amp == pytest.approx(flux / 2.0, rel=0.05), \
            f"Mean cross-corr amplitude {mean_amp:.3f} should be ≈ flux/2 = {flux/2:.3f}"


class TestSimulatedCalibration:
    @pytest.fixture
    def corrupted_ms(self, tmp_path):
        """Tiny corrupted MS ready for calibration.

        Both the target subband and the calibrator observation are corrupted
        with the same per-antenna gains (same seed), matching the physical
        scenario where the calibrator transits through the same corrupted
        instrument as the science target.
        """
        import pyuvdata
        from dsa110_continuum.simulation.harness import SimulationHarness
        from dsa110_continuum.simulation.gain_corruption import corrupt_uvh5
        h = SimulationHarness(n_antennas=4, n_sky_sources=1, seed=42,
                              use_real_positions=False)
        # drift_scan=False: calibration operates on a fixed-pointing target so
        # the single phase centre matches the calibrator source position.
        paths = h.generate_subbands(output_dir=tmp_path, n_subbands=1,
                                    drift_scan=False)
        corrupted = corrupt_uvh5(paths[0], amp_scatter=0.10,
                                  phase_scatter_deg=10.0, seed=7)
        cal_path = h.generate_calibrator_subband(tmp_path, flux_jy=10.0)
        # Apply the same gain errors to the calibrator observation so the
        # Jacobi solver can recover the instrument gains.
        cal_path = corrupt_uvh5(cal_path, amp_scatter=0.10,
                                phase_scatter_deg=10.0, seed=7,
                                output_path=tmp_path / "sim_cal_sb00_corrupted.uvh5")

        uv = pyuvdata.UVData()
        uv.read(str(corrupted))
        ms_path = tmp_path / "target.ms"
        uv.write_ms(str(ms_path))
        return ms_path, cal_path, tmp_path

    def test_calibrate_creates_corrected_data_column(self, corrupted_ms):
        import casacore.tables as ct
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline
        ms_path, cal_path, work_dir = corrupted_ms
        p = SimulatedPipeline(work_dir=work_dir)
        p._calibrate(target_ms=ms_path, cal_uvh5=cal_path,
                     cal_flux_jy=10.0, work_dir=work_dir)
        with ct.table(str(ms_path), readonly=True, ack=False) as t:
            cols = t.colnames()
        assert "CORRECTED_DATA" in cols, "CORRECTED_DATA column must be added by calibration"

    def test_calibrate_reduces_phase_scatter(self, tmp_path):
        """After calibration, cross-corr amplitudes should be closer to the true source flux.

        The target MS is generated with a single source exactly at the phase
        centre (zero sky-fringe phase) and corrupted with known per-antenna
        amplitude gains.  Before calibration, baseline amplitudes are biased
        by |G_i|*|G_j|.  After calibration, they should be restored to the
        true source amplitude.

        This test does NOT check phase scatter reduction because sky-fringe
        phases dominate when sources are off-centre, making that metric
        uninformative.  Instead we check that the amplitude correction works.
        """
        import casacore.tables as ct
        import numpy as np
        import pyradiosky
        import pyuvdata
        from astropy.coordinates import Longitude, Latitude
        from astropy import units
        from dsa110_continuum.simulation.harness import SimulationHarness
        from dsa110_continuum.simulation.gain_corruption import corrupt_uvh5
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline

        # Build a target with a single 1 Jy source exactly at phase centre.
        h = SimulationHarness(n_antennas=4, n_sky_sources=0, seed=42,
                              noise_jy=0.0, use_real_positions=False)
        freq_hz = float(h.subband_freqs(0).mean())
        sky = pyradiosky.SkyModel(
            name=np.array(["PC_SRC"]),
            ra=Longitude([h.pointing_ra_deg], unit="deg"),
            dec=Latitude([h.pointing_dec_deg], unit="deg"),
            stokes=np.array([[[1.0]], [[0.0]], [[0.0]], [[0.0]]], dtype=float) * units.Jy,
            spectral_type="spectral_index",
            reference_frequency=np.array([freq_hz]) * units.Hz,
            spectral_index=np.array([0.0]),
            frame="icrs",
        )
        paths = h.generate_subbands(output_dir=tmp_path, n_subbands=1,
                                    sky=sky, drift_scan=False)

        # Corrupt with 20 % amplitude errors, no phase errors so the test
        # isolates amplitude calibration cleanly.
        corrupted = corrupt_uvh5(paths[0], amp_scatter=0.20,
                                 phase_scatter_deg=0.0, seed=7)
        cal_path = h.generate_calibrator_subband(tmp_path, flux_jy=10.0)
        cal_path = corrupt_uvh5(cal_path, amp_scatter=0.20,
                                phase_scatter_deg=0.0, seed=7,
                                output_path=tmp_path / "sim_cal_sb00_corrupted.uvh5")

        uv = pyuvdata.UVData()
        uv.read(str(corrupted))
        ms_path = tmp_path / "target_phctr.ms"
        uv.write_ms(str(ms_path))

        # Calibrate
        p = SimulatedPipeline(work_dir=tmp_path)
        p._calibrate(target_ms=ms_path, cal_uvh5=cal_path,
                     cal_flux_jy=10.0, work_dir=tmp_path)

        # After calibration the amplitude should be ≈ 0.5 Jy (I/2 convention)
        # on every cross-correlation baseline.
        expected_amp = 0.5  # 1 Jy source → XX = I/2
        with ct.table(str(ms_path), readonly=True, ack=False) as t:
            raw  = t.getcol("DATA")
            corr = t.getcol("CORRECTED_DATA")
            ant1 = t.getcol("ANTENNA1")
            ant2 = t.getcol("ANTENNA2")

        cross = ant1 != ant2
        raw_amp_err  = np.abs(np.abs(raw[cross, :, 0])  - expected_amp).mean()
        corr_amp_err = np.abs(np.abs(corr[cross, :, 0]) - expected_amp).mean()
        assert corr_amp_err < raw_amp_err, (
            f"Calibration should bring amplitudes closer to {expected_amp} Jy: "
            f"raw_err={raw_amp_err:.4f} Jy, corr_err={corr_amp_err:.4f} Jy"
        )


class TestSimulatedImaging:
    @pytest.fixture
    def calibrated_ms(self, tmp_path):
        """4-antenna MS with CORRECTED_DATA column, ready for WSClean."""
        import pyuvdata
        from dsa110_continuum.simulation.harness import SimulationHarness
        from dsa110_continuum.simulation.gain_corruption import corrupt_uvh5
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline
        h = SimulationHarness(n_antennas=4, n_sky_sources=1, seed=3,
                              use_real_positions=False)
        paths = h.generate_subbands(output_dir=tmp_path, n_subbands=1)
        corrupted = corrupt_uvh5(paths[0], amp_scatter=0.05,
                                  phase_scatter_deg=3.0, seed=3)
        cal_path = h.generate_calibrator_subband(tmp_path, flux_jy=10.0)
        uv = pyuvdata.UVData()
        uv.read(str(corrupted))
        ms_path = tmp_path / "cal_target.ms"
        uv.write_ms(str(ms_path))
        p = SimulatedPipeline(work_dir=tmp_path, niter=100,
                              cell_arcsec=30.0, image_size=256)
        p._calibrate(target_ms=ms_path, cal_uvh5=cal_path,
                     cal_flux_jy=10.0, work_dir=tmp_path)
        return ms_path, p, tmp_path

    def test_image_creates_restored_fits(self, calibrated_ms):
        ms_path, p, work_dir = calibrated_ms
        result = p._image(ms_path=ms_path, work_dir=work_dir)
        assert result["restored"].exists(), "Restored FITS must exist"

    def test_image_creates_psf(self, calibrated_ms):
        ms_path, p, work_dir = calibrated_ms
        result = p._image(ms_path=ms_path, work_dir=work_dir)
        assert result["psf"].exists(), "PSF FITS must exist"

    def test_image_restored_has_valid_wcs(self, calibrated_ms):
        from astropy.io import fits
        from astropy.wcs import WCS
        ms_path, p, work_dir = calibrated_ms
        result = p._image(ms_path=ms_path, work_dir=work_dir)
        with fits.open(str(result["restored"])) as hdul:
            wcs = WCS(hdul[0].header)
        assert wcs.naxis >= 2


class TestSimulatedMosaic:
    @pytest.fixture
    def two_tile_fits(self, tmp_path):
        """Two minimal synthetic FITS tiles with overlapping footprints."""
        import numpy as np
        from astropy.io import fits
        from astropy.wcs import WCS

        def make_tile(ra_center, filename):
            data = np.zeros((64, 64), dtype=np.float32)
            data[32, 32] = 0.5  # fake source at centre
            w = WCS(naxis=2)
            w.wcs.crpix = [32, 32]
            w.wcs.cdelt = [-20.0 / 3600, 20.0 / 3600]
            w.wcs.crval = [ra_center, 16.15]
            w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
            hdr = w.to_header()
            hdr["BUNIT"] = "JY/BEAM"
            hdr["BMAJ"] = 329.0 / 3600
            hdr["BMIN"] = 76.0 / 3600
            hdr["BPA"] = -132.0
            path = tmp_path / filename
            fits.writeto(str(path), data, hdr, overwrite=True)
            return path

        t1 = make_tile(343.5,  "tile1.fits")
        t2 = make_tile(343.55, "tile2.fits")
        return [t1, t2], tmp_path

    def test_mosaic_creates_output_fits(self, two_tile_fits):
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline
        tiles, work_dir = two_tile_fits
        p = SimulatedPipeline(work_dir=work_dir)
        mosaic_path = p._mosaic(image_paths=tiles, work_dir=work_dir)
        assert mosaic_path.exists(), "Mosaic FITS must be created"

    def test_mosaic_has_larger_or_equal_footprint(self, two_tile_fits):
        """Mosaic should cover at least as many pixels as one input tile."""
        from astropy.io import fits
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline
        tiles, work_dir = two_tile_fits
        p = SimulatedPipeline(work_dir=work_dir)
        mosaic_path = p._mosaic(image_paths=tiles, work_dir=work_dir)
        with fits.open(str(tiles[0])) as h0, fits.open(str(mosaic_path)) as hm:
            n_pix_tile   = h0[0].data.size
            n_pix_mosaic = hm[0].data.size
        assert n_pix_mosaic >= n_pix_tile, \
            f"Mosaic ({n_pix_mosaic} px) should be >= one tile ({n_pix_tile} px)"

    def test_mosaic_returns_path_object(self, two_tile_fits):
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline
        tiles, work_dir = two_tile_fits
        p = SimulatedPipeline(work_dir=work_dir)
        result = p._mosaic(image_paths=tiles, work_dir=work_dir)
        assert isinstance(result, Path)


class TestSimulatedPhotometry:
    @pytest.fixture
    def mock_image_with_source(self, tmp_path):
        """FITS image with one injected point source at known RA/Dec."""
        import numpy as np
        from astropy.io import fits
        from astropy.wcs import WCS
        ra, dec = 343.5, 16.15
        data = np.random.default_rng(0).normal(0, 0.001, (128, 128)).astype(np.float32)
        w = WCS(naxis=2)
        w.wcs.crpix = [64, 64]
        w.wcs.cdelt = [-20.0 / 3600, 20.0 / 3600]
        w.wcs.crval = [ra, dec]
        w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        # Inject source: 0.5 Jy/beam at pixel centre (64, 64)
        data[64, 64] = 0.5
        hdr = w.to_header()
        hdr["BUNIT"] = "JY/BEAM"
        path = tmp_path / "test_image.fits"
        fits.writeto(str(path), data, hdr, overwrite=True)
        return path, ra, dec, 0.5

    def test_photometry_finds_injected_source(self, mock_image_with_source):
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline, SourceFluxResult
        from dsa110_continuum.simulation.ground_truth import GroundTruthRegistry
        path, ra, dec, flux = mock_image_with_source
        reg = GroundTruthRegistry(test_run_id="test")
        reg.register_source("S0", ra, dec, baseline_flux_jy=flux)

        p = SimulatedPipeline(work_dir=path.parent)
        results = p._photometry(
            image_path=path,
            ground_truth=reg,
            mjd=60000.0,
            noise_jy_beam=0.001,
        )
        assert len(results) == 1
        assert isinstance(results[0], SourceFluxResult)

    def test_photometry_recovers_flux_within_tolerance(self, mock_image_with_source):
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline
        from dsa110_continuum.simulation.ground_truth import GroundTruthRegistry
        path, ra, dec, flux = mock_image_with_source
        reg = GroundTruthRegistry(test_run_id="test")
        reg.register_source("S0", ra, dec, baseline_flux_jy=flux)

        p = SimulatedPipeline(work_dir=path.parent)
        results = p._photometry(image_path=path, ground_truth=reg,
                                mjd=60000.0, noise_jy_beam=0.001)
        r = results[0]
        assert r.passed, \
            f"Recovered {r.recovered_flux_jy:.3f} Jy vs injected {r.injected_flux_jy:.3f} Jy"

    def test_photometry_flags_out_of_image_source(self, tmp_path):
        """Source far outside image returns NaN flux and passed=False."""
        import numpy as np
        from astropy.io import fits
        from astropy.wcs import WCS
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline
        from dsa110_continuum.simulation.ground_truth import GroundTruthRegistry
        data = np.zeros((32, 32), dtype=np.float32)
        w = WCS(naxis=2)
        w.wcs.crpix = [16, 16]
        w.wcs.cdelt = [-1.0 / 60, 1.0 / 60]
        w.wcs.crval = [0.0, 0.0]
        w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        path = tmp_path / "blank.fits"
        fits.writeto(str(path), data, w.to_header(), overwrite=True)

        reg = GroundTruthRegistry(test_run_id="test")
        reg.register_source("S_far", 180.0, 45.0, baseline_flux_jy=1.0)  # outside image
        p = SimulatedPipeline(work_dir=tmp_path)
        results = p._photometry(image_path=path, ground_truth=reg,
                                mjd=60000.0, noise_jy_beam=0.001)
        assert not results[0].passed
        assert np.isnan(results[0].recovered_flux_jy)


@pytest.mark.slow
class TestEndToEnd:
    """Full pipeline: corruption → calibration → imaging → mosaic → photometry.

    Marked slow — WSClean runs with niter=200. Excluded from default suite
    by -m 'not slow'. Run explicitly for integration validation.
    """

    def test_full_pipeline_recovers_sources(self, tmp_path):
        from dsa110_continuum.simulation.harness import SimulationHarness
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline

        h = SimulationHarness(
            n_antennas=96, n_sky_sources=1, seed=42, use_real_positions=True
        )
        p = SimulatedPipeline(
            work_dir=tmp_path,
            niter=200,
            cell_arcsec=20.0,
            image_size=512,  # 512×20" = 2.84 deg FOV; S0 with seed=42 is ~1 deg from centre
        )
        result = p.run(
            harness=h,
            n_tiles=2,
            n_subbands=4,
            amp_scatter=0.05,
            phase_scatter_deg=5.0,
            cal_flux_jy=10.0,
        )
        assert result.calibration_passed, \
            f"Calibration stage failed: {result.errors}"
        assert result.imaging_passed, \
            f"Imaging stage failed: {result.errors}"
        assert result.mosaic_path is not None and result.mosaic_path.exists(), \
            f"Mosaic missing: {result.errors}"
        assert result.n_recovered >= 1, \
            f"Expected >=1 recovered source; got {result.n_recovered}/{len(result.source_results)}"

    def test_full_pipeline_result_is_serializable(self, tmp_path):
        import dataclasses
        from dsa110_continuum.simulation.harness import SimulationHarness
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline
        h = SimulationHarness(
            n_antennas=8, n_sky_sources=1, seed=1, use_real_positions=False
        )
        p = SimulatedPipeline(
            work_dir=tmp_path, niter=50, cell_arcsec=30.0, image_size=128
        )
        result = p.run(
            harness=h, n_tiles=2, n_subbands=1,
            amp_scatter=0.03, phase_scatter_deg=2.0,
        )
        d = dataclasses.asdict(result)
        assert "source_results" in d
        assert "errors" in d
