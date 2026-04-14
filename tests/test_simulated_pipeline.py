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
        paths = h.generate_subbands(output_dir=tmp_path, n_subbands=1)
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

    def test_calibrate_reduces_phase_scatter(self, corrupted_ms):
        """After calibration, cross-corr phases should be more tightly clustered."""
        import casacore.tables as ct
        import numpy as np
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline
        ms_path, cal_path, work_dir = corrupted_ms
        p = SimulatedPipeline(work_dir=work_dir)
        p._calibrate(target_ms=ms_path, cal_uvh5=cal_path,
                     cal_flux_jy=10.0, work_dir=work_dir)
        with ct.table(str(ms_path), readonly=True, ack=False) as t:
            raw = t.getcol("DATA")
            corr = t.getcol("CORRECTED_DATA")
        phase_raw  = np.angle(raw).std()
        phase_corr = np.angle(corr).std()
        assert phase_corr < phase_raw, \
            f"Calibration should reduce phase scatter: raw={phase_raw:.3f}, corr={phase_corr:.3f}"


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
