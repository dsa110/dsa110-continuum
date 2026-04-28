"""Regression tests for WSClean skymodel phase-center selection."""

import numpy as np


def test_predict_uses_dec_from_casa_column_major_phase_dir(tmp_path, monkeypatch):
    """CASA-backed PHASE_DIR shape should feed the correct Dec to WSClean."""
    from dsa110_continuum.adapters import casa_tables
    from dsa110_continuum.calibration import skymodels

    phase_dir = np.array([
        [[np.radians(180.0)], [np.radians(22.0)]],
        [[np.radians(180.0)], [np.radians(22.0)]],
    ])
    commands = []

    class FakeSkyModel:
        Ncomponents = 1

    class FakeTable:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def close(self):
            return None

        def colnames(self):
            if self.path.endswith("::SPECTRAL_WINDOW"):
                return ["CHAN_FREQ", "TOTAL_BANDWIDTH"]
            if self.path.endswith("::FIELD"):
                return ["PHASE_DIR"]
            return ["MODEL_DATA", "CORRECTED_DATA"]

        def nrows(self):
            return 0

        def getcol(self, name):
            if self.path.endswith("::SPECTRAL_WINDOW") and name == "CHAN_FREQ":
                return np.array([[1.4e9, 1.41e9]])
            if self.path.endswith("::FIELD") and name == "PHASE_DIR":
                return phase_dir
            raise KeyError(name)

    def fake_table(path, *args, **kwargs):
        return FakeTable(path)

    def fake_write_source_list(sky_model, txt_path, freq_ghz):
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("# fake source list\n")

    def fake_run(cmd, check, timeout, capture_output):
        commands.append(list(cmd))
        if "-draw-model" in cmd:
            name_idx = cmd.index("-name") + 1
            term_path = f"{cmd[name_idx]}-term-0.fits"
            with open(term_path, "w", encoding="utf-8") as f:
                f.write("fake fits")

    monkeypatch.setattr(casa_tables, "table", fake_table)
    monkeypatch.setattr(skymodels, "write_wsclean_source_list", fake_write_source_list)
    monkeypatch.setattr(skymodels.subprocess, "run", fake_run)

    skymodels.predict_from_skymodel_wsclean(
        "fake.ms",
        FakeSkyModel(),
        field="0~1",
        wsclean_path="/bin/true",
        temp_dir=str(tmp_path),
        cleanup=False,
    )

    draw_cmd = next(cmd for cmd in commands if "-draw-model" in cmd)
    centre_idx = draw_cmd.index("-draw-centre")
    assert draw_cmd[centre_idx + 1] == "12h0m0.000s"
    assert draw_cmd[centre_idx + 2] == "+22d0m0.000s"
