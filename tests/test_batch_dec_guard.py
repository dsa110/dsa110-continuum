"""Tests for the Dec-strip guard in batch_pipeline."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_dec_guard_passes_when_dec_matches():
    from batch_pipeline import check_dec_strip
    check_dec_strip(observed_dec=16.1, expected_dec=16.1, threshold_deg=5.0)


def test_dec_guard_passes_within_threshold():
    from batch_pipeline import check_dec_strip
    check_dec_strip(observed_dec=19.0, expected_dec=16.1, threshold_deg=5.0)


def test_dec_guard_raises_on_mismatch():
    from batch_pipeline import check_dec_strip
    with pytest.raises(SystemExit):
        check_dec_strip(observed_dec=33.0, expected_dec=16.1, threshold_deg=5.0)


def test_dec_guard_raises_on_large_negative_mismatch():
    from batch_pipeline import check_dec_strip
    with pytest.raises(SystemExit):
        check_dec_strip(observed_dec=54.5, expected_dec=16.1, threshold_deg=5.0)
