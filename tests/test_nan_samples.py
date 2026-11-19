"""A NaN sample must not crash pitch/HNR, and Python and Rust must agree.

praatfan_rust used to panic on any NaN sample in pitch and harmonicity
(the whole-file amplitude statistic). Frames whose window covers the NaN
are undefined; the rest are computed as usual.
"""
import warnings

import numpy as np
import pytest

from praatfan.sound import Sound


def _signal(nan_at):
    rng = np.random.default_rng(0)
    t = np.arange(48000) / 16000
    x = (0.3 * np.sin(2 * np.pi * 150 * t) * (np.sin(2 * np.pi * t) > 0)
         + 1e-3 * rng.standard_normal(t.size))
    x[nan_at] = np.nan
    return x


def _f0(pitch):
    v = pitch.values() if callable(getattr(pitch, "values", None)) else pitch.values
    v = np.asarray(v, dtype=float)
    return np.where(v > 0, v, np.nan)


@pytest.mark.parametrize("nan_at", [[5000], [5000, 20000, 30000]])
@pytest.mark.parametrize("referenced", [False, True])
def test_nan_sample_python_matches_rust(nan_at, referenced):
    pr = pytest.importorskip("praatfan_rust")
    x = _signal(nan_at)
    py, rs = Sound(x, 16000), pr.Sound(x, 16000.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if referenced:
            a = _f0(py.to_pitch_ac_referenced(0.0, 75.0, 600.0))
        else:
            a = _f0(py.to_pitch(0.0, 75.0, 600.0))
    b = _f0(rs.to_pitch_ac_referenced(0.0, 75.0, 600.0) if referenced
            else rs.to_pitch_ac(0.0, 75.0, 600.0))
    assert np.array_equal(np.isnan(a), np.isnan(b))
    both = ~np.isnan(a) & ~np.isnan(b)
    assert both.any()
    assert np.max(np.abs(a[both] - b[both])) < 1e-9
    rs.to_harmonicity_ac(0.01, 75.0, 0.1, 4.5)
    rs.to_harmonicity_cc(0.01, 75.0, 0.1, 1.0)
