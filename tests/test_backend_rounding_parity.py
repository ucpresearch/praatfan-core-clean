"""Pure Python and Rust must pick the same samples when a frame centre falls
exactly on a half sample (any odd-length input at the default grids).

Python's round() is half-to-even, Rust's f64::round half-away-from-zero; until
0.1.16 that made whole files diverge between the backends (praatfan._rounding).
"""
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

praatfan_rust = pytest.importorskip("praatfan_rust")
import praatfan

FIX = Path(__file__).parent / "fixtures" / "one_two_three_four_five_16k.wav"


@pytest.mark.parametrize("trim", [0, 1])
def test_pitch_and_hnr_identical_across_backends(trim):
    x, sr = sf.read(FIX, dtype="float64")
    n = len(x) - trim
    if n % 2 == 0:
        n -= 1                      # odd length -> half-sample frame centres
    x = x[:n]
    out = {}
    for b in ("praatfan", "praatfan_rust"):
        praatfan.set_backend(b)
        s = praatfan.Sound(x, sampling_frequency=sr)
        out[b] = [np.asarray(s.to_pitch_ac().strengths()),
                  np.asarray(s.to_pitch_cc().strengths()),
                  np.asarray(s.to_harmonicity_ac().values()),
                  np.asarray(s.to_harmonicity_cc().values())]
    for a, b in zip(out["praatfan"], out["praatfan_rust"]):
        assert a.shape == b.shape
        assert np.array_equal(np.isnan(a), np.isnan(b))
        m = ~np.isnan(a)
        assert np.max(np.abs(a[m] - b[m])) < 1e-9
