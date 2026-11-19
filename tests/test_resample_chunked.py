"""Chunked resampling of long inputs (praatfan.formant._resample).

Long inputs are resampled in overlapping blocks so that no whole-file FFT is
needed (wasm32 cannot hold one past 2^25 samples). These tests force small
blocks and compare against the whole-signal path. (Rust's chunked path was
checked against this one by hand: a real input past 2^23 samples takes minutes
in pure Python.)
"""
import numpy as np
import pytest

from praatfan.formant import _resample


def _signal(n, sr, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    gate = ((np.arange(n) // 7000) % 3 != 2).astype(float)
    tone = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 1700 * t)
            + 0.02 * np.sin(2 * np.pi * 5400 * t))
    return gate * tone + 1e-3 * (rng.random(n) - 0.5)


def test_short_input_uses_whole_fft():
    x = _signal(30_000, 16_000)
    assert np.array_equal(_resample(x, 16_000, 11_000),
                          _resample(x, 16_000, 11_000, max_whole_nfft=1 << 40))


@pytest.mark.parametrize("old,new", [(16_000, 11_000), (44_100, 16_000), (48_000, 11_025)])
@pytest.mark.parametrize("n", [8191, 8193, 20_000])
def test_chunked_length(n, old, new):
    x = _signal(n, old)
    whole = _resample(x, old, new)
    chunked = _resample(x, old, new, max_whole_nfft=1, core=4096, overlap=256)
    assert len(chunked) == len(whole)


def test_chunked_close_to_whole():
    x = _signal(100_003, 16_000)
    whole = _resample(x, 16_000, 11_000)
    short = _resample(x, 16_000, 11_000, max_whole_nfft=1, core=8192, overlap=256)
    long = _resample(x, 16_000, 11_000, max_whole_nfft=1, core=8192, overlap=4096)
    d_short = np.max(np.abs(short - whole))
    d_long = np.max(np.abs(long - whole))
    assert d_long < d_short
    assert d_long < 1e-4


def test_overlap_must_cover_kernel():
    with pytest.raises(ValueError):
        _resample(_signal(20_000, 16_000), 16_000, 11_000,
                  max_whole_nfft=1, core=4096, overlap=10)
