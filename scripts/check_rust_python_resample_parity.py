"""
Check that the Rust _resample is bit-identical (modulo FFT floating-point
rounding) to the Python _resample.

Both should be the same two-stage algorithm. Differences should be
~ machine precision (1e-15) per sample.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

from praatfan.sound import Sound as PFSound
from praatfan.formant import _resample as py_resample
import praatfan_rust


FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def rust_resample(samples, old_rate, new_rate):
    snd = praatfan_rust.Sound(samples.astype(np.float64), old_rate)
    snd_r = snd.resample(new_rate)
    return np.asarray(snd_r.values(), dtype=np.float64)


def compare(label, samples, old_rate, new_rate):
    py_out = py_resample(samples, old_rate, new_rate)
    rs_out = rust_resample(samples, old_rate, new_rate)
    n = min(len(py_out), len(rs_out))
    print(f"{label}: py n={len(py_out)} rs n={len(rs_out)}")
    if n == 0:
        return
    d = np.abs(py_out[:n] - rs_out[:n])
    bit_identical = np.allclose(py_out[:n], rs_out[:n], atol=0, rtol=0)
    print(f"  bit-identical: {bit_identical}")
    print(f"  mean abs diff: {np.mean(d):.3e}")
    print(f"  max abs diff:  {np.max(d):.3e}")
    print(f"  p99 abs diff:  {np.percentile(d, 99):.3e}")

    # Compare both vs parselmouth
    snd_p = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=old_rate)
    praat = call(snd_p, "Resample", new_rate, 50).values[0]
    np_p = min(n, len(praat))
    py_d = np.abs(py_out[:np_p] - praat[:np_p])
    rs_d = np.abs(rs_out[:np_p] - praat[:np_p])
    print(f"  vs Praat: py mean={np.mean(py_d):.3e}  rs mean={np.mean(rs_d):.3e}")


def main():
    # 1. Real audio 24k → 11k
    snd = parselmouth.Sound(str(FIXTURES / "one_two_three_four_five.wav"))
    orig = snd.values[0].copy()
    compare("Real audio 24k→11k", orig, 24000.0, 11000.0)

    # 2. Synth sine 22k → 11k (integer 2x downsample)
    n = 2000
    t = (np.arange(n) + 0.5) / 22000.0
    x = 0.5 * np.sin(2 * np.pi * 500 * t)
    compare("\n500Hz sine 22k→11k", x, 22000.0, 11000.0)

    # 3. Impulse 24k → 11k
    n = 20000
    x = np.zeros(n); x[1000] = 1.0
    compare("\nImpulse 24k→11k", x, 24000.0, 11000.0)

    # 4. Identity (no resample)
    n = 1000
    x = np.random.default_rng(0).standard_normal(n) * 0.1
    compare("\nIdentity (no resample)", x, 16000.0, 16000.0)


if __name__ == "__main__":
    main()
