"""
Implement Praat-style windowed-sinc resample and compare against Praat's
Resample output.

Default kernel:
    - precision N = 50
    - for each output sample at time t_out:
        x = t_out * old_rate                        (input sample index, real)
        step = max(old_rate / new_rate, 1)          (stretches kernel on
                                                     downsampling)
        depth = N * step                            (kernel half-width in input
                                                     sample units)
        for k in [ceil(x - depth) .. floor(x + depth)] ∩ [0, n-1]:
            phi = k - x                             (offset, input-sample units)
            arg = phi / step
            sinc  = sin(pi * arg) / (pi * arg)      (sinc(0) := 1)
            win   = 0.5 + 0.5 * cos(pi * phi / depth)
            acc  += input[k] * sinc * win
        output[m] = acc / step
"""

from __future__ import annotations

import math
import numpy as np
import parselmouth
from parselmouth.praat import call
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
FILE = FIXTURES / "one_two_three_four_five.wav"


def sinc_resample(samples: np.ndarray, old_rate: float, new_rate: float,
                  precision: int = 50) -> np.ndarray:
    if abs(old_rate - new_rate) < 1e-9:
        return samples.copy()
    n_in = len(samples)
    n_out = int(math.floor(n_in * new_rate / old_rate))
    step = max(old_rate / new_rate, 1.0)
    depth = precision * step
    out = np.zeros(n_out)
    # Precompute 1/step to avoid overhead
    inv_step = 1.0 / step
    # Observed by black-box against Praat: the resampler uses input-sample-
    # aligned mapping: output sample m (0-indexed) corresponds to input
    # position m * (old_rate/new_rate). Note: this does NOT respect the
    # 0.5-centered metadata convention — empirically Praat's Resample is
    # aligned such that output[0] ≈ input[0] for integer ratios.
    ratio = old_rate / new_rate
    # Formula D (empirically matches Praat best for downsample):
    #   x = m * ratio + 0.5 - 0.5/ratio
    # = m * ratio + (ratio - 1) / (2 * ratio)
    offset = 0.5 - 0.5 / ratio if ratio >= 1.0 else 0.0
    for m in range(n_out):
        x = m * ratio + offset
        low = max(0, int(math.ceil(x - depth)))
        high = min(n_in - 1, int(math.floor(x + depth)))
        acc = 0.0
        for k in range(low, high + 1):
            phi = k - x
            arg = phi * inv_step
            if abs(arg) < 1e-12:
                s = 1.0
            else:
                pa = math.pi * arg
                s = math.sin(pa) / pa
            w = 0.5 + 0.5 * math.cos(math.pi * phi / depth)
            acc += samples[k] * s * w
        out[m] = acc * inv_step
    return out


def main():
    snd = parselmouth.Sound(str(FILE))
    orig = snd.values[0].copy()
    old_rate = snd.sampling_frequency
    new_rate = 11000.0

    # Praat ground truth
    snd_r = call(snd, "Resample", new_rate, 50)
    praat_out = snd_r.values[0].copy()

    # Ours
    print("running sinc_resample... (slow Python, may take ~30 s)")
    ours = sinc_resample(orig, old_rate, new_rate, precision=50)

    n = min(len(praat_out), len(ours))
    praat_out = praat_out[:n]
    ours = ours[:n]

    d = ours - praat_out
    print(f"n_out = {n}")
    print(f"  mean_abs = {np.mean(np.abs(d)):.3e}")
    print(f"  p99      = {np.percentile(np.abs(d), 99):.3e}")
    print(f"  max      = {np.max(np.abs(d)):.3e}")
    print(f"  scale: max|praat|={np.max(np.abs(praat_out)):.4f}")

    # Show where disagreements live
    idx = np.argsort(-np.abs(d))[:10]
    print("top 10 worst samples:")
    for i in idx:
        print(f"    idx={i} (t={i/new_rate:.5f}) ours={ours[i]:+.6f} praat={praat_out[i]:+.6f} diff={d[i]:+.3e}")


if __name__ == "__main__":
    main()
