"""
Debug the sinc resampler: test on synthetic signals with controlled spectra.

If my algo is correct, low-bandwidth signals (well below new Nyquist) should
match Praat bit-for-bit. If they diverge, the kernel/window/normalization
math is wrong.
"""

from __future__ import annotations

import math
import numpy as np
import parselmouth
from parselmouth.praat import call


def sinc_resample_v1(samples, old_rate, new_rate, precision=50):
    """Symmetric window clamp; centered (0.5-offset) timing."""
    n_in = len(samples)
    n_out = int(math.floor(n_in * new_rate / old_rate))
    step = max(old_rate / new_rate, 1.0)
    depth = precision * step
    ratio = old_rate / new_rate
    out = np.zeros(n_out)
    inv_step = 1.0 / step
    for m in range(n_out):
        x = (m + 0.5) * ratio - 0.5
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


def compare(label, old_rate, new_rate, samples, precision=50):
    # Praat
    snd = parselmouth.Sound(samples, sampling_frequency=old_rate)
    snd_r = call(snd, "Resample", new_rate, precision)
    praat_out = snd_r.values[0].copy()

    # Ours
    ours = sinc_resample_v1(samples, old_rate, new_rate, precision)
    n = min(len(praat_out), len(ours))
    praat_out = praat_out[:n]
    ours = ours[:n]
    d = ours - praat_out
    scale = np.max(np.abs(praat_out)) or 1.0
    print(f"{label}: n={n} mean_abs={np.mean(np.abs(d)):.2e} p99={np.percentile(np.abs(d), 99):.2e} "
          f"max={np.max(np.abs(d)):.2e}   (scale={scale:.4f}, rel_max={np.max(np.abs(d))/scale:.2e})")
    return d, praat_out, ours


def main():
    # Signal 1: 500 Hz sine at sr=24000, resample to 11000 (well below Nyquist)
    n = 24000 * 2
    t = np.arange(n) / 24000
    x = 0.5 * np.sin(2 * np.pi * 500 * t)
    compare("500Hz sine 24→11k", 24000, 11000, x)

    # Signal 2: 4000 Hz sine at sr=24000, resample to 11000 (closer to out Nyquist=5500)
    x = 0.5 * np.sin(2 * np.pi * 4000 * t)
    compare("4000Hz sine 24→11k", 24000, 11000, x)

    # Signal 3: random noise, 24→11k (broadband, hits Nyquist)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n) * 0.1
    compare("noise 24→11k     ", 24000, 11000, x)

    # Signal 4: 500 Hz sine, integer 2x upsampling: 11k→22k
    n2 = 11000 * 2
    t2 = np.arange(n2) / 11000
    x2 = 0.5 * np.sin(2 * np.pi * 500 * t2)
    compare("500Hz sine 11→22k", 11000, 22000, x2)

    # Signal 5: 500 Hz sine, integer 2x downsampling: 22k→11k
    n3 = 22000 * 2
    t3 = np.arange(n3) / 22000
    x3 = 0.5 * np.sin(2 * np.pi * 500 * t3)
    compare("500Hz sine 22→11k", 22000, 11000, x3)


if __name__ == "__main__":
    main()
