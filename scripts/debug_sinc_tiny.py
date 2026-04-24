"""Tiny case: upsample 2x and inspect each output sample."""
import math
import numpy as np
import parselmouth
from parselmouth.praat import call


def sinc_v1(samples, old_rate, new_rate, precision=50):
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


def main():
    # 500 Hz sine at 11k, upsample to 22k. Use enough samples.
    old_rate = 11000.0
    new_rate = 22000.0
    n = 200
    t = (np.arange(n) + 0.5) / old_rate  # Praat-style centered
    samples = 0.5 * np.sin(2 * np.pi * 500 * t)

    snd = parselmouth.Sound(samples, sampling_frequency=old_rate)
    snd_r = call(snd, "Resample", new_rate, 50)
    praat = snd_r.values[0].copy()

    ours = sinc_v1(samples, old_rate, new_rate, 50)

    n_out = min(len(praat), len(ours))
    print(f"n_out praat={len(praat)}, ours={len(ours)}")
    print(f"first 10 samples:")
    for i in range(10):
        t_out = (i + 0.5) / new_rate
        expected = 0.5 * math.sin(2 * math.pi * 500 * t_out)
        print(f"  i={i} t={t_out:.6f} praat={praat[i]:+.8f} ours={ours[i]:+.8f} "
              f"expected={expected:+.8f} diff_praat={ours[i]-praat[i]:+.2e}")

    # Middle of signal (far from boundaries)
    mid = n_out // 2
    print(f"middle samples ({mid}):")
    for i in range(mid, mid + 5):
        t_out = (i + 0.5) / new_rate
        expected = 0.5 * math.sin(2 * math.pi * 500 * t_out)
        print(f"  i={i} t={t_out:.6f} praat={praat[i]:+.8f} ours={ours[i]:+.8f} "
              f"expected={expected:+.8f} diff_praat={ours[i]-praat[i]:+.2e}")


if __name__ == "__main__":
    main()
