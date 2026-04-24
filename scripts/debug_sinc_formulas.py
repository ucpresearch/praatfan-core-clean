"""Try multiple x_pos formulas and see which one Praat uses.

Candidates:
  A. x = m * ratio                         (integer-aligned; matches upsample obs)
  B. x = (m + 0.5) * ratio - 0.5           (0.5-centered time mapping)
  C. x = m * ratio + offset                (some fixed offset)
"""
import math
import numpy as np
import parselmouth
from parselmouth.praat import call


def sinc(samples, old_rate, new_rate, precision, x_of_m):
    n_in = len(samples)
    n_out = int(math.floor(n_in * new_rate / old_rate))
    step = max(old_rate / new_rate, 1.0)
    depth = precision * step
    out = np.zeros(n_out)
    inv_step = 1.0 / step
    for m in range(n_out):
        x = x_of_m(m, old_rate / new_rate)
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


def test(label, old_rate, new_rate, samples):
    snd = parselmouth.Sound(samples, sampling_frequency=old_rate)
    praat = call(snd, "Resample", new_rate, 50).values[0].copy()

    out_A = sinc(samples, old_rate, new_rate, 50, lambda m, r: m * r)
    out_B = sinc(samples, old_rate, new_rate, 50, lambda m, r: (m + 0.5) * r - 0.5)
    out_C = sinc(samples, old_rate, new_rate, 50, lambda m, r: m * r + 0.5 * (r - 1))
    out_D = sinc(samples, old_rate, new_rate, 50, lambda m, r: m * r + 0.5 - 0.5 / r)

    n = min(len(praat), len(out_A), len(out_B))
    praat = praat[:n]
    for name, arr in [("A: m*r        ", out_A[:n]),
                       ("B: (m+.5)*r-.5", out_B[:n]),
                       ("C: m*r+.5(r-1)", out_C[:n]),
                       ("D: m*r+.5-.5/r", out_D[:n])]:
        d = arr - praat
        print(f"  {name}: mean={np.mean(np.abs(d)):.3e} p99={np.percentile(np.abs(d), 99):.3e} max={np.max(np.abs(d)):.3e}")


def main():
    # Tiny 500Hz sine upsample 2x
    n = 200
    old = 11000.0; new = 22000.0
    t = (np.arange(n) + 0.5) / old
    x = 0.5 * np.sin(2 * np.pi * 500 * t)
    print("== 500Hz upsample 11k→22k (n=200)")
    test("", old, new, x)

    # Downsample 24k → 11k
    old = 24000.0; new = 11000.0
    n = 2000
    t = (np.arange(n) + 0.5) / old
    x = 0.5 * np.sin(2 * np.pi * 500 * t)
    print("\n== 500Hz downsample 24k→11k (n=2000)")
    test("", old, new, x)

    # Random noise, 24k→11k
    rng = np.random.default_rng(0)
    old = 24000.0; new = 11000.0
    x = rng.standard_normal(2000) * 0.1
    print("\n== noise downsample 24k→11k (n=2000)")
    test("", old, new, x)


if __name__ == "__main__":
    main()
