"""
Test boundary-depth options for the windowed-sinc resampler.

Focus: match Praat's output at BOUNDARY samples (first/last ~100 samples).
"""
from __future__ import annotations

import math
import numpy as np
import parselmouth
from parselmouth.praat import call
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def sinc_option(samples, old_rate, new_rate, precision=50, variant="A"):
    """
    variant:
      A: symmetric depth = precision*step (zero-pad)
      B: symmetric depth = precision*step + 0.5 (half-sample wider)
      C: asymmetric: leftDepth/rightDepth clipped to available samples
      D: reflect-extend samples at boundary (symmetric depth)
      E: constant-extend samples at boundary (clamp to first/last)
    """
    n_in = len(samples)
    n_out = int(math.floor(n_in * new_rate / old_rate))
    step = max(old_rate / new_rate, 1.0)
    ratio = old_rate / new_rate
    inv_step = 1.0 / step
    base_depth = precision * step

    out = np.zeros(n_out)
    for m in range(n_out):
        x = (m + 0.5) * ratio - 0.5
        if variant == "A":
            depth = base_depth
            low = max(0, int(math.ceil(x - depth)))
            high = min(n_in - 1, int(math.floor(x + depth)))
            acc = 0.0
            for k in range(low, high + 1):
                phi = k - x
                s = np.sinc(phi * inv_step)
                w = 0.5 + 0.5 * math.cos(math.pi * phi / depth)
                acc += samples[k] * s * w
            out[m] = acc * inv_step
        elif variant == "B":
            depth = base_depth + 0.5
            low = max(0, int(math.ceil(x - depth)))
            high = min(n_in - 1, int(math.floor(x + depth)))
            acc = 0.0
            for k in range(low, high + 1):
                phi = k - x
                s = np.sinc(phi * inv_step)
                w = 0.5 + 0.5 * math.cos(math.pi * phi / depth)
                acc += samples[k] * s * w
            out[m] = acc * inv_step
        elif variant == "C":
            # Asymmetric: shrink left/right depth when near boundary
            left_depth = min(base_depth, x + 0.5)
            right_depth = min(base_depth, (n_in - 1 - x) + 0.5)
            low = max(0, int(math.ceil(x - left_depth)))
            high = min(n_in - 1, int(math.floor(x + right_depth)))
            acc = 0.0
            for k in range(low, high + 1):
                phi = k - x
                s = np.sinc(phi * inv_step)
                # Two-sided hann: scale phi to left/right depth on each side
                if phi < 0:
                    w = 0.5 + 0.5 * math.cos(math.pi * phi / left_depth)
                else:
                    w = 0.5 + 0.5 * math.cos(math.pi * phi / right_depth)
                acc += samples[k] * s * w
            out[m] = acc * inv_step
        elif variant == "D":
            # Reflect-extend (mirror at boundaries)
            depth = base_depth
            low = int(math.ceil(x - depth))
            high = int(math.floor(x + depth))
            acc = 0.0
            for k in range(low, high + 1):
                if k < 0:
                    src = -k - 1  # reflect: -1 -> 0, -2 -> 1
                elif k >= n_in:
                    src = 2 * n_in - k - 1
                else:
                    src = k
                if 0 <= src < n_in:
                    phi = k - x
                    s = np.sinc(phi * inv_step)
                    w = 0.5 + 0.5 * math.cos(math.pi * phi / depth)
                    acc += samples[src] * s * w
            out[m] = acc * inv_step
        elif variant == "E":
            # Constant extension (clamp)
            depth = base_depth
            low = int(math.ceil(x - depth))
            high = int(math.floor(x + depth))
            acc = 0.0
            for k in range(low, high + 1):
                src = max(0, min(n_in - 1, k))
                phi = k - x
                s = np.sinc(phi * inv_step)
                w = 0.5 + 0.5 * math.cos(math.pi * phi / depth)
                acc += samples[src] * s * w
            out[m] = acc * inv_step
    return out


def main():
    # Use real audio for the test
    snd = parselmouth.Sound(str(FIXTURES / "one_two_three_four_five.wav"))
    orig = snd.values[0].copy()
    old_rate = snd.sampling_frequency
    new_rate = 11000.0

    praat = call(snd, "Resample", new_rate, 50).values[0].copy()

    for variant in ["A", "B", "C", "D", "E"]:
        ours = sinc_option(orig, old_rate, new_rate, 50, variant)
        n = min(len(praat), len(ours))
        d = ours[:n] - praat[:n]
        # Separate into boundary (first/last 100) vs interior
        boundary_mask = np.zeros(n, dtype=bool)
        boundary_mask[:100] = True
        boundary_mask[-100:] = True
        db = d[boundary_mask]
        di = d[~boundary_mask]
        print(f"variant {variant}: "
              f"boundary mean={np.mean(np.abs(db)):.2e} max={np.max(np.abs(db)):.2e} | "
              f"interior mean={np.mean(np.abs(di)):.2e} max={np.max(np.abs(di)):.2e}")


if __name__ == "__main__":
    main()
