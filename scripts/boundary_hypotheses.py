"""
Boundary-handling hypothesis testing for Praat `Sound: Resample` parity.

Tests multiple candidate formulas for how the windowed-sinc kernel handles
near-boundary samples. Ground truth: parselmouth's `Resample`.

Variants (all share the 0.5-centered time mapping and Hann window):
  (a) current_zeropad         — depth=precision*step, zero-pad outside
  (b) symmetric_half_sample   — depth=precision*step + 0.5, zero-pad outside
  (c) asymmetric              — leftDepth=x-left+1, rightDepth=right-x+1 (per GPL doc)
  (d) reflect                 — mirror samples across boundary indices
  (e) edge_clamp              — samples[0] / samples[-1] for out-of-bound
  (f) hypothesis_outer_ceil   — depth widened by +1.0 on each side, zero-pad
  (g) full_kernel_norm        — current kernel, but normalize by sum(window) not 1/step
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "one_two_three_four_five.wav"


# ---------- Variant implementations -----------------------------------------

def _common_params(n_in: int, old_rate: float, new_rate: float, precision: int):
    n_out = int(math.floor(n_in * new_rate / old_rate))
    step = max(old_rate / new_rate, 1.0)
    depth = precision * step
    ratio = old_rate / new_rate
    return n_out, step, depth, ratio


def resample_a_current(samples, old_rate, new_rate, precision=50):
    n_in = len(samples)
    n_out, step, depth, ratio = _common_params(n_in, old_rate, new_rate, precision)
    out = np.empty(n_out, dtype=np.float64)
    for m in range(n_out):
        x = (m + 0.5) * ratio - 0.5
        low = max(0, int(math.ceil(x - depth)))
        high = min(n_in - 1, int(math.floor(x + depth)))
        if high < low:
            out[m] = 0.0
            continue
        k = np.arange(low, high + 1, dtype=np.float64)
        phi = k - x
        s = np.sinc(phi / step)
        w = 0.5 + 0.5 * np.cos(np.pi * phi / depth)
        out[m] = np.dot(samples[low:high + 1], s * w) / step
    return out


def resample_b_sym_half(samples, old_rate, new_rate, precision=50):
    n_in = len(samples)
    n_out, step, depth0, ratio = _common_params(n_in, old_rate, new_rate, precision)
    depth = depth0 + 0.5
    out = np.empty(n_out, dtype=np.float64)
    for m in range(n_out):
        x = (m + 0.5) * ratio - 0.5
        low = max(0, int(math.ceil(x - depth)))
        high = min(n_in - 1, int(math.floor(x + depth)))
        if high < low:
            out[m] = 0.0
            continue
        k = np.arange(low, high + 1, dtype=np.float64)
        phi = k - x
        s = np.sinc(phi / step)
        w = 0.5 + 0.5 * np.cos(np.pi * phi / depth)
        out[m] = np.dot(samples[low:high + 1], s * w) / step
    return out


def resample_c_asymmetric(samples, old_rate, new_rate, precision=50):
    n_in = len(samples)
    n_out, step, depth0, ratio = _common_params(n_in, old_rate, new_rate, precision)
    out = np.empty(n_out, dtype=np.float64)
    for m in range(n_out):
        x = (m + 0.5) * ratio - 0.5
        low = max(0, int(math.ceil(x - depth0)))
        high = min(n_in - 1, int(math.floor(x + depth0)))
        if high < low:
            out[m] = 0.0
            continue
        left_depth = x - low + 1.0
        right_depth = high - x + 1.0
        k = np.arange(low, high + 1, dtype=np.float64)
        phi = k - x
        s = np.sinc(phi / step)
        w = np.where(phi < 0,
                     0.5 + 0.5 * np.cos(np.pi * phi / left_depth),
                     0.5 + 0.5 * np.cos(np.pi * phi / right_depth))
        out[m] = np.dot(samples[low:high + 1], s * w) / step
    return out


def _virtual_sample(samples, idx, mode):
    n = len(samples)
    if 0 <= idx < n:
        return samples[idx]
    if mode == "zero":
        return 0.0
    if mode == "clamp":
        return samples[0] if idx < 0 else samples[-1]
    if mode == "reflect":
        # mirror: index -1 -> 0, -2 -> 1, n -> n-1, n+1 -> n-2  (no-repeat of edge)
        if idx < 0:
            j = -idx - 1
        else:
            j = 2 * n - idx - 1
        if 0 <= j < n:
            return samples[j]
        return 0.0
    return 0.0


def _resample_with_extension(samples, old_rate, new_rate, precision, mode,
                              depth_bonus=0.0):
    n_in = len(samples)
    n_out, step, depth0, ratio = _common_params(n_in, old_rate, new_rate, precision)
    depth = depth0 + depth_bonus
    out = np.empty(n_out, dtype=np.float64)
    for m in range(n_out):
        x = (m + 0.5) * ratio - 0.5
        low = int(math.ceil(x - depth))
        high = int(math.floor(x + depth))
        acc = 0.0
        for k in range(low, high + 1):
            phi = k - x
            if step == 1.0:
                s = 1.0 if phi == 0 else math.sin(math.pi * phi) / (math.pi * phi)
            else:
                a = math.pi * phi / step
                s = 1.0 if a == 0 else math.sin(a) / a
            w = 0.5 + 0.5 * math.cos(math.pi * phi / depth)
            acc += _virtual_sample(samples, k, mode) * s * w
        out[m] = acc / step
    return out


def resample_d_reflect(samples, old_rate, new_rate, precision=50):
    return _resample_with_extension(samples, old_rate, new_rate, precision, "reflect")


def resample_e_edge_clamp(samples, old_rate, new_rate, precision=50):
    return _resample_with_extension(samples, old_rate, new_rate, precision, "clamp")


def resample_f_outer_ceil(samples, old_rate, new_rate, precision=50):
    """Own hypothesis: widen depth by +1.0 (next full lobe), zero-pad outside.
    Rationale: Praat's `NUMsinc` family often rounds `ceil(precision*step)` up
    to a whole number of zero-crossings, effectively adding ~1 input sample."""
    n_in = len(samples)
    n_out, step, depth0, ratio = _common_params(n_in, old_rate, new_rate, precision)
    depth = math.ceil(depth0)  # round up to integer
    out = np.empty(n_out, dtype=np.float64)
    for m in range(n_out):
        x = (m + 0.5) * ratio - 0.5
        low = max(0, int(math.ceil(x - depth)))
        high = min(n_in - 1, int(math.floor(x + depth)))
        if high < low:
            out[m] = 0.0
            continue
        k = np.arange(low, high + 1, dtype=np.float64)
        phi = k - x
        s = np.sinc(phi / step)
        w = 0.5 + 0.5 * np.cos(np.pi * phi / depth)
        out[m] = np.dot(samples[low:high + 1], s * w) / step
    return out


def resample_g_window_norm(samples, old_rate, new_rate, precision=50):
    """Variant: normalize by sum(window) rather than 1/step.  Often used when
    boundary truncation removes mass; this renormalizes each output sample."""
    n_in = len(samples)
    n_out, step, depth, ratio = _common_params(n_in, old_rate, new_rate, precision)
    out = np.empty(n_out, dtype=np.float64)
    for m in range(n_out):
        x = (m + 0.5) * ratio - 0.5
        low = max(0, int(math.ceil(x - depth)))
        high = min(n_in - 1, int(math.floor(x + depth)))
        if high < low:
            out[m] = 0.0
            continue
        k = np.arange(low, high + 1, dtype=np.float64)
        phi = k - x
        s = np.sinc(phi / step)
        w = 0.5 + 0.5 * np.cos(np.pi * phi / depth)
        kernel = s * w
        denom = kernel.sum()
        if abs(denom) < 1e-12:
            out[m] = 0.0
        else:
            out[m] = np.dot(samples[low:high + 1], kernel) / denom
    return out


VARIANTS = {
    "a_current_zeropad":     resample_a_current,
    "b_symmetric_half":      resample_b_sym_half,
    "c_asymmetric":          resample_c_asymmetric,
    "d_reflect":             resample_d_reflect,
    "e_edge_clamp":          resample_e_edge_clamp,
    "f_outer_ceil":          resample_f_outer_ceil,
    "g_window_norm":         resample_g_window_norm,
}


# ---------- Ground truth & scoring ------------------------------------------

def praat_resample(samples, old_rate, new_rate, precision=50):
    snd = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=old_rate)
    res = call(snd, "Resample", new_rate, precision)
    return res.values[0]


def score(name, got, truth):
    n = min(len(got), len(truth))
    got = got[:n]
    truth = truth[:n]
    diff = got - truth
    absdiff = np.abs(diff)
    b = min(200, n)
    left = absdiff[:b]
    right = absdiff[-b:]
    mid = absdiff[b:-b] if n > 2 * b else absdiff
    return {
        "name": name,
        "n": n,
        "mean": absdiff.mean(),
        "max": absdiff.max(),
        "left_mean": left.mean(), "left_max": left.max(),
        "right_mean": right.mean(), "right_max": right.max(),
        "mid_mean": mid.mean(), "mid_max": mid.max(),
        "bit_exact": bool(np.all(diff == 0.0)),
        "near_exact": bool(absdiff.max() < 1e-12),
    }


def print_table(results, title):
    print(f"\n=== {title} ===")
    hdr = ("variant", "mean", "max", "L_mean", "L_max", "R_mean", "R_max", "mid_max", "bit?")
    print("{:22s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>6s}".format(*hdr))
    for r in results:
        print("{:22s} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:>6s}".format(
            r["name"], r["mean"], r["max"], r["left_mean"], r["left_max"],
            r["right_mean"], r["right_max"], r["mid_max"],
            "YES" if r["bit_exact"] else ("~0" if r["near_exact"] else "no")
        ))


def run_test(title, samples, old_rate, new_rate, precision=50):
    print(f"\n\n########## {title} ##########")
    print(f"  n_in={len(samples)}  old_rate={old_rate}  new_rate={new_rate}  precision={precision}")
    truth = praat_resample(samples, old_rate, new_rate, precision)
    print(f"  parselmouth n_out={len(truth)}")
    results = []
    for name, fn in VARIANTS.items():
        got = fn(samples, old_rate, new_rate, precision)
        results.append(score(name, got, truth))
    results.sort(key=lambda r: r["mean"])
    print_table(results, "sorted by global mean |diff|")
    boundary_sorted = sorted(results, key=lambda r: max(r["left_max"], r["right_max"]))
    print_table(boundary_sorted, "sorted by boundary max |diff|")
    return results


def main():
    # Test 1: integer ratio, synthetic sine
    sr = 22000
    new_sr = 11000
    t = np.arange(int(sr * 0.5)) / sr
    sine = 0.5 * np.sin(2 * np.pi * 500 * t)
    run_test("Integer 22000 -> 11000, 500 Hz sine, 0.5 s", sine, sr, new_sr, precision=50)

    # Test 2: non-integer, real WAV
    snd = parselmouth.Sound(str(FIXTURE))
    samples = snd.values[0]
    run_test(f"Non-integer 24000 -> 11000, {FIXTURE.name}",
             samples, snd.sampling_frequency, 11000, precision=50)

    # Test 3 (bonus): shorter precision on small signal to stress boundaries
    t2 = np.arange(int(48000 * 0.1)) / 48000
    sine2 = 0.3 * np.sin(2 * np.pi * 1000 * t2) + 0.2 * np.sin(2 * np.pi * 3000 * t2)
    run_test("Integer 48000 -> 16000, 2-tone, 0.1 s, precision=50",
             sine2, 48000, 16000, precision=50)


if __name__ == "__main__":
    sys.exit(main())
