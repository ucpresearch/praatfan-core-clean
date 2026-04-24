"""
Timing-hypothesis sweep for windowed-sinc resampler.

Tests alternative x_pos(m) mappings against parselmouth's `Sound: Resample`
output on several signals (integer down/up, non-integer real audio, short
signals, DC). Prints mean/max abs-diff, FFT-phase delay, and first/last 5
samples for each variant.

Run:
    source /home/urielc/local/scr/venvs/praatfan-core-clean/bin/activate
    python scripts/timing_hypotheses.py
"""
from __future__ import annotations

import math
import sys
from math import gcd
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import parselmouth  # noqa: E402
from parselmouth.praat import call  # noqa: E402


# --------------------------------------------------------------------------
# Pluggable windowed-sinc resampler: x_pos(m) is the hypothesis under test.
# --------------------------------------------------------------------------
def resample_with_xpos(samples, old_rate, new_rate, x_pos_fn, precision=50):
    if abs(old_rate - new_rate) < 1e-6:
        return samples.copy()
    n_in = len(samples)
    n_out = int(math.floor(n_in * new_rate / old_rate))
    ratio = old_rate / new_rate
    step = max(ratio, 1.0)
    depth = precision * step
    inv_step = 1.0 / step
    inv_depth = 1.0 / depth

    out = np.empty(n_out, dtype=np.float64)
    for m in range(n_out):
        x = x_pos_fn(m, ratio, old_rate, new_rate)
        low = max(0, int(math.ceil(x - depth)))
        high = min(n_in - 1, int(math.floor(x + depth)))
        if high < low:
            out[m] = 0.0
            continue
        k = np.arange(low, high + 1, dtype=np.float64)
        phi = k - x
        s = np.sinc(phi * inv_step)
        w = 0.5 + 0.5 * np.cos(np.pi * phi * inv_depth)
        out[m] = float(np.dot(samples[low:high + 1], s * w)) * inv_step
    return out


# --------------------------------------------------------------------------
# Hypotheses. Each is a callable (m, ratio, old_rate, new_rate) -> x.
# --------------------------------------------------------------------------
def xpos_a_int_aligned(m, r, o, n):
    return m * r


def xpos_b_half_centered(m, r, o, n):
    return (m + 0.5) * r - 0.5


def xpos_c_center_first_step(m, r, o, n):
    return m * r + (r - 1) / 2


def xpos_d_empirical(m, r, o, n):
    return m * r + 0.5 - 0.5 / r


def xpos_e_rational_grid(m, r, o, n):
    # (m+0.5)*r - 0.5 snapped to nearest 1/lcm(old,new) grid on time axis.
    # Time-axis grid: x (input samples) = t * old_rate; t step = 1/lcm.
    o_i, n_i = int(round(o)), int(round(n))
    L = o_i * n_i // gcd(o_i, n_i)          # lcm
    x = (m + 0.5) * r - 0.5
    # grid spacing in input-sample units = old_rate / L
    g = o / L
    return round(x / g) * g


def xpos_f_time_domain(m, r, o, n):
    # Treat output sample m as sitting at time (m+0.5)/new_rate, map back
    # to input-sample index via t*old_rate - 0.5.
    return (m + 0.5) / n * o - 0.5


def xpos_g_plus_half(m, r, o, n):
    # Own hypothesis: Praat's time t_out(m) = t1_out + m*dt_out where
    # t1 = 0.5*dt. Map via input t1 = 0.5*dt_in. So x = m*r + 0.5*(r-1).
    # (Equivalent to (c) — kept as sanity; use a distinct alt here.)
    # Alt: first-output at x=0 exactly, step = r (no centering offset).
    # This one is "t1_out maps to input sample 0".
    return m * r


VARIANTS = [
    ("a_int_aligned",         xpos_a_int_aligned),
    ("b_half_centered",       xpos_b_half_centered),
    ("c_center_first_step",   xpos_c_center_first_step),
    ("d_empirical",           xpos_d_empirical),
    ("e_rational_grid",       xpos_e_rational_grid),
    ("f_time_domain",         xpos_f_time_domain),
    ("g_zero_first",          xpos_g_plus_half),
]


# --------------------------------------------------------------------------
# FFT-phase-fit delay estimate.
# --------------------------------------------------------------------------
def fft_phase_delay(ours, praat):
    n = min(len(ours), len(praat))
    if n < 32:
        return float("nan")
    a = ours[:n].astype(np.float64)
    b = praat[:n].astype(np.float64)
    if np.max(np.abs(b)) < 1e-12:
        return float("nan")
    fa = np.fft.rfft(a)
    fb = np.fft.rfft(b)
    freqs = np.fft.rfftfreq(n, d=1.0)
    mag = np.abs(fb)
    thresh = np.percentile(mag, 50)
    mask = (mag > thresh) & (mag > 1e-10)
    if mask.sum() < 5:
        return float("nan")
    ratio = fa[mask] / fb[mask]
    phase = np.unwrap(np.angle(ratio))
    slope = np.polyfit(freqs[mask], phase, 1)[0]
    return -slope / (2 * np.pi)


# --------------------------------------------------------------------------
# Ground truth via parselmouth.
# --------------------------------------------------------------------------
def praat_resample(samples, old_rate, new_rate, precision=50):
    snd = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=old_rate)
    out = call(snd, "Resample", new_rate, precision)
    return np.asarray(out.values[0], dtype=np.float64)


# --------------------------------------------------------------------------
# Test cases.
# --------------------------------------------------------------------------
def case_integer_downsample():
    sr = 22000
    t = np.arange(int(0.1 * sr)) / sr
    x = np.sin(2 * np.pi * 500 * t)
    return "int_down_22k_to_11k_500Hz", x, sr, 11000


def case_integer_upsample():
    sr = 11000
    t = np.arange(int(0.1 * sr)) / sr
    x = np.sin(2 * np.pi * 500 * t)
    return "int_up_11k_to_22k_500Hz", x, sr, 22000


def case_noninteger_real():
    import soundfile as sf
    path = REPO / "tests" / "fixtures" / "one_two_three_four_five.wav"
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        data = data[:, 0]
    # Trim to ~0.5s so this doesn't take forever
    data = data[: int(0.5 * sr)]
    if abs(sr - 24000) > 1:
        # resample via parselmouth to 24000 so input rate is exactly 24000
        snd = parselmouth.Sound(data.astype(np.float64), sampling_frequency=float(sr))
        snd2 = call(snd, "Resample", 24000, 50)
        data = np.asarray(snd2.values[0], dtype=np.float64)
        sr = 24000
    return f"real_{sr}_to_11000", data, sr, 11000


def case_very_short():
    sr = 24000
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100) * 0.1
    return "short_N100_24k_to_11k", x, sr, 11000


def case_dc():
    sr = 24000
    x = np.ones(int(0.05 * sr))
    return "dc_24k_to_11k", x, sr, 11000


CASES = [
    case_integer_downsample,
    case_integer_upsample,
    case_noninteger_real,
    case_very_short,
    case_dc,
]


# --------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------
def run():
    results = {}
    for case_fn in CASES:
        name, x, old_rate, new_rate = case_fn()
        praat = praat_resample(x, old_rate, new_rate)
        print(f"\n===== {name}  (old={old_rate}, new={new_rate}, "
              f"N_in={len(x)}, N_out={len(praat)}) =====")
        print(f"  parselmouth first5: {praat[:5]}")
        print(f"  parselmouth last5:  {praat[-5:]}")
        rows = []
        for vname, fn in VARIANTS:
            ours = resample_with_xpos(x, old_rate, new_rate, fn)
            n = min(len(ours), len(praat))
            diff = np.abs(ours[:n] - praat[:n])
            mean_d = float(diff.mean())
            max_d = float(diff.max())
            delay = fft_phase_delay(ours, praat)
            rows.append((vname, mean_d, max_d, delay, ours))
        rows.sort(key=lambda r: r[1])
        print(f"  {'variant':<24} {'mean':>12} {'max':>12} {'delay_samp':>12}")
        for vname, mean_d, max_d, delay, _ in rows:
            print(f"  {vname:<24} {mean_d:12.3e} {max_d:12.3e} {delay:12.4f}")
        # Show first/last 5 of top-3 variants
        print("  -- first5 / last5 (top 3 by mean) --")
        for vname, _, _, _, ours in rows[:3]:
            print(f"  {vname:<24} first5: {ours[:5]}")
            print(f"  {'':<24} last5:  {ours[-5:]}")
        results[name] = rows
    return results


if __name__ == "__main__":
    run()
