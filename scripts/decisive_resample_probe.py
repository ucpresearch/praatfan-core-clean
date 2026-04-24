"""Decisive probe of Praat's Sound: Resample behaviour.

Blackbox-only (via parselmouth). Answers Q1–Q6 in the task brief.
No Praat source code is read.

Usage:
    python scripts/decisive_resample_probe.py

Writes a short textual report to stdout.
"""
from __future__ import annotations

import math
import numpy as np
import parselmouth
from parselmouth.praat import call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sound(samples: np.ndarray, rate: float) -> parselmouth.Sound:
    snd = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=rate)
    return snd


def praat_resample(samples: np.ndarray, old_rate: float, new_rate: float,
                   precision: int = 50) -> np.ndarray:
    snd = make_sound(samples, old_rate)
    out = call(snd, "Resample", new_rate, precision)
    return np.asarray(out.values[0], dtype=np.float64)


def impulse(n: int, pos: int, amp: float = 1.0) -> np.ndarray:
    x = np.zeros(n, dtype=np.float64)
    x[pos] = amp
    return x


# ---------------------------------------------------------------------------
# Q1: Does precision actually change output?
# ---------------------------------------------------------------------------

def q1_precision_effect():
    print("\n=== Q1: precision parameter effect ===")
    N = 4096
    pos = N // 2
    x = impulse(N, pos)
    old_rate = 20000.0
    new_rate = 10000.0
    results = {}
    for p in (1, 2, 3, 5, 10, 50, 100, 1000, 10000):
        y = praat_resample(x, old_rate, new_rate, precision=p)
        results[p] = y
    base = results[50]
    print(f"Reference precision=50 output length: {len(base)}")
    print("precision | max|y| | nonzero count (|y|>1e-12) | max diff vs p=50")
    for p, y in results.items():
        nnz = int(np.sum(np.abs(y) > 1e-12))
        maxabs = float(np.max(np.abs(y)))
        diff = float(np.max(np.abs(y - base))) if len(y) == len(base) else float("nan")
        print(f"  {p:>7d} | {maxabs:.4e} | {nnz:>6d} | {diff:.3e}")
    # Measure "support" at each precision: first index where kernel goes to 0
    # Use p=1 vs p=50 difference to tell if precision is honoured.
    p1, p50 = results[1], results[50]
    if np.allclose(p1, p50, atol=1e-13):
        print("-> precision IGNORED (p=1 identical to p=50)")
    else:
        print(f"-> precision HONOURED. p=1 diff from p=50: max={np.max(np.abs(p1 - p50)):.3e}")


# ---------------------------------------------------------------------------
# Q2: Position invariance (shift-invariant interior kernel?)
# ---------------------------------------------------------------------------

def q2_position_invariance():
    print("\n=== Q2: Position invariance of kernel (interior) ===")
    N = 8192
    old_rate = 20000.0
    new_rate = 10000.0  # integer 2:1
    positions = [500, 1000, 2000, 4000, 6000]
    # The output position corresponding to input position p using Praat's
    # 0.5-centered convention is q = (p + 0.5) * (new/old) - 0.5
    ratio = new_rate / old_rate
    kernels = []
    for p in positions:
        y = praat_resample(impulse(N, p), old_rate, new_rate)
        # Find peak location in output
        qmax = int(np.argmax(np.abs(y)))
        # Extract a window around the peak
        W = 40
        lo = max(0, qmax - W)
        hi = min(len(y), qmax + W + 1)
        seg = y[lo:hi]
        # Align by peak index
        aligned = np.zeros(2 * W + 1)
        peak_in_seg = qmax - lo
        start = W - peak_in_seg
        aligned[start:start + len(seg)] = seg
        kernels.append((p, qmax, aligned))
    ref = kernels[len(kernels) // 2][2]
    print("pos | expected q | peak q | max|y-ref| after peak-align")
    for p, qmax, aligned in kernels:
        q_exp = (p + 0.5) * ratio - 0.5
        diff = float(np.max(np.abs(aligned - ref)))
        print(f"  {p:>5d} | {q_exp:>9.2f} | {qmax:>5d} | {diff:.3e}")


# ---------------------------------------------------------------------------
# Q3: Bit-exactness across ratios
# ---------------------------------------------------------------------------

def q3_bit_exactness():
    print("\n=== Q3: Bit-exactness across ratios ===")
    # Use a known bandlimited sine well inside both Nyquists
    old_rate = 24000.0
    N = 4096
    t = np.arange(N) / old_rate
    freq = 200.0  # well below any Nyquist we'll test
    x = np.sin(2 * np.pi * freq * t)

    ratios = {
        "1x (identity)": old_rate,
        "2x down": old_rate / 2,   # 12000
        "3x down": old_rate / 3,   # 8000
        "4x down": old_rate / 4,   # 6000
        "2x up":   old_rate * 2,   # 48000
        "3x up":   old_rate * 3,   # 72000
        "3/2 down (16000)": 16000.0,
        "5/4 down (19200)": 19200.0,
        "non-integer 22050": 22050.0,
        "weird 7637":        7637.0,
    }
    print(f"{'ratio':<22} | new_rate | n_out | mean |y-ref| | max |y-ref|")
    for name, new_rate in ratios.items():
        y = praat_resample(x, old_rate, new_rate)
        # Build reference on the Praat 0.5-centered output grid.
        # Output sample m is at time (m + 0.5) / new_rate in the input domain
        # *if* we align output edges to input edges (common convention):
        #   ref = sin(2 pi f * (m + 0.5) / new_rate  -  0.5/old_rate)   ???
        # Try the simplest: t_out = (m + 0.5) / new_rate - 0.5 / old_rate
        # equivalent to x_in = (m + 0.5) * old_rate/new_rate - 0.5, in sample units
        m = np.arange(len(y))
        x_in_samples = (m + 0.5) * (old_rate / new_rate) - 0.5
        t_out = x_in_samples / old_rate
        ref = np.sin(2 * np.pi * freq * t_out)
        # Ignore edge 5% on each side where sinc truncation hurts
        edge = max(10, len(y) // 20)
        core = slice(edge, len(y) - edge)
        mean_err = float(np.mean(np.abs(y[core] - ref[core])))
        max_err = float(np.max(np.abs(y[core] - ref[core])))
        print(f"  {name:<20} | {new_rate:>7.0f} | {len(y):>5d} | {mean_err:.3e} | {max_err:.3e}")


# ---------------------------------------------------------------------------
# Q4: Symmetry of kernel
# ---------------------------------------------------------------------------

def q4_symmetry():
    print("\n=== Q4: Kernel symmetry ===")
    N = 8192
    old_rate, new_rate = 20000.0, 10000.0
    p = 4000
    y = praat_resample(impulse(N, p), old_rate, new_rate)
    q0 = int(np.argmax(np.abs(y)))
    W = 30
    left = y[q0 - W:q0]
    right = y[q0 + 1:q0 + 1 + W]
    # Reverse left to align
    left_rev = left[::-1]
    diff = np.abs(left_rev - right)
    print(f"peak q={q0}, peak value={y[q0]:.6e}")
    print(f"max |left_rev - right| over {W} samples: {np.max(diff):.3e}")
    print(f"mean |left_rev - right|: {np.mean(diff):.3e}")
    # Also check integer-ratio case where impulse falls ON an output grid point
    # p = 4000, ratio 2:1 -> output index at x_in=p means q such that
    # (q+0.5)*(old/new) - 0.5 = p   =>  q = p*new/old  - 0.5 + 0.5*new/old
    # For 2:1 with half-sample offset this may not hit exactly.
    if np.max(diff) < 1e-12:
        print("-> kernel SYMMETRIC")
    else:
        print("-> kernel NOT bit-exact symmetric (may be due to fractional peak position)")


# ---------------------------------------------------------------------------
# Q5: Kernel shape fitting
# ---------------------------------------------------------------------------

def q5_kernel_shape():
    print("\n=== Q5: Kernel shape (impulse response envelope fitting) ===")
    N = 16384
    old_rate, new_rate = 20000.0, 10000.0   # 2:1
    p = N // 2
    y = praat_resample(impulse(N, p), old_rate, new_rate)
    # Extract kernel in window around peak
    q0 = int(np.argmax(np.abs(y)))
    # Find zero crossings
    W = 200
    lo = max(0, q0 - W)
    hi = min(len(y), q0 + W + 1)
    seg = y[lo:hi]
    idx_peak = q0 - lo

    zero_crossings = []
    for i in range(idx_peak + 1, len(seg) - 1):
        if seg[i] == 0.0 or (seg[i] * seg[i + 1] < 0):
            zero_crossings.append(i)
        if len(zero_crossings) >= 25:
            break
    print(f"First zero crossings (samples after peak): "
          f"{[zc - idx_peak for zc in zero_crossings[:10]]}")

    # Extrema between zero crossings -> envelope samples
    extrema_x = []
    extrema_y = []
    prev_zc = idx_peak
    for zc in zero_crossings:
        sub = seg[prev_zc:zc + 1]
        if len(sub) == 0:
            continue
        k = int(np.argmax(np.abs(sub)))
        extrema_x.append((prev_zc + k) - idx_peak)
        extrema_y.append(abs(seg[prev_zc + k]))
        prev_zc = zc
    extrema_x = np.array(extrema_x, dtype=np.float64)
    extrema_y = np.array(extrema_y, dtype=np.float64)
    # Drop the peak itself at x=0
    mask = extrema_x > 0
    ex = extrema_x[mask]
    ey = extrema_y[mask]
    # Normalize
    peak = abs(seg[idx_peak])
    ey_norm = ey / peak
    print(f"peak={peak:.6e}")
    print("Extrema (sample-offset | abs value | normalized):")
    for xv, yv, yn in zip(ex[:15], ey[:15], ey_norm[:15]):
        print(f"  {xv:>5.0f} | {yv:.4e} | {yn:.4e}")

    if len(ex) < 5:
        print("Not enough extrema to fit envelope.")
        return

    # Fit candidates. Let N_half = support half-width (last zero crossing)
    N_half = ex[-1] + 1.0
    # Hann: env(x) = 0.5 + 0.5 * cos(pi * x / N_half)  for x in [0, N_half]
    hann = 0.5 + 0.5 * np.cos(np.pi * ex / N_half)
    # Hamming-ish: 0.54 + 0.46 cos
    hamming = 0.54 + 0.46 * np.cos(np.pi * ex / N_half)
    # Triangular
    tri = np.maximum(0.0, 1.0 - ex / N_half)
    # Gaussian (fit sigma by least squares)
    # ey_norm ~ exp(-0.5*(x/sigma)^2). Take log where ey_norm>0.
    pos = ey_norm > 1e-6
    if pos.sum() > 3:
        # log y = -0.5 x^2 / s^2
        A = ex[pos] ** 2
        B = -2.0 * np.log(ey_norm[pos])
        sigma2 = float(np.sum(A * A) / np.sum(A * B)) if np.sum(A * B) != 0 else np.nan
        sigma = math.sqrt(sigma2) if sigma2 > 0 else float("nan")
    else:
        sigma = float("nan")
    gauss = np.exp(-0.5 * (ex / sigma) ** 2) if sigma == sigma else np.zeros_like(ex)
    # Kaiser envelope approximation: I0(beta*sqrt(1-(x/N)^2))/I0(beta); fit beta
    from numpy import i0
    best_beta = None
    best_chi = float("inf")
    for beta in np.linspace(2.0, 20.0, 37):
        t = 1.0 - (ex / N_half) ** 2
        t = np.clip(t, 0.0, None)
        k = i0(beta * np.sqrt(t)) / i0(beta)
        chi = float(np.sum((ey_norm - k) ** 2))
        if chi < best_chi:
            best_chi = chi
            best_beta = beta
    kaiser = i0(best_beta * np.sqrt(np.clip(1 - (ex / N_half) ** 2, 0, None))) / i0(best_beta)

    def chi(pred):
        return float(np.sum((ey_norm - pred) ** 2))

    print(f"N_half (last extremum offset + 1): {N_half}")
    print(f"chi² for candidate windows (smaller is better):")
    print(f"  Hann          : {chi(hann):.4e}")
    print(f"  Hamming       : {chi(hamming):.4e}")
    print(f"  Triangular    : {chi(tri):.4e}")
    print(f"  Gaussian σ={sigma:.2f} : {chi(gauss):.4e}")
    print(f"  Kaiser β={best_beta:.2f}: {best_chi:.4e}")


# ---------------------------------------------------------------------------
# Q6: Input-length dependence
# ---------------------------------------------------------------------------

def q6_length_dependence():
    print("\n=== Q6: Does input length change the kernel? ===")
    old_rate, new_rate = 20000.0, 10000.0
    lengths = [1024, 2048, 5000, 10000, 39168]
    first_kernel = None
    first_len = None
    for N in lengths:
        p = N // 2
        y = praat_resample(impulse(N, p), old_rate, new_rate)
        q0 = int(np.argmax(np.abs(y)))
        W = 100
        lo = max(0, q0 - W)
        hi = min(len(y), q0 + W + 1)
        seg = y[lo:hi].copy()
        # Align by peak index
        aligned = np.zeros(2 * W + 1)
        peak_in_seg = q0 - lo
        start = W - peak_in_seg
        aligned[start:start + len(seg)] = seg
        if first_kernel is None:
            first_kernel = aligned
            first_len = N
            print(f"Reference N={N}: peak={aligned[W]:.6e}")
        else:
            diff = float(np.max(np.abs(aligned - first_kernel)))
            print(f"N={N:>6d}: max diff vs N={first_len}: {diff:.3e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    q1_precision_effect()
    q2_position_invariance()
    q3_bit_exactness()
    q4_symmetry()
    q5_kernel_shape()
    q6_length_dependence()
