"""
Kernel-shape hypothesis testing for Praat's Sound: Resample.

Blackbox enumeration over window shape, sinc-argument scaling, precision
interpretation, and normalization. For each variant we compare against
parselmouth's Resample ground truth on four test signals.

Usage: python scripts/kernel_hypotheses.py
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Tuple

import numpy as np
import parselmouth
from parselmouth.praat import call
import soundfile as sf


# ---------------------------------------------------------------------------
# Kernel variants
# ---------------------------------------------------------------------------

WindowFn = Callable[[np.ndarray, float, float, int], np.ndarray]
# w(phi, depth, step, precision) -> window values


def win_hann_depth(phi, depth, step, precision):
    # (a) Hann over +/- depth (baseline)
    return 0.5 + 0.5 * np.cos(np.pi * phi / depth)


def win_half_hann(phi, depth, step, precision):
    # (b) quarter-cosine: cos(pi*phi/(2*depth)) - never reaches 0 at edges
    return 0.5 + 0.5 * np.cos(np.pi * phi / (2.0 * depth))


def win_hann_prec_plus1(phi, depth, step, precision):
    # (c) precision+1
    return 0.5 + 0.5 * np.cos(np.pi * phi / ((precision + 1) * step))


def win_hann_prec_minus1(phi, depth, step, precision):
    # (c) precision-1
    d = max(precision - 1, 1) * step
    return 0.5 + 0.5 * np.cos(np.pi * phi / d)


def win_hann_prec_plus_half(phi, depth, step, precision):
    # (f) precision + 0.5
    return 0.5 + 0.5 * np.cos(np.pi * phi / ((precision + 0.5) * step))


def win_triangular(phi, depth, step, precision):
    # (d) triangular / Bartlett
    return np.maximum(0.0, 1.0 - np.abs(phi) / depth)


def win_blackman(phi, depth, step, precision):
    # (e) Blackman-ish
    return 0.42 + 0.5 * np.cos(np.pi * phi / depth) + 0.08 * np.cos(2 * np.pi * phi / depth)


def win_rectangular(phi, depth, step, precision):
    return np.ones_like(phi)


WINDOWS: Dict[str, WindowFn] = {
    "hann_depth": win_hann_depth,
    "half_hann": win_half_hann,
    "hann_prec+1": win_hann_prec_plus1,
    "hann_prec-1": win_hann_prec_minus1,
    "hann_prec+0.5": win_hann_prec_plus_half,
    "triangular": win_triangular,
    "blackman": win_blackman,
    "rect": win_rectangular,
}


SincFn = Callable[[np.ndarray, float, float, float], np.ndarray]
# s(phi, step, old_rate, new_rate) -> sinc values


def sinc_over_step(phi, step, old_rate, new_rate):
    # (i) sinc(phi / step)
    return np.sinc(phi / step)


def sinc_direct(phi, step, old_rate, new_rate):
    # (ii) sinc(phi)
    return np.sinc(phi)


def sinc_ratio(phi, step, old_rate, new_rate):
    # (iii) sinc(phi * new_rate / old_rate)
    return np.sinc(phi * new_rate / old_rate)


def sinc_adaptive(phi, step, old_rate, new_rate):
    # (iv) sinc(phi) upsample, sinc(phi/step) downsample
    if new_rate >= old_rate:
        return np.sinc(phi)
    return np.sinc(phi / step)


SINCS: Dict[str, SincFn] = {
    "sinc/step": sinc_over_step,
    "sinc(phi)": sinc_direct,
    "sinc*ratio": sinc_ratio,
    "sinc_adaptive": sinc_adaptive,
}


# Precision interpretation: depth function
PrecisionFn = Callable[[int, float], float]


def prec_lobes(precision, step):
    # (alpha) depth = precision * step (current)
    return precision * step


def prec_in_samples(precision, step):
    # (beta) depth = precision (input samples)
    return float(precision)


def prec_out_samples(precision, step):
    # (gamma) depth = precision / step * step? actually precision output samples
    # output sample spacing in input domain is step; so window reaches out
    # precision output samples = precision / (new/old) = precision * old/new = precision * step input samples
    # Wait: step = old/new when downsampling; then "precision output samples" in input index = precision * step
    # That's identical to lobes. For upsample step=1 either way. Skip: redundant. Use precision/step as a distinct test
    return precision / max(step, 1.0) * step if step > 1 else precision


PRECS: Dict[str, PrecisionFn] = {
    "alpha (prec*step)": prec_lobes,
    "beta (prec samples)": prec_in_samples,
}


# Normalization
NormFn = Callable[[np.ndarray, np.ndarray, float], float]
# norm(kernel, product_with_samples_result=None, step) -> divisor


NORMS: Dict[str, str] = {
    "div_step": "div_step",
    "div_kernel_sum": "div_kernel_sum",
    "no_norm": "no_norm",
    "div_sqrt_step": "div_sqrt_step",
}


# ---------------------------------------------------------------------------
# Core resampler
# ---------------------------------------------------------------------------


def wsinc_resample(
    samples: np.ndarray,
    old_rate: float,
    new_rate: float,
    precision: int = 50,
    *,
    window: str = "hann_depth",
    sinc: str = "sinc/step",
    prec_interp: str = "alpha (prec*step)",
    norm: str = "div_step",
) -> np.ndarray:
    if abs(old_rate - new_rate) < 1e-6:
        return samples.copy()

    n_in = len(samples)
    n_out = int(math.floor(n_in * new_rate / old_rate))
    step = max(old_rate / new_rate, 1.0)
    depth = PRECS[prec_interp](precision, step)
    ratio = old_rate / new_rate

    win_fn = WINDOWS[window]
    sinc_fn = SINCS[sinc]

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
        s = sinc_fn(phi, step, old_rate, new_rate)
        w = win_fn(phi, depth, step, precision)
        kernel = s * w
        acc = float(np.dot(samples[low:high + 1], kernel))

        if norm == "div_step":
            out[m] = acc / step
        elif norm == "div_kernel_sum":
            ks = float(np.sum(kernel))
            out[m] = acc / ks if abs(ks) > 1e-20 else 0.0
        elif norm == "no_norm":
            out[m] = acc
        elif norm == "div_sqrt_step":
            out[m] = acc / math.sqrt(step)
        else:
            raise ValueError(norm)
    return out


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def parselmouth_resample(samples: np.ndarray, old_rate: float, new_rate: float,
                         precision: int = 50) -> np.ndarray:
    snd = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=old_rate)
    out = call(snd, "Resample", float(new_rate), precision)
    return np.asarray(out.values[0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def build_test_cases() -> List[Tuple[str, np.ndarray, float, float]]:
    cases = []

    # 1. DC signal
    dc = np.ones(24000, dtype=np.float64)
    cases.append(("dc_24k_to_11k", dc, 24000.0, 11000.0))

    # 2. Impulse at sample 1000
    impulse = np.zeros(24000, dtype=np.float64)
    impulse[1000] = 1.0
    cases.append(("impulse_24k_to_11k", impulse, 24000.0, 11000.0))

    # 3. 500Hz sine at 22000 -> 11000 (integer 2x)
    t = np.arange(22000) / 22000.0
    sine = np.sin(2 * np.pi * 500.0 * t)
    cases.append(("sine500_22k_to_11k", sine, 22000.0, 11000.0))

    # 4. Real file
    fixture = "/home/urielc/local/decfiles/private/Dev/git/praatfan-core-clean/tests/fixtures/one_two_three_four_five.wav"
    if os.path.exists(fixture):
        data, sr = sf.read(fixture)
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(np.float64)
        # Resample from its native rate to 11000. Keep old_rate = sr.
        # Task says 24000 -> 11000. If not 24000, resample with scipy first or accept native.
        if abs(sr - 24000) < 1:
            cases.append((f"file_{sr:.0f}_to_11k", data, float(sr), 11000.0))
        else:
            # Use its native rate; still a valid downsample test
            cases.append((f"file_{int(sr)}_to_11k", data, float(sr), 11000.0))

    return cases


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class Result:
    test: str
    window: str
    sinc: str
    prec: str
    norm: str
    mean_abs: float
    max_abs: float


def evaluate(variant, test_name, samples, old_rate, new_rate, truth) -> Result:
    out = wsinc_resample(samples, old_rate, new_rate, precision=50, **variant)
    n = min(len(out), len(truth))
    diff = out[:n] - truth[:n]
    return Result(
        test=test_name,
        window=variant["window"],
        sinc=variant["sinc"],
        prec=variant["prec_interp"],
        norm=variant["norm"],
        mean_abs=float(np.mean(np.abs(diff))),
        max_abs=float(np.max(np.abs(diff))),
    )


def main():
    cases = build_test_cases()
    print(f"# Test cases: {len(cases)}")

    # Pre-compute ground truth
    truths = {}
    for name, samples, old_rate, new_rate in cases:
        print(f"  truth: {name} ({len(samples)} @ {old_rate} -> {new_rate})")
        truths[name] = parselmouth_resample(samples, old_rate, new_rate)

    # Phase 1: main sweep (all windows x all sincs, fixed prec=alpha, norm=div_step)
    print("\n## Phase 1: window x sinc (prec=alpha, norm=div_step)")
    print(f"{'test':25s} {'window':16s} {'sinc':16s} {'mean_abs':>12s} {'max_abs':>12s}")
    phase1_results = []
    for window, sinc_name in product(WINDOWS.keys(), SINCS.keys()):
        variant = dict(window=window, sinc=sinc_name,
                       prec_interp="alpha (prec*step)", norm="div_step")
        for name, samples, old_rate, new_rate in cases:
            r = evaluate(variant, name, samples, old_rate, new_rate, truths[name])
            phase1_results.append(r)
            print(f"{name:25s} {window:16s} {sinc_name:16s} {r.mean_abs:12.3e} {r.max_abs:12.3e}")

    # Phase 2: normalization sweep on best-so-far window/sinc combo
    # Find best per-test and then across-tests
    print("\n## Phase 2: normalization sweep (window=hann_depth, sinc/step)")
    for norm in NORMS:
        variant = dict(window="hann_depth", sinc="sinc/step",
                       prec_interp="alpha (prec*step)", norm=norm)
        for name, samples, old_rate, new_rate in cases:
            r = evaluate(variant, name, samples, old_rate, new_rate, truths[name])
            print(f"{name:25s} norm={norm:20s} mean={r.mean_abs:12.3e} max={r.max_abs:12.3e}")

    # Phase 3: precision interpretation
    print("\n## Phase 3: precision interpretation (hann_depth, sinc/step, div_step)")
    for prec in PRECS:
        variant = dict(window="hann_depth", sinc="sinc/step",
                       prec_interp=prec, norm="div_step")
        for name, samples, old_rate, new_rate in cases:
            r = evaluate(variant, name, samples, old_rate, new_rate, truths[name])
            print(f"{name:25s} prec={prec:20s} mean={r.mean_abs:12.3e} max={r.max_abs:12.3e}")

    # Phase 4: rank overall combos by mean across non-DC tests
    print("\n## Phase 4: best overall (avg mean_abs across tests)")
    # build aggregate
    agg: Dict[Tuple[str, str, str, str], List[float]] = {}
    all_variants = []
    for window, sinc_name, prec, norm in product(WINDOWS.keys(), SINCS.keys(),
                                                  PRECS.keys(), NORMS.keys()):
        key = (window, sinc_name, prec, norm)
        all_variants.append(key)

    for key in all_variants:
        window, sinc_name, prec, norm = key
        variant = dict(window=window, sinc=sinc_name, prec_interp=prec, norm=norm)
        errs = []
        for name, samples, old_rate, new_rate in cases:
            r = evaluate(variant, name, samples, old_rate, new_rate, truths[name])
            errs.append(r.mean_abs)
        agg[key] = errs

    avg_scored = [(key, float(np.mean(v))) for key, v in agg.items()]
    avg_scored.sort(key=lambda kv: kv[1])

    print(f"\n{'rank':>4s} {'window':16s} {'sinc':16s} {'prec':22s} {'norm':16s} {'avg_mean':>12s}")
    for rank, (key, score) in enumerate(avg_scored[:15], 1):
        w, s, p, n = key
        print(f"{rank:4d} {w:16s} {s:16s} {p:22s} {n:16s} {score:12.3e}")

    # Flag any near bit-exact
    print("\n## Near bit-exact hits (< 1e-10 on any test)")
    near_exact = []
    for key, errs in agg.items():
        for (name, _, _, _), err in zip(cases, errs):
            if err < 1e-10:
                near_exact.append((key, name, err))
    if near_exact:
        for key, name, err in near_exact:
            print(f"  {key} on {name}: mean_abs={err:.3e}")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
