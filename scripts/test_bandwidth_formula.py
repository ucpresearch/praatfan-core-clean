"""
DP22: bandwidth formula blackbox.

Synthesize a signal from a known 2nd-order AR filter whose pole is z = r*e^(iθ).
The formant frequency should be F = θ*sr/(2π) and bandwidth depends on convention:

    K=1:  B = -log(|z|)  * nyquist / π  = -log(r)  * sr / (2π)
    K=2:  B = -log(|z|²) * nyquist / π  = -log(r)  * sr / π     (= 2× K=1)

Our current Python code: `bandwidth = -log(r) * sample_rate / π`
where sample_rate is the resampled rate. That matches K=2.

This script runs Praat's To Formant (burg) on the synth signal and checks
which formula agrees.
"""

from __future__ import annotations

import math
import numpy as np
import parselmouth
from parselmouth.praat import call


def synth_ar2(n_samples: int, r: float, theta: float, sr: float, seed: int = 0) -> np.ndarray:
    """
    Generate a 2nd-order AR process with poles r*exp(±iθ):

        y[n] = 2 r cos(θ) y[n-1] - r² y[n-2] + e[n]

    e[n] ~ N(0, 1). Returns a signal with strong resonance at θ*sr/(2π).
    """
    rng = np.random.default_rng(seed)
    a1 = 2 * r * np.cos(theta)
    a2 = -(r ** 2)
    y = np.zeros(n_samples)
    e = rng.standard_normal(n_samples)
    for n in range(n_samples):
        v = e[n]
        if n >= 1:
            v += a1 * y[n - 1]
        if n >= 2:
            v += a2 * y[n - 2]
        y[n] = v
    # Normalize
    y /= np.max(np.abs(y))
    return y


def main():
    # Use sr = 2*max_formant = 11000 so Praat doesn't resample.
    sr = 11000.0
    max_formant = sr / 2.0  # 5500 Hz
    dur = 2.0
    n = int(dur * sr)

    # Test cases: (r, f_hz) pairs. F2, F3-region resonances.
    cases = [
        (0.95, 500.0),
        (0.95, 1500.0),
        (0.98, 800.0),
        (0.90, 2500.0),
    ]

    print(f"sr={sr}, max_formant={max_formant}")
    print(f"{'r':>6} {'freq':>8} | {'theta':>8} | {'K1 bw':>8} {'K2 bw':>8} | {'praat_F':>8} {'praat_BW':>8}")

    for r, f_hz in cases:
        theta = 2 * np.pi * f_hz / sr
        samples = synth_ar2(n, r, theta, sr)

        snd = parselmouth.Sound(samples, sampling_frequency=sr)
        fmt = call(snd, "To Formant (burg)", 0.005, 5, max_formant, 0.025, 50.0)
        n_frames = call(fmt, "Get number of frames")

        # Query at mid-clip to avoid boundary effects
        t_mid = dur / 2.0

        # Find the formant whose frequency is closest to the expected freq
        best_k = 1
        best_df = math.inf
        for k in (1, 2, 3, 4, 5):
            try:
                f = call(fmt, "Get value at time", k, t_mid, "Hertz", "Linear")
            except Exception:
                continue
            if not math.isnan(f):
                df = abs(f - f_hz)
                if df < best_df:
                    best_df = df
                    best_k = k

        praat_f = call(fmt, "Get value at time", best_k, t_mid, "Hertz", "Linear")
        praat_bw = call(fmt, "Get bandwidth at time", best_k, t_mid, "Hertz", "Linear")

        bw_k1 = -math.log(r) * sr / (2 * math.pi)
        bw_k2 = -math.log(r) * sr / math.pi

        print(f"{r:>6.3f} {f_hz:>8.1f} | {theta:>8.4f} | "
              f"{bw_k1:>8.2f} {bw_k2:>8.2f} | "
              f"{praat_f:>8.2f} {praat_bw:>8.2f}  "
              f"(closer to {'K1' if abs(praat_bw - bw_k1) < abs(praat_bw - bw_k2) else 'K2'})")


if __name__ == "__main__":
    main()
