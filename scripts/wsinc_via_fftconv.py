"""
Hypothesis: Praat implements wsinc via FFT convolution, which gives the same
ideal math PLUS FP-noise behavior (needed by Burg for silent-frame stability).

Recipe:
  1. Design FIR kernel: sinc(phi/step) × Hann(phi / N_half)
     with phi ∈ integer grid in input samples
     N_half = (precision + 0.5) * step
  2. FFT-convolve input with kernel (length >> 2 N_half to avoid circular wrap)
  3. For each output sample m, pick at input position x = (m+0.5)*ratio − 0.5
     via nearest-neighbor or linear interpolation of convolved signal

Compare against wsinc (direct), scipy pow-2, and Praat.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy import signal as ss

from praatfan.sound import Sound as PFSound
from praatfan import formant as fmod
from praatfan.formant import _resample_wsinc


FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def wsinc_fftconv(samples, old_rate, new_rate, precision=50):
    """Equivalent of _resample_wsinc but via scipy.signal.fftconvolve."""
    if abs(old_rate - new_rate) < 1e-6:
        return samples.copy()
    n_in = len(samples)
    n_out = int(math.floor(n_in * new_rate / old_rate))
    step = max(old_rate / new_rate, 1.0)
    n_half = (precision + 0.5) * step
    ratio = old_rate / new_rate

    # Build FIR kernel on a DENSE grid (one tap per input sample).
    # Kernel support: ±n_half input samples.
    half_int = int(math.ceil(n_half))
    kernel_len = 2 * half_int + 1
    kernel_idx = np.arange(-half_int, half_int + 1, dtype=np.float64)
    # phi = kernel_idx (kernel at integer offsets from peak)
    sinc_vals = np.sinc(kernel_idx / step)
    window = 0.5 + 0.5 * np.cos(np.pi * kernel_idx / n_half)
    window *= (np.abs(kernel_idx) <= n_half).astype(np.float64)
    kernel = sinc_vals * window / step

    # FFT convolve. Result length = n_in + kernel_len - 1; center-align.
    filtered = ss.fftconvolve(samples, kernel, mode="same")  # same-length output

    # Now decimate: for each output m, we want filtered at x_pos = (m+0.5)*ratio - 0.5.
    # Linear interp between nearest integer positions.
    x_pos = (np.arange(n_out) + 0.5) * ratio - 0.5
    lo = np.floor(x_pos).astype(np.int64)
    hi = lo + 1
    frac = x_pos - lo
    lo_clipped = np.clip(lo, 0, n_in - 1)
    hi_clipped = np.clip(hi, 0, n_in - 1)
    out = filtered[lo_clipped] * (1 - frac) + filtered[hi_clipped] * frac
    return out


def main():
    FILE = FIXTURES / "one_two_three_four_five.wav"
    snd = parselmouth.Sound(str(FILE))
    orig = snd.values[0].copy()
    praat = call(snd, "Resample", 11000.0, 50).values[0]

    direct = _resample_wsinc(orig, 24000.0, 11000.0, precision=50)
    fftc = wsinc_fftconv(orig, 24000.0, 11000.0, precision=50)

    n = min(len(praat), len(direct), len(fftc))
    print(f"=== Match Praat's Resample (mean & max abs diff, n={n}) ===")
    for label, arr in [("direct wsinc", direct), ("fftconv wsinc", fftc)]:
        d = arr[:n] - praat[:n]
        print(f"  {label:16s}: mean={np.mean(np.abs(d)):.3e} max={np.max(np.abs(d)):.3e}")

    # Silence-region test
    print()
    print("=== Silent-region stats (samples 0:400) ===")
    for label, arr in [("direct wsinc", direct), ("fftconv wsinc", fftc), ("praat", praat)]:
        seg = arr[:400]
        print(f"  {label:16s}: rms={np.sqrt(np.mean(seg**2)):.3e} "
              f"Nyquist_bin={np.abs(np.fft.rfft(seg)[-1]):.3e}")

    # Impulse response test
    print()
    print("=== Impulse @ input[1000], 20000 samples ===")
    x_imp = np.zeros(20000); x_imp[1000] = 1.0
    snd_imp = parselmouth.Sound(x_imp, sampling_frequency=24000.0)
    praat_imp = call(snd_imp, "Resample", 11000.0, 50).values[0]
    direct_imp = _resample_wsinc(x_imp, 24000.0, 11000.0, precision=50)
    fftc_imp = wsinc_fftconv(x_imp, 24000.0, 11000.0, precision=50)
    # At d=200 out-of-support distance
    peak_idx = 472  # approximately
    for d in [1, 10, 50, 100, 200, 500, 1000]:
        i = peak_idx + d
        if i < min(len(praat_imp), len(direct_imp), len(fftc_imp)):
            print(f"  d={d:4d}: praat={praat_imp[i]:+.3e}  direct={direct_imp[i]:+.3e}  fftconv={fftc_imp[i]:+.3e}")


if __name__ == "__main__":
    main()
