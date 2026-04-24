"""
FFT-convolution of our explicit Hann-windowed-sinc kernel.

Math-identical to _resample_wsinc when the kernel support equals what direct
wsinc uses. But with FFT convolution we can afford a MUCH longer kernel
(effectively signal-length), so the kernel reaches into the 1/d tail region
where Praat also has non-zero response.

Two implementations tested:
  (1) polyphase via scipy.signal.upfirdn — exact sampling at output positions.
      For rational old/new, works great.
  (2) FFT-convolve the kernel with input at old-rate, then pick samples at
      fractional positions via linear interp. Baseline simple approach.

We focus on (1) because (2) has interpolation error for high-frequency content.
"""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy import signal as ss

from praatfan.formant import _resample_wsinc

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def resample_polyphase(samples, old_rate, new_rate, precision=50):
    """Polyphase (scipy.signal.upfirdn) with our Hann-windowed-sinc kernel."""
    if abs(old_rate - new_rate) < 1e-9:
        return samples.copy()
    # new_rate / old_rate = L / M
    frac = Fraction(int(new_rate), int(old_rate))
    L = frac.numerator
    M = frac.denominator

    # Filter designed at upsampled rate L*old_rate.
    # step (in upsampled domain) is max(L, M) so sinc zero-crossings land on
    # integer multiples of step — equivalent to upsampling-domain Nyquist.
    step = max(L, M)
    half_len = precision * step
    k = np.arange(-half_len, half_len + 1, dtype=np.float64)
    sinc_vals = np.sinc(k / step)
    n_half = (precision + 0.5) * step
    window = np.where(np.abs(k) <= n_half,
                      0.5 + 0.5 * np.cos(np.pi * k / n_half), 0.0)
    h = L * sinc_vals * window / step  # L gain for the zero-insertion upsampling

    y_full = ss.upfirdn(h, samples, up=L, down=M)

    # Align: filter center = half_len in upsampled domain.
    # Effective delay in output samples ≈ half_len / M.
    n_out = int(len(samples) * new_rate / old_rate)
    delay_out = half_len // M
    # Integer trim; residual sub-output-sample offsets handled by kernel center.
    return y_full[delay_out:delay_out + n_out]


def compare_all(label, samples, old_rate, new_rate):
    snd = parselmouth.Sound(samples, sampling_frequency=old_rate)
    praat = call(snd, "Resample", new_rate, 50).values[0]
    direct = _resample_wsinc(samples, old_rate, new_rate, precision=50)
    poly = resample_polyphase(samples, old_rate, new_rate, precision=50)

    n = min(len(praat), len(direct), len(poly))
    praat = praat[:n]
    for lbl, arr in [("direct wsinc", direct[:n]), ("polyphase FFT", poly[:n])]:
        d = arr - praat
        print(f"  {lbl:18s}: mean={np.mean(np.abs(d)):.3e} max={np.max(np.abs(d)):.3e}")


def main():
    # Real audio
    snd = parselmouth.Sound(str(FIXTURES / "one_two_three_four_five.wav"))
    orig = snd.values[0].copy()
    print("Real audio 24k → 11k:")
    compare_all("real", orig, 24000.0, 11000.0)

    # Impulse — check 1/d tail
    x = np.zeros(20000); x[1000] = 1.0
    print("\nImpulse @ input[1000], 24k→11k:")
    praat = call(parselmouth.Sound(x, sampling_frequency=24000.0), "Resample", 11000.0, 50).values[0]
    direct = _resample_wsinc(x, 24000.0, 11000.0, precision=50)
    poly = resample_polyphase(x, 24000.0, 11000.0, precision=50)
    n = min(len(praat), len(direct), len(poly))
    print(f"  direct wsinc: mean={np.mean(np.abs(direct[:n]-praat[:n])):.3e} max={np.max(np.abs(direct[:n]-praat[:n])):.3e}")
    print(f"  polyphase   : mean={np.mean(np.abs(poly[:n]-praat[:n])):.3e} max={np.max(np.abs(poly[:n]-praat[:n])):.3e}")
    # Tail values
    pp = int(np.argmax(np.abs(praat)))
    pd = int(np.argmax(np.abs(direct)))
    py = int(np.argmax(np.abs(poly)))
    print(f"  peaks: praat@{pp}={praat[pp]:+.4f}  direct@{pd}={direct[pd]:+.4f}  poly@{py}={poly[py]:+.4f}")
    print("  tail values:")
    for d in [50, 100, 200, 500, 1000]:
        ip = pp + d
        id_ = pd + d
        iy = py + d
        if max(ip, id_, iy) < n:
            print(f"    d={d:4d}: praat={praat[ip]:+.3e}  direct={direct[id_]:+.3e}  poly={poly[iy]:+.3e}")

    # Silent-region behavior
    print("\nSilent region [0:400]:")
    snd_wav = parselmouth.Sound(str(FIXTURES / "one_two_three_four_five.wav"))
    orig = snd_wav.values[0].copy()
    praat_real = call(snd_wav, "Resample", 11000.0, 50).values[0]
    direct_real = _resample_wsinc(orig, 24000.0, 11000.0, precision=50)
    poly_real = resample_polyphase(orig, 24000.0, 11000.0, precision=50)
    for lbl, arr in [("praat", praat_real), ("direct wsinc", direct_real), ("polyphase", poly_real)]:
        seg = arr[:400]
        print(f"  {lbl:14s}: rms={np.sqrt(np.mean(seg**2)):.3e}  Nyq_bin={np.abs(np.fft.rfft(seg)[-1]):.3e}")


if __name__ == "__main__":
    main()
