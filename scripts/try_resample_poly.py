"""
Quick check: does scipy.signal.resample_poly (polyphase windowed-sinc) close
most of the resampler gap vs Praat, without custom code?
"""

from __future__ import annotations

import math
import numpy as np
import parselmouth
from parselmouth.praat import call
from pathlib import Path
from scipy import signal as ss
from scipy.signal import resample_poly
from fractions import Fraction

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
FILE = FIXTURES / "one_two_three_four_five.wav"


def main():
    snd = parselmouth.Sound(str(FILE))
    old_rate = snd.sampling_frequency
    new_rate = 11000.0

    # Praat resample (ground truth)
    snd_r = call(snd, "Resample", new_rate, 50)
    praat_out = snd_r.values[0].copy()
    praat_out_t = np.arange(len(praat_out)) / new_rate

    # Our current: scipy.signal.resample (FFT-based)
    orig = snd.values[0].copy()
    new_length = int(len(orig) * new_rate / old_rate)
    pad_factor = 5
    padded = np.zeros(len(orig) * pad_factor)
    padded[:len(orig)] = orig
    current = ss.resample(padded, new_length * pad_factor)[:new_length]

    # scipy.signal.resample_poly: L/M polyphase
    frac = Fraction(int(new_rate), int(old_rate))
    up = frac.numerator
    down = frac.denominator
    print(f"resample_poly L={up} M={down}")
    # Kaiser window at high precision
    poly_out = resample_poly(orig, up, down, window=('kaiser', 14.0))

    # Align lengths
    n_min = min(len(praat_out), len(current), len(poly_out))
    praat_out = praat_out[:n_min]
    current = current[:n_min]
    poly_out = poly_out[:n_min]

    # Compare to Praat
    def _stats(label, x):
        d = x - praat_out
        print(f"  {label:20s}: mean_abs={np.mean(np.abs(d)):.2e} "
              f"p99={np.percentile(np.abs(d), 99):.2e} "
              f"max={np.max(np.abs(d)):.2e}")

    _stats("current (fft+pad)", current)
    _stats("resample_poly kaiser", poly_out)

    # What's the max value in Praat output, for scale
    print(f"Praat output scale: max|x|={np.max(np.abs(praat_out)):.4f} rms={np.sqrt(np.mean(praat_out**2)):.4f}")


if __name__ == "__main__":
    main()
