"""
Does padding strategy affect the phase gap between scipy and Praat?

Try:
  A. End-padding (current): [signal][zeros]
  B. Centered padding: [zeros][signal][zeros]
  C. Larger end-padding
  D. Pad to power of 2
"""
from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy import signal as ss
from pathlib import Path


FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def resample_padded(x, old_rate, new_rate, pad_before=0, pad_after_factor=5):
    n = len(x)
    total = pad_before + n + max(0, pad_after_factor - 1) * n
    new_length = int(n * new_rate / old_rate)
    padded = np.zeros(total)
    padded[pad_before:pad_before + n] = x
    # Compute new length for the padded array
    new_total = int(total * new_rate / old_rate)
    resampled = ss.resample(padded, new_total)
    # Corresponding segment in output is at [new_pad_before, new_pad_before + new_length]
    new_pad_before = int(pad_before * new_rate / old_rate)
    return resampled[new_pad_before:new_pad_before + new_length]


def resample_pow2(x, old_rate, new_rate):
    """Pad to a power-of-2 length before FFT."""
    n = len(x)
    new_length = int(n * new_rate / old_rate)
    pad_len = 1
    while pad_len < n * 5:
        pad_len *= 2
    padded = np.zeros(pad_len)
    padded[:n] = x
    new_pad_len = int(pad_len * new_rate / old_rate)
    return ss.resample(padded, new_pad_len)[:new_length]


def main():
    snd = parselmouth.Sound(str(FIXTURES / "one_two_three_four_five.wav"))
    orig = snd.values[0].copy()
    old_rate = snd.sampling_frequency
    new_rate = 11000.0

    praat = call(snd, "Resample", new_rate, 50).values[0]
    n_p = len(praat)

    # Compute aligned phase error and per-sample diff
    def eval_strategy(label, out):
        n = min(n_p, len(out))
        diff = out[:n] - praat[:n]
        # Phase
        fp = np.fft.rfft(praat[:n])
        fo = np.fft.rfft(out[:n])
        freqs = np.fft.rfftfreq(n, d=1/new_rate)
        # Consider only bins with significant magnitude
        m = np.abs(fp) > np.percentile(np.abs(fp), 50)
        ph = np.unwrap(np.angle(fo[m] / fp[m]))
        # Linear fit (group delay)
        slope, intercept = np.polyfit(freqs[m], ph, 1)
        tau = -slope / (2 * np.pi)
        resid_phase = ph - (slope * freqs[m] + intercept)
        print(f"  {label:30s}: mean_diff={np.mean(np.abs(diff)):.2e} max={np.max(np.abs(diff)):.2e} "
              f"delay={tau*new_rate:+.4f} samples residPhmax={np.max(np.abs(resid_phase)):.3e}")

    # Baseline
    pad = 5
    padded = np.zeros(len(orig) * pad)
    padded[:len(orig)] = orig
    out = ss.resample(padded, int(len(orig) * new_rate / old_rate) * pad)[:int(len(orig) * new_rate / old_rate)]
    eval_strategy("end-pad 5x (current)", out)

    # No padding
    eval_strategy("no padding", ss.resample(orig, int(len(orig) * new_rate / old_rate)))

    # Centered padding: equal on both sides
    for pad_factor in [2, 3, 5, 10]:
        n = len(orig)
        total = (2 * pad_factor + 1) * n
        padded = np.zeros(total)
        padded[pad_factor * n:pad_factor * n + n] = orig
        new_total = int(total * new_rate / old_rate)
        new_pad_before = int(pad_factor * n * new_rate / old_rate)
        new_length = int(n * new_rate / old_rate)
        r = ss.resample(padded, new_total)
        eval_strategy(f"centered pad {pad_factor}+1+{pad_factor}",
                      r[new_pad_before:new_pad_before + new_length])

    # Pow-2
    eval_strategy("pad to power of 2", resample_pow2(orig, old_rate, new_rate))

    # Edge padding
    for mode in ["edge", "reflect", "symmetric"]:
        n = len(orig)
        padded = np.pad(orig, (2 * n, 2 * n), mode=mode)
        new_total = int(len(padded) * new_rate / old_rate)
        new_pad_before = int(2 * n * new_rate / old_rate)
        new_length = int(n * new_rate / old_rate)
        r = ss.resample(padded, new_total)
        eval_strategy(f"centered {mode}-pad 2x",
                      r[new_pad_before:new_pad_before + new_length])


if __name__ == "__main__":
    main()
