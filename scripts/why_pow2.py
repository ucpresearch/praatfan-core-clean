"""
Why does pow-2 padding beat 5x linear padding for Burg parity?

Inspect what differs between the two outputs:
- Sample-level diffs: where are they concentrated?
- Spectral diffs: which frequency bands?
- Does pow-2 output match Praat more closely everywhere? Or just in some
  specific frequency band that Burg cares about?
"""
from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy import signal as ss
from pathlib import Path


FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def main():
    snd = parselmouth.Sound(str(FIXTURES / "one_two_three_four_five.wav"))
    orig = snd.values[0].copy()
    old_rate = snd.sampling_frequency
    new_rate = 11000.0
    n_in = len(orig)
    n_out = int(n_in * new_rate / old_rate)

    praat = call(snd, "Resample", new_rate, 50).values[0]
    n_p = len(praat)

    # Variant 1: 5x linear pad (integer rate output, 195840 -> 89760)
    pad1 = 5 * n_in
    padded1 = np.zeros(pad1); padded1[:n_in] = orig
    new_pad1 = int(pad1 * new_rate / old_rate)
    y_5x = ss.resample(padded1, new_pad1)[:n_out]

    # Variant 2: pow-2 pad (131072 -> 60074, slight rate drift)
    pad2 = 1
    while pad2 < n_in * 2: pad2 *= 2
    padded2 = np.zeros(pad2); padded2[:n_in] = orig
    new_pad2 = int(pad2 * new_rate / old_rate)
    y_pow2 = ss.resample(padded2, new_pad2)[:n_out]

    n = min(n_p, n_out, len(y_5x), len(y_pow2))
    print(f"n_in={n_in}, n_out={n_out}")
    print(f"5x linear pad: pad_len={pad1} ({pad1}={pad1//24000}×24000), new_pad_len={new_pad1}")
    print(f"pow-2 pad:     pad_len={pad2} (2^{int(np.log2(pad2))}), new_pad_len={new_pad2}")
    print(f"  5x linear ratio check: {pad1 * new_rate / old_rate} (int? {(pad1*new_rate/old_rate)%1==0})")
    print(f"  pow-2 ratio check:     {pad2 * new_rate / old_rate} (int? {(pad2*new_rate/old_rate)%1==0})")

    # Sample-level diffs
    d5x = y_5x[:n] - praat[:n]
    dp2 = y_pow2[:n] - praat[:n]
    dboth = y_5x[:n] - y_pow2[:n]
    print()
    print(f"5x   vs praat: mean_abs={np.mean(np.abs(d5x)):.3e} max={np.max(np.abs(d5x)):.3e}")
    print(f"pow2 vs praat: mean_abs={np.mean(np.abs(dp2)):.3e} max={np.max(np.abs(dp2)):.3e}")
    print(f"5x   vs pow2:  mean_abs={np.mean(np.abs(dboth)):.3e} max={np.max(np.abs(dboth)):.3e}")

    # Where are the diffs concentrated?
    print()
    print(f"5x vs pow2 — worst 10 sample locations:")
    idx = np.argsort(-np.abs(dboth))[:10]
    for i in idx:
        print(f"  idx={i:5d} t={i/new_rate:.4f}  5x={y_5x[i]:+.5f} pow2={y_pow2[i]:+.5f} praat={praat[i]:+.5f} "
              f"  (5x-p2={dboth[i]:+.3e}, 5x-praat={d5x[i]:+.3e}, p2-praat={dp2[i]:+.3e})")

    # Spectral diff
    print()
    fp = np.fft.rfft(praat[:n])
    f5 = np.fft.rfft(y_5x[:n])
    fp2 = np.fft.rfft(y_pow2[:n])
    freqs = np.fft.rfftfreq(n, d=1/new_rate)

    bands = [(0, 100), (100, 500), (500, 2000), (2000, 5000), (5000, 5400), (5400, 5500.01)]
    print(f"{'band':>18} | {'|5x-praat|':>12} {'|pow2-praat|':>14} | {'ratio p2/5x':>12}")
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        if not m.any(): continue
        a5 = np.mean(np.abs(f5[m] - fp[m]))
        ap = np.mean(np.abs(fp2[m] - fp[m]))
        print(f"{lo:5.0f}..{hi:7.2f} Hz  | {a5:12.3e} {ap:14.3e} | {ap/a5:12.4f}")


if __name__ == "__main__":
    main()
