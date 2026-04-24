"""
Where does scipy FFT resample differ spectrally from Praat's Resample?

If we can identify which frequency band accounts for the 2.76e-3 mean
point-diff, we can apply a corrective filter to close it.
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
    praat_r = call(snd, "Resample", 11000.0, 50).values[0]

    new_length = int(len(orig) * 11000 / 24000)
    scipy_r = ss.resample(orig, new_length)
    # Current scipy pad5
    pad = 5
    padded = np.zeros(len(orig) * pad)
    padded[:len(orig)] = orig
    scipy_pad5 = ss.resample(padded, new_length * pad)[:new_length]

    # Align lengths
    n = min(len(praat_r), len(scipy_r), len(scipy_pad5))
    praat = praat_r[:n]
    sc_raw = scipy_r[:n]
    sc_pad = scipy_pad5[:n]

    # Compute spectral diff: what's scipy minus praat in the freq domain?
    fp = np.fft.rfft(praat)
    fs_raw = np.fft.rfft(sc_raw)
    fs_pad = np.fft.rfft(sc_pad)
    freqs = np.fft.rfftfreq(n, d=1/11000.0)

    diff_raw = fs_raw - fp
    diff_pad = fs_pad - fp

    # Bin the magnitude of the diff by frequency range
    bands = [(0, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 3000),
             (3000, 4000), (4000, 5000), (5000, 5400), (5400, 5500)]
    print(f"{'freq band':>15} | {'scipy_raw - praat':>25} | {'scipy_pad5 - praat':>25}")
    print(f"{'(Hz)':>15} | {'mean_abs':>12} {'max':>12} | {'mean_abs':>12} {'max':>12}")
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            continue
        mr = np.abs(diff_raw[mask])
        mp = np.abs(diff_pad[mask])
        print(f"{lo:5d}..{hi:5d}    | {np.mean(mr):12.3e} {np.max(mr):12.3e} | "
              f"{np.mean(mp):12.3e} {np.max(mp):12.3e}")

    # Relative: as fraction of Praat spectrum magnitude
    print()
    print(f"Relative diff (scipy_pad5 - praat) / |praat|, mean per band:")
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            continue
        mp = np.abs(diff_pad[mask])
        pp = np.abs(fp[mask])
        rel = mp / (pp + 1e-12)
        print(f"  {lo:5d}..{hi:5d} Hz: mean_rel={np.mean(rel):.3e} max_rel={np.max(rel):.3e} "
              f"| Praat mag mean={np.mean(pp):.3e}")


if __name__ == "__main__":
    main()
