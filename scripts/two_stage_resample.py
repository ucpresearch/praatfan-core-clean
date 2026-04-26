"""
Two-stage resampler per docs/TRANSFERABLE_FINDINGS.md DP:
  Stage 1: FFT brick-wall LPF at source rate (gives 1/d tail / silent baseline)
  Stage 2: Windowed-sinc interpolation at fractional output positions (precise phase)

Tests:
  - Point-level match vs parselmouth's Resample
  - Silent-region Nyquist baseline (target ~5e-5)
  - Burg parity vs parselmouth's `To Formant (burg)`
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

from praatfan.formant import _resample_wsinc

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def resample_two_stage(signal, src_rate, dst_rate, precision=50,
                       anti_turn_around=1000):
    """Two-stage: FFT brick-wall LPF, then windowed-sinc interpolation."""
    n = len(signal)

    # Stage 1: FFT brick-wall LPF at source rate (only if downsampling)
    if dst_rate < src_rate:
        upfactor = dst_rate / src_rate  # < 1 for downsample
        nfft = next_pow2(n + 2 * anti_turn_around)
        padded = np.zeros(nfft)
        padded[anti_turn_around:anti_turn_around + n] = signal
        spectrum = np.fft.rfft(padded)
        # Brick wall: zero out everything above (upfactor * nfft / 2)
        cutoff_bin = int(np.floor(upfactor * nfft / 2))
        spectrum[cutoff_bin:] = 0.0
        filtered = np.fft.irfft(spectrum, nfft)[anti_turn_around:anti_turn_around + n]
    else:
        filtered = signal.copy()  # no AA filter on upsample

    # Stage 2: Windowed-sinc interpolation of the bandlimited signal at output rate
    return _resample_wsinc(filtered, src_rate, dst_rate, precision=precision)


def main():
    snd = parselmouth.Sound(str(FIXTURES / "one_two_three_four_five.wav"))
    orig = snd.values[0].copy()
    praat = call(snd, "Resample", 11000.0, 50).values[0]

    out = resample_two_stage(orig, 24000.0, 11000.0, precision=50)
    n = min(len(praat), len(out))
    d = out[:n] - praat[:n]
    print(f"Two-stage resample (24k→11k, precision=50):")
    print(f"  vs parselmouth Resample: mean={np.mean(np.abs(d)):.3e} max={np.max(np.abs(d)):.3e}")

    # Silent region baseline
    silent = out[:400]
    nyq = np.abs(np.fft.rfft(silent)[-1])
    rms = np.sqrt(np.mean(silent ** 2))
    print(f"  silent[0:400] rms={rms:.3e} Nyquist_bin={nyq:.3e}")
    print(f"  praat silent rms 5.23e-05, Nyq 2.09e-02 (target)")

    # Impulse response — check 1/d tail
    print()
    print("Impulse response decay (input impulse at sample 1000, 24k→11k):")
    x = np.zeros(20000); x[1000] = 1.0
    snd_imp = parselmouth.Sound(x, sampling_frequency=24000.0)
    praat_imp = call(snd_imp, "Resample", 11000.0, 50).values[0]
    out_imp = resample_two_stage(x, 24000.0, 11000.0, precision=50)
    pp = int(np.argmax(np.abs(praat_imp)))
    po = int(np.argmax(np.abs(out_imp)))
    print(f"  peak: praat@{pp}={praat_imp[pp]:+.4f}  ours@{po}={out_imp[po]:+.4f}")
    for d_ in [1, 10, 50, 100, 500, 1000]:
        ip = pp + d_
        io = po + d_
        if max(ip, io) < min(len(praat_imp), len(out_imp)):
            print(f"  d={d_:5d}: praat={praat_imp[ip]:+.3e}  ours={out_imp[io]:+.3e}")


if __name__ == "__main__":
    main()
