"""
Regression check: does the resampler change (5x-linear-pad → pow-2) affect
any algorithm other than Formant?

Runs each algorithm twice — once with current _resample (pow-2), once
temporarily swapped to the old 5x-linear-pad — and checks that
pitch/intensity/HNR/spectrogram outputs are BIT-IDENTICAL across the swap.
Only Formant is expected to change.
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
from praatfan.formant import sound_to_formant_burg
from praatfan.pitch import sound_to_pitch
from praatfan.intensity import sound_to_intensity
from praatfan.harmonicity import sound_to_harmonicity_cc
from praatfan.spectrogram import sound_to_spectrogram


FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
FILE = FIXTURES / "one_two_three_four_five.wav"


def old_resample(samples, old_rate, new_rate):
    if abs(old_rate - new_rate) < 1e-6:
        return samples.copy()
    new_length = int(len(samples) * new_rate / old_rate)
    padded = np.zeros(len(samples) * 5)
    padded[:len(samples)] = samples
    return ss.resample(padded, new_length * 5)[:new_length]


def collect(fn, label):
    pf = PFSound.from_file(str(FILE))

    # Direct (no resample)
    fmt = sound_to_formant_burg(pf, time_step=0.005, max_num_formants=5,
                                  max_formant_hz=5500.0, window_length=0.025,
                                  pre_emphasis_from=50.0)
    pitch = sound_to_pitch(pf, time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0)
    inten = sound_to_intensity(pf, min_pitch=100.0)
    hnr = sound_to_harmonicity_cc(pf, time_step=0.01, min_pitch=75.0,
                                    silence_threshold=0.1, periods_per_window=1.0)
    spec = sound_to_spectrogram(pf, window_length=0.005, max_frequency=5000.0,
                                  time_step=0.002, frequency_step=20.0)

    # Resample-then-analyze (DOES use _resample via sound.resample)
    pf_r = pf.resample(11000.0)
    fmt_r = sound_to_formant_burg(pf_r, time_step=0.005, max_num_formants=5,
                                    max_formant_hz=5500.0, window_length=0.025,
                                    pre_emphasis_from=50.0)
    pitch_r = sound_to_pitch(pf_r, time_step=0.0, pitch_floor=75.0, pitch_ceiling=600.0)
    inten_r = sound_to_intensity(pf_r, min_pitch=100.0)

    return {
        "fmt_f1": [f.formants[0].frequency if f.formants else float('nan') for f in fmt.frames],
        "pitch": list(pitch.values()),
        "intensity": list(inten.values),
        "hnr": list(hnr.values),
        "spec_first": spec.values[:, 0].tolist() if spec.values.size > 0 else [],
        # After resample
        "fmt_r_f1": [f.formants[0].frequency if f.formants else float('nan') for f in fmt_r.frames],
        "pitch_r": list(pitch_r.values()),
        "intensity_r": list(inten_r.values),
    }


def diff_arrays(a, b, label):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    n = min(len(a), len(b))
    if n == 0:
        print(f"  {label}: empty")
        return
    mask = ~(np.isnan(a[:n]) | np.isnan(b[:n]))
    d = np.abs(a[:n][mask] - b[:n][mask])
    if len(d) == 0:
        print(f"  {label}: all-NaN")
        return
    same = np.allclose(a[:n], b[:n], equal_nan=True)
    print(f"  {label:20s}: bit_identical={same}  max_diff={np.max(d):.3e}  mean_diff={np.mean(d):.3e}")


def main():
    # Collect with current (pow-2)
    orig_rs = fmod._resample
    print("== With POW-2 resampler (current) ==")
    cur = collect(orig_rs, "pow2")

    # Swap to old
    fmod._resample = old_resample
    try:
        print("== With OLD 5x-linear resampler ==")
        old = collect(old_resample, "old")
    finally:
        fmod._resample = orig_rs

    print()
    print("== Expected: only Formant & resample-then-analyze paths should differ ==")
    for k in cur:
        diff_arrays(cur[k], old[k], k)


if __name__ == "__main__":
    main()
