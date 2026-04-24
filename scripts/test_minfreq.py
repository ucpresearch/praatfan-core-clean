"""Sweep min_freq and max_freq filter values to see which matches Praat."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

from praatfan.sound import Sound as PFSound
from praatfan import formant as fmod

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
FILES = ["one_two_three_four_five.wav",
         "one_two_three_four_five_16k.wav",
         "one_two_three_four_five-gain5.wav"]

MAX_FORMANT = 5500.0


def evaluate(min_freq, max_freq):
    # Monkeypatch: ignore caller-supplied filter bounds, use the ones we're testing
    orig = fmod._roots_to_formants

    def patched(roots, sample_rate, min_freq=None, max_freq=None, _mf=min_freq, _Mf=max_freq):
        from praatfan.formant import FormantPoint
        formants = []
        for root in roots:
            if root.imag <= 0:
                continue
            r = abs(root)
            theta = np.angle(root)
            freq = theta * sample_rate / (2 * np.pi)
            bandwidth = -np.log(r) * sample_rate / np.pi if r > 0 else float('inf')
            if _mf <= freq <= _Mf and bandwidth > 0:
                formants.append(FormantPoint(freq, bandwidth))
        formants.sort(key=lambda f: f.frequency)
        return formants

    fmod._roots_to_formants = patched
    try:
        all_diffs = {k: [] for k in ("f1", "f2", "f3")}
        for name in FILES:
            path = FIXTURES / name
            snd = parselmouth.Sound(str(path))
            fmt_p = call(snd, "To Formant (burg)", 0.005, 5, MAX_FORMANT, 0.025, 50.0)
            pf = PFSound.from_file(str(path))
            fmt_o = fmod.sound_to_formant_burg(pf, time_step=0.005, max_num_formants=5,
                                                max_formant_hz=MAX_FORMANT, window_length=0.025,
                                                pre_emphasis_from=50.0)
            n = call(fmt_p, "Get number of frames")
            times_o = np.array([f.time for f in fmt_o.frames])
            for i in range(1, n + 1):
                t = call(fmt_p, "Get time from frame number", i)
                j = int(np.argmin(np.abs(times_o - t)))
                for k in (1, 2, 3):
                    p = call(fmt_p, "Get value at time", k, t, "Hertz", "Linear")
                    fp = fmt_o.frames[j].get_formant(k)
                    if fp is None or math.isnan(p):
                        continue
                    all_diffs[f"f{k}"].append(abs(fp.frequency - p))
    finally:
        fmod._roots_to_formants = orig
    return all_diffs


def main():
    print(f"{'min_freq':>9} {'max_freq':>9} | {'F1 mean':>8} {'F1 p95':>8} {'F1 p99':>8} {'F1 max':>8} | {'F2 mean':>8} {'F2 p99':>8} | {'F3 mean':>8} {'F3 p99':>8}")
    configs = [
        (50, 5450),    # current
        (0, 5500),     # no filter on freq
        (0, 5450),
        (20, 5450),
        (50, 5500),
        (50, MAX_FORMANT),
        (0, MAX_FORMANT),
    ]
    for lo, hi in configs:
        d = evaluate(lo, hi)
        print(f"{lo:>9.0f} {hi:>9.0f} | "
              f"{np.mean(d['f1']):8.2f} {np.percentile(d['f1'], 95):8.2f} {np.percentile(d['f1'], 99):8.2f} {np.max(d['f1']):8.2f} | "
              f"{np.mean(d['f2']):8.2f} {np.percentile(d['f2'], 99):8.2f} | "
              f"{np.mean(d['f3']):8.2f} {np.percentile(d['f3'], 99):8.2f}")


if __name__ == "__main__":
    main()
