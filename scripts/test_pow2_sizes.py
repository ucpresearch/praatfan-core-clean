"""Try different pow-2 multipliers for FFT padding."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy import signal as ss

from praatfan.sound import Sound as PFSound
from praatfan import formant as fmod

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
FILES = ["one_two_three_four_five.wav",
         "one_two_three_four_five_16k.wav",
         "one_two_three_four_five-gain5.wav"]


def make_resample(floor_factor):
    def f(samples, old_rate, new_rate):
        if abs(old_rate - new_rate) < 1e-6:
            return samples.copy()
        new_length = int(len(samples) * new_rate / old_rate)
        pad_len = 1
        while pad_len < len(samples) * floor_factor:
            pad_len *= 2
        padded = np.zeros(pad_len)
        padded[:len(samples)] = samples
        new_pad_len = int(pad_len * new_rate / old_rate)
        return ss.resample(padded, new_pad_len)[:new_length]
    return f


def evaluate(label, resample_fn):
    orig_rs = fmod._resample
    fmod._resample = resample_fn
    try:
        d = {"f1": [], "f2": [], "f3": []}
        for name in FILES:
            path = FIXTURES / name
            snd = parselmouth.Sound(str(path))
            fmt_p = call(snd, "To Formant (burg)", 0.005, 5, 5500.0, 0.025, 50.0)
            pf = PFSound.from_file(str(path))
            fmt_o = fmod.sound_to_formant_burg(pf, time_step=0.005, max_num_formants=5,
                                                 max_formant_hz=5500.0, window_length=0.025,
                                                 pre_emphasis_from=50.0)
            n = call(fmt_p, "Get number of frames")
            times = np.array([f.time for f in fmt_o.frames])
            for i in range(1, n + 1):
                t = call(fmt_p, "Get time from frame number", i)
                j = int(np.argmin(np.abs(times - t)))
                for k in (1, 2, 3):
                    p = call(fmt_p, "Get value at time", k, t, "Hertz", "Linear")
                    fp = fmt_o.frames[j].get_formant(k)
                    if fp is None or math.isnan(p):
                        continue
                    d[f"f{k}"].append(abs(fp.frequency - p))
    finally:
        fmod._resample = orig_rs
    d1 = np.array(d["f1"]); d2 = np.array(d["f2"]); d3 = np.array(d["f3"])
    print(f"{label:30s} | F1 mean={np.mean(d1):5.2f} p95={np.percentile(d1, 95):5.2f} p99={np.percentile(d1, 99):6.2f} max={np.max(d1):7.1f} | "
          f"F2 mean={np.mean(d2):5.2f} p99={np.percentile(d2, 99):6.2f} | "
          f"F3 mean={np.mean(d3):5.2f} p99={np.percentile(d3, 99):6.2f}")


def main():
    # Old default (5x linear pad) for reference
    def old_5x(samples, old_rate, new_rate):
        if abs(old_rate - new_rate) < 1e-6: return samples.copy()
        nl = int(len(samples) * new_rate / old_rate)
        padded = np.zeros(len(samples) * 5)
        padded[:len(samples)] = samples
        return ss.resample(padded, nl * 5)[:nl]

    evaluate("OLD: linear 5x pad", old_5x)
    for factor in [2, 3, 5, 7, 10, 15, 20]:
        evaluate(f"pow-2 ≥ {factor}×", make_resample(factor))


if __name__ == "__main__":
    main()
