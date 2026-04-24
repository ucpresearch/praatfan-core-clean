"""
Fine sweep of min_freq AND secondary criteria (bandwidth cap) to find the
filter policy that best matches Praat.
"""
from __future__ import annotations

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


def make_patched(filter_fn):
    def patched(roots, sample_rate, min_freq=None, max_freq=None):
        from praatfan.formant import FormantPoint
        formants = []
        for root in roots:
            if root.imag <= 0:
                continue
            r = abs(root)
            theta = np.angle(root)
            freq = theta * sample_rate / (2 * np.pi)
            bw = -np.log(r) * sample_rate / np.pi if r > 0 else float('inf')
            if filter_fn(freq, bw):
                formants.append(FormantPoint(freq, bw))
        formants.sort(key=lambda f: f.frequency)
        return formants
    return patched


def evaluate(label, filter_fn):
    orig = fmod._roots_to_formants
    fmod._roots_to_formants = make_patched(filter_fn)
    try:
        all_d = {"f1": [], "f2": [], "f3": []}
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
                    all_d[f"f{k}"].append(abs(fp.frequency - p))
    finally:
        fmod._roots_to_formants = orig
    d1, d2, d3 = (np.array(all_d[k]) for k in ("f1", "f2", "f3"))
    print(f"{label:50s} | F1 mean={np.mean(d1):5.2f} p99={np.percentile(d1, 99):7.2f} max={np.max(d1):7.1f} | "
          f"F2 mean={np.mean(d2):5.2f} p99={np.percentile(d2, 99):7.2f} | "
          f"F3 mean={np.mean(d3):5.2f} p99={np.percentile(d3, 99):7.2f}")


def main():
    # Current
    evaluate("min_freq=50, max=5450 (current)",
             lambda f, b: 50 <= f <= 5450 and b > 0)
    # Just drop max filter (it's reasonable)
    for mf in [0, 10, 20, 30, 40, 45, 48, 50, 55, 60]:
        evaluate(f"min_freq={mf}, max=Nyquist",
                 lambda f, b, mf=mf: mf <= f <= 5500 and b > 0)
    # Conditional: allow freq<50 if bw<X
    for bw_limit in [100, 200, 300, 400, 500, 800, 1000]:
        evaluate(f"min_freq=50 OR (freq>0 AND bw<{bw_limit})",
                 lambda f, b, bl=bw_limit: (b > 0 and b < bl and f > 0) or (50 <= f <= 5500 and b > 0))
    # Praat-like: allow all in [0, Nyquist] but maybe reject spurious near-DC
    for min_bw in [1, 5, 10, 20, 50]:
        evaluate(f"min_freq=0, bw>{min_bw}",
                 lambda f, b, mb=min_bw: 0 <= f <= 5500 and b > mb)


if __name__ == "__main__":
    main()
