"""
Conditional filter: keep min_freq=50 normally, but if fewer than N formants
were found and there's a valid pole < 50 Hz, include it.
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


def evaluate(label, filter_fn):
    orig = fmod._roots_to_formants
    from praatfan.formant import FormantPoint

    def patched(roots, sample_rate, min_freq=None, max_freq=None):
        # Produce ALL candidate formants and let filter_fn decide ordering/keeps
        cands = []
        for root in roots:
            if root.imag <= 0:
                continue
            r = abs(root)
            theta = np.angle(root)
            freq = theta * sample_rate / (2 * np.pi)
            bw = -np.log(r) * sample_rate / np.pi if r > 0 else float('inf')
            if bw > 0:
                cands.append(FormantPoint(freq, bw))
        return filter_fn(cands)

    fmod._roots_to_formants = patched
    try:
        all_d = {"f1": [], "f2": [], "f3": []}
        max_err_frames = {"f1": (0, "", 0)}
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
                    e = abs(fp.frequency - p)
                    all_d[f"f{k}"].append(e)
                    if k == 1 and e > max_err_frames["f1"][0]:
                        max_err_frames["f1"] = (e, name, t)
    finally:
        fmod._roots_to_formants = orig
    d1, d2, d3 = (np.array(all_d[k]) for k in ("f1", "f2", "f3"))
    print(f"{label:60s} | F1 mean={np.mean(d1):5.2f} p99={np.percentile(d1, 99):7.2f} max={np.max(d1):7.1f} | "
          f"F2 mean={np.mean(d2):5.2f} p99={np.percentile(d2, 99):7.2f} | "
          f"F3 mean={np.mean(d3):5.2f} p99={np.percentile(d3, 99):7.2f} "
          f"| max @ {max_err_frames['f1'][1]}/t={max_err_frames['f1'][2]:.3f}")


def current_filter(cands):
    result = [c for c in cands if 50 <= c.frequency <= 5450]
    result.sort(key=lambda c: c.frequency)
    return result[:5]


def no_floor_filter(cands):
    result = [c for c in cands if 0 <= c.frequency <= 5500]
    result.sort(key=lambda c: c.frequency)
    return result[:5]


def conditional_filter_rescue_low(cands, target=5, relaxed_min=0):
    # Standard
    primary = [c for c in cands if 50 <= c.frequency <= 5500]
    primary.sort(key=lambda c: c.frequency)
    if len(primary) >= target:
        return primary[:target]
    # Rescue: include below-50 poles with freq > relaxed_min, by frequency order
    rescue = [c for c in cands if relaxed_min <= c.frequency < 50 and c.bandwidth < 1000]
    rescue.sort(key=lambda c: c.frequency)
    # Combine: rescue goes first (lowest freq)
    combined = rescue + primary
    combined.sort(key=lambda c: c.frequency)
    return combined[:target]


def conditional_filter_narrow_bw(cands, target=5):
    primary = [c for c in cands if 50 <= c.frequency <= 5500]
    primary.sort(key=lambda c: c.frequency)
    if len(primary) >= target:
        return primary[:target]
    rescue = [c for c in cands if 0 <= c.frequency < 50 and c.bandwidth < 400]
    rescue.sort(key=lambda c: c.frequency)
    combined = rescue + primary
    combined.sort(key=lambda c: c.frequency)
    return combined[:target]


def main():
    evaluate("current (min_freq=50)", current_filter)
    evaluate("no floor", no_floor_filter)
    evaluate("rescue low (relaxed=0, bw<1000)", conditional_filter_rescue_low)
    evaluate("rescue low (bw<400)", conditional_filter_narrow_bw)
    for t in [5, 10, 20, 30]:
        evaluate(f"rescue low (relaxed={t})",
                 lambda cands, t=t: conditional_filter_rescue_low(cands, relaxed_min=t))


if __name__ == "__main__":
    main()
