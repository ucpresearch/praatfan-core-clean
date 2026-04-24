"""
Directly compare wsinc (Agent D formula) vs scipy pow-2 for Burg parity,
partitioned by frame energy. Hypothesis: wsinc wins on voice frames
(high energy), loses on silent/boundary frames (low energy or zero input).
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

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
FILES = ["one_two_three_four_five.wav",
         "one_two_three_four_five_16k.wav",
         "one_two_three_four_five_32float.wav",
         "one_two_three_four_five.flac",
         "one_two_three_four_five-gain5.wav"]


def make_wsinc():
    def rs(samples, old_rate, new_rate):
        return fmod._resample_wsinc(samples, old_rate, new_rate, precision=50)
    return rs


def make_scipy_pow2():
    def rs(samples, old_rate, new_rate):
        if abs(old_rate - new_rate) < 1e-6: return samples.copy()
        n = len(samples)
        new_length = int(n * new_rate / old_rate)
        pad_len = 1
        while pad_len < n * 2: pad_len *= 2
        padded = np.zeros(pad_len); padded[:n] = samples
        new_pad_len = int(pad_len * new_rate / old_rate)
        return ss.resample(padded, new_pad_len)[:new_length]
    return rs


def evaluate(label, resample_fn):
    orig = fmod._resample
    fmod._resample = resample_fn
    # Rows: (file, t, f1_err, f2_err, f3_err, frame_rms)
    rows = []
    try:
        for name in FILES:
            path = FIXTURES / name
            snd = parselmouth.Sound(str(path))
            sr = snd.sampling_frequency
            orig_samples = snd.values[0].copy()
            fmt_p = call(snd, "To Formant (burg)", 0.005, 5, 5500.0, 0.025, 50.0)
            pf = PFSound.from_file(str(path))
            fmt_o = fmod.sound_to_formant_burg(pf, time_step=0.005, max_num_formants=5,
                                                 max_formant_hz=5500.0, window_length=0.025,
                                                 pre_emphasis_from=50.0)
            n = call(fmt_p, "Get number of frames")
            times = np.array([f.time for f in fmt_o.frames])
            half = int(0.025 * sr)
            for i in range(1, n + 1):
                t = call(fmt_p, "Get time from frame number", i)
                j = int(np.argmin(np.abs(times - t)))
                c = int(round(t * sr))
                a = max(0, c - half); b = min(len(orig_samples), c + half)
                rms = float(np.sqrt(np.mean(orig_samples[a:b] ** 2)))
                row = {"rms": rms}
                for k in (1, 2, 3):
                    p = call(fmt_p, "Get value at time", k, t, "Hertz", "Linear")
                    fp = fmt_o.frames[j].get_formant(k)
                    if fp is not None and not math.isnan(p):
                        row[f"f{k}"] = abs(fp.frequency - p)
                    else:
                        row[f"f{k}"] = None
                rows.append(row)
    finally:
        fmod._resample = orig
    return rows


def summarize(label, rows):
    rms_vals = np.array([r["rms"] for r in rows])
    thr_silent = np.percentile(rms_vals, 20)
    thr_voice = np.percentile(rms_vals, 50)
    print(f"\n== {label}")
    print(f"   silent_thr(p20)={thr_silent:.3e}  voice_thr(p50)={thr_voice:.3e}")
    for region_name, mask in [
        ("overall",  np.ones(len(rows), dtype=bool)),
        ("silent  ", rms_vals < thr_silent),
        ("mid     ", (rms_vals >= thr_silent) & (rms_vals < thr_voice)),
        ("voice   ", rms_vals >= thr_voice),
    ]:
        idx = np.where(mask)[0]
        for k in ("f1", "f2", "f3"):
            vals = np.array([rows[i][k] for i in idx if rows[i][k] is not None])
            if len(vals) == 0:
                continue
            print(f"   {region_name} {k}: n={len(vals):4d} "
                  f"mean={np.mean(vals):6.2f} p95={np.percentile(vals, 95):6.2f} "
                  f"p99={np.percentile(vals, 99):7.2f} max={np.max(vals):7.1f}")


def main():
    rows_scipy = evaluate("scipy pow-2", make_scipy_pow2())
    rows_wsinc = evaluate("wsinc (Agent D)", make_wsinc())
    summarize("scipy pow-2", rows_scipy)
    summarize("wsinc (Agent D formula)", rows_wsinc)


if __name__ == "__main__":
    main()
