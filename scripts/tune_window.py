"""
Try different Gaussian alpha values and window types for the Burg frame
window. Measure aggregate Burg parity vs parselmouth.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

from praatfan.sound import Sound as PFSound
from praatfan import formant as fmod
from praatfan.formant import sound_to_formant_burg

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"

TIME_STEP = 0.005
MAX_N_FORMANTS = 5
MAX_FORMANT = 5500.0
WINDOW = 0.025
PRE_EMPH = 50.0

FILES = ["one_two_three_four_five.wav", "one_two_three_four_five_16k.wav"]


def make_gauss(alpha):
    def gw(n):
        if n <= 1:
            return np.array([1.0])
        mid = (n - 1) / 2.0
        i = np.arange(n)
        x = (i - mid) / mid
        w = np.exp(-alpha * x * x)
        edge = math.exp(-alpha)
        w = (w - edge) / (1.0 - edge)
        return np.maximum(w, 0.0)
    return gw


def hanning(n):
    if n <= 1:
        return np.array([1.0])
    i = np.arange(n)
    return 0.5 - 0.5 * np.cos(2 * np.pi * i / (n - 1))


def hamming(n):
    if n <= 1:
        return np.array([1.0])
    i = np.arange(n)
    return 0.54 - 0.46 * np.cos(2 * np.pi * i / (n - 1))


def blackman(n):
    if n <= 1:
        return np.array([1.0])
    i = np.arange(n)
    x = 2 * np.pi * i / (n - 1)
    return 0.42 - 0.5 * np.cos(x) + 0.08 * np.cos(2 * x)


def gauss_no_edge(alpha):
    def gw(n):
        if n <= 1:
            return np.array([1.0])
        mid = (n - 1) / 2.0
        i = np.arange(n)
        x = (i - mid) / mid
        return np.exp(-alpha * x * x)
    return gw


def run(label, win_factory):
    orig = fmod._gaussian_window
    fmod._gaussian_window = win_factory
    try:
        all_diffs = {"f1": [], "f2": [], "f3": []}
        for name in FILES:
            path = FIXTURES / name
            snd = parselmouth.Sound(str(path))
            fmt_p = call(snd, "To Formant (burg)", TIME_STEP, MAX_N_FORMANTS, MAX_FORMANT, WINDOW, PRE_EMPH)

            pf = PFSound.from_file(str(path))
            fmt_o = sound_to_formant_burg(pf, time_step=TIME_STEP, max_num_formants=MAX_N_FORMANTS,
                                            max_formant_hz=MAX_FORMANT, window_length=WINDOW,
                                            pre_emphasis_from=PRE_EMPH)
            n = call(fmt_p, "Get number of frames")
            for i in range(1, n + 1):
                t = call(fmt_p, "Get time from frame number", i)
                times_o = np.array([f.time for f in fmt_o.frames])
                j = int(np.argmin(np.abs(times_o - t)))
                for k in (1, 2, 3):
                    p = call(fmt_p, "Get value at time", k, t, "Hertz", "Linear")
                    fp = fmt_o.frames[j].get_formant(k)
                    if fp is None or math.isnan(p):
                        continue
                    all_diffs[f"f{k}"].append(abs(fp.frequency - p))
        print(f"{label:22s}:", end="")
        for k in ("f1", "f2", "f3"):
            d = np.array(all_diffs[k])
            if len(d) == 0:
                print(f" {k}: no data", end="")
            else:
                print(f" {k}: mean={np.mean(d):5.2f} p95={np.percentile(d, 95):6.2f} p99={np.percentile(d, 99):7.2f}", end="")
        print()
    finally:
        fmod._gaussian_window = orig


def main():
    # Baseline
    run("baseline(alpha=12+edge)", fmod._gaussian_window)
    # Gaussian variations
    for alpha in [4, 6, 8, 10, 12, 14, 16, 20, 24]:
        run(f"gauss α={alpha}+edge", make_gauss(alpha))
    # Without edge-subtraction
    for alpha in [6, 12, 20]:
        run(f"gauss α={alpha} (no edge)", gauss_no_edge(alpha))
    # Other windows
    run("hanning", hanning)
    run("hamming", hamming)
    run("blackman", blackman)


if __name__ == "__main__":
    main()
