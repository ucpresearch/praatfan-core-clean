"""
Decisive test: is the gap caused by our resampler?

Method: Use parselmouth's Sound: Resample as ground truth, then run our Burg
pipeline on THAT resampled signal. If resulting formants match Praat's full
pipeline, the gap is in our resampler. If not, it's elsewhere (window, Burg,
frame timing).
"""

from __future__ import annotations

import math
import numpy as np
import parselmouth
from parselmouth.praat import call
from pathlib import Path

from praatfan.sound import Sound as PFSound
from praatfan.formant import sound_to_formant_burg

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
FILE = FIXTURES / "one_two_three_four_five.wav"

TIME_STEP = 0.005
MAX_N_FORMANTS = 5
MAX_FORMANT = 5500.0
WINDOW = 0.025
PRE_EMPH = 50.0


def main():
    # 1. Praat end-to-end ground truth
    snd_p = parselmouth.Sound(str(FILE))
    fmt_praat = call(snd_p, "To Formant (burg)", TIME_STEP, MAX_N_FORMANTS, MAX_FORMANT, WINDOW, PRE_EMPH)

    # 2. Our pipeline on original data
    pf = PFSound.from_file(str(FILE))
    fmt_ours_orig = sound_to_formant_burg(pf, time_step=TIME_STEP, max_num_formants=MAX_N_FORMANTS,
                                            max_formant_hz=MAX_FORMANT, window_length=WINDOW,
                                            pre_emphasis_from=PRE_EMPH)

    # 3. Our pipeline on Praat-resampled data.
    #    Resample via Praat to target rate 2*max_formant = 11000 Hz.
    target_rate = 2 * MAX_FORMANT
    snd_r = call(snd_p, "Resample", target_rate, 50)  # precision=50, Praat's default
    resampled_samples = snd_r.values[0]  # (1, n) -> n
    pf_r = PFSound(resampled_samples, target_rate)
    # Run with max_formant = sr/2 so no further resampling; BUT our code
    # still runs through _resample since it sees target=original (skipped).
    fmt_ours_praat_r = sound_to_formant_burg(
        pf_r, time_step=TIME_STEP, max_num_formants=MAX_N_FORMANTS,
        max_formant_hz=MAX_FORMANT, window_length=WINDOW, pre_emphasis_from=PRE_EMPH
    )

    # Compare at a handful of times
    check_times = [0.311, 0.356, 0.361, 0.371, 0.500, 0.800, 1.000]

    def _summary(label, fmt):
        print(f"\n-- {label}")
        print(f"   {'t':>6} | {'F1':>7} {'F2':>7} {'F3':>7} {'F4':>7} {'F5':>7} {'n':>2}")
        for t in check_times:
            if hasattr(fmt, 'frames'):  # our type
                times = np.array([f.time for f in fmt.frames])
                i = int(np.argmin(np.abs(times - t)))
                fr = fmt.frames[i]
                vals = [(fp.frequency if fp else math.nan) for fp in
                        [fr.get_formant(k) for k in (1, 2, 3, 4, 5)]]
                n = fr.n_formants
                print(f"   {fr.time:>6.3f} | " + " ".join(f"{v:>7.1f}" if not math.isnan(v) else "   ---" for v in vals) + f" {n:>3}")
            else:  # parselmouth
                n = call(fmt, "Get number of formants", int(round(call(fmt, "Get frame number from time", t))))
                vals = []
                for k in (1, 2, 3, 4, 5):
                    try:
                        f = call(fmt, "Get value at time", k, t, "Hertz", "Linear")
                        vals.append(f)
                    except Exception:
                        vals.append(math.nan)
                print(f"   {t:>6.3f} | " + " ".join(f"{v:>7.1f}" if not math.isnan(v) else "   ---" for v in vals) + f" {int(n):>3}")

    _summary("Praat end-to-end", fmt_praat)
    _summary("ours (our resampler)", fmt_ours_orig)
    _summary("ours (Praat-resampled input)", fmt_ours_praat_r)


if __name__ == "__main__":
    main()
