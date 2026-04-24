"""
Inspect a bad frame in detail: what formants do we produce vs Praat?
Specifically looking for formant-assignment shifts (whole lineup off by one).
"""

from __future__ import annotations

import math
import numpy as np
import parselmouth
from parselmouth.praat import call

from praatfan.sound import Sound as PFSound
from praatfan.formant import sound_to_formant_burg

from pathlib import Path
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

FILE = FIXTURES / "one_two_three_four_five.wav"
TIMES_OF_INTEREST = [0.311, 0.356, 0.361, 0.371]

TIME_STEP = 0.005
MAX_N_FORMANTS = 5
MAX_FORMANT = 5500.0
WINDOW = 0.025
PRE_EMPH = 50.0


def main():
    # Praat side
    snd = parselmouth.Sound(str(FILE))
    fmt_p = call(snd, "To Formant (burg)", TIME_STEP, MAX_N_FORMANTS, MAX_FORMANT, WINDOW, PRE_EMPH)

    # Ours
    pf_snd = PFSound.from_file(str(FILE))
    fmt_o = sound_to_formant_burg(pf_snd, time_step=TIME_STEP, max_num_formants=MAX_N_FORMANTS,
                                   max_formant_hz=MAX_FORMANT, window_length=WINDOW, pre_emphasis_from=PRE_EMPH)

    # Locate nearest frame for each time
    for t in TIMES_OF_INTEREST:
        # Praat: find frame number whose time is closest
        n_p = call(fmt_p, "Get number of frames")
        # Use Get frame number from time
        frame_p = int(round(call(fmt_p, "Get frame number from time", t)))
        t_p = call(fmt_p, "Get time from frame number", frame_p)
        nf_p = call(fmt_p, "Get number of formants", frame_p)

        ours_times = np.array([f.time for f in fmt_o.frames])
        frame_o = int(np.argmin(np.abs(ours_times - t)))
        t_o = ours_times[frame_o]
        frame = fmt_o.frames[frame_o]

        print(f"\n=== t={t:.3f} (praat frame {frame_p}, t={t_p:.4f} | ours frame {frame_o}, t={t_o:.4f})")
        print(f"  praat: n_formants={nf_p}")
        for k in range(1, int(nf_p) + 1):
            f = call(fmt_p, "Get value at time", k, t_p, "Hertz", "Linear")
            b = call(fmt_p, "Get bandwidth at time", k, t_p, "Hertz", "Linear")
            print(f"    F{k} = {f:8.2f} Hz  B{k} = {b:8.2f} Hz")
        print(f"  ours:  n_formants={frame.n_formants}")
        for k, fp in enumerate(frame.formants, 1):
            print(f"    F{k} = {fp.frequency:8.2f} Hz  B{k} = {fp.bandwidth:8.2f} Hz")


if __name__ == "__main__":
    main()
