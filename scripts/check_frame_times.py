"""
Check if our frame timings exactly match Praat's.
"""
from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth.praat import call
from pathlib import Path

from praatfan.sound import Sound as PFSound
from praatfan.formant import sound_to_formant_burg

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
FILE = FIXTURES / "one_two_three_four_five.wav"


def main():
    snd = parselmouth.Sound(str(FILE))
    fmt_p = call(snd, "To Formant (burg)", 0.005, 5, 5500.0, 0.025, 50.0)
    n = call(fmt_p, "Get number of frames")
    praat_times = [call(fmt_p, "Get time from frame number", i) for i in range(1, n + 1)]

    pf = PFSound.from_file(str(FILE))
    fmt_o = sound_to_formant_burg(pf, time_step=0.005, max_num_formants=5,
                                    max_formant_hz=5500.0, window_length=0.025,
                                    pre_emphasis_from=50.0)
    ours_times = [f.time for f in fmt_o.frames]

    print(f"Praat: n={len(praat_times)} t1={praat_times[0]:.8f} t_end={praat_times[-1]:.8f} dt={praat_times[1]-praat_times[0]:.8f}")
    print(f"Ours:  n={len(ours_times)} t1={ours_times[0]:.8f} t_end={ours_times[-1]:.8f} dt={ours_times[1]-ours_times[0]:.8f}")

    # Diff
    diffs = np.array(praat_times) - np.array(ours_times)
    print(f"max |t_praat - t_ours|: {np.max(np.abs(diffs)):.3e}")


if __name__ == "__main__":
    main()
