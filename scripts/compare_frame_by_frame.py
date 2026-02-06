#!/usr/bin/env python3
"""
Compare Python and Rust formant results frame by frame.
"""

import numpy as np
import subprocess
import json
import sys
sys.path.insert(0, 'src')

from praatfan.sound import Sound
from praatfan.formant import sound_to_formant_burg

def main():
    path = "tests/fixtures/tam-haʃaʁav-haɡadol-mono.wav"

    # Python formant
    sound = Sound.from_file(path)
    formant = sound_to_formant_burg(sound, 0.0, 5, 5500.0, 0.025, 50.0)

    print(f"Python: {formant.n_frames} frames")

    # Get F1 values
    f1_python = formant.formant_values(1)

    # Print first 20 frames
    print("\nFrame-by-frame F1 comparison (first 20 voiced frames):")
    print("Frame  Time      Python F1")
    print("-" * 35)

    voiced_count = 0
    for i, frame in enumerate(formant.frames):
        if len(frame.formants) > 0:
            f1 = frame.formants[0].frequency
            print(f"{i:4d}   {frame.time:.4f}s   {f1:7.1f} Hz")
            voiced_count += 1
            if voiced_count >= 20:
                break

    # Statistics
    valid_f1 = f1_python[~np.isnan(f1_python)]
    print(f"\nPython statistics:")
    print(f"  Total frames: {formant.n_frames}")
    print(f"  Voiced frames: {len(valid_f1)}")
    print(f"  Mean F1: {np.mean(valid_f1):.1f} Hz")
    print(f"  Std F1: {np.std(valid_f1):.1f} Hz")
    print(f"  Min F1: {np.min(valid_f1):.1f} Hz")
    print(f"  Max F1: {np.max(valid_f1):.1f} Hz")

    # Also compare with parselmouth
    try:
        import parselmouth
        from parselmouth.praat import call

        snd = parselmouth.Sound(path)
        pm_formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        n_frames = call(pm_formant, "Get number of frames")

        f1_praat = []
        for i in range(1, n_frames + 1):
            t = call(pm_formant, "Get time from frame number", i)
            f1 = call(pm_formant, "Get value at time", 1, t, "Hertz", "Linear")
            f1_praat.append(f1)

        f1_praat = np.array(f1_praat)
        valid_praat = f1_praat[~np.isnan(f1_praat)]

        print(f"\nParselmouth (ground truth):")
        print(f"  Total frames: {n_frames}")
        print(f"  Voiced frames: {len(valid_praat)}")
        print(f"  Mean F1: {np.mean(valid_praat):.1f} Hz")
        print(f"  Std F1: {np.std(valid_praat):.1f} Hz")

        # Compare Python vs Praat
        if len(valid_f1) == len(valid_praat):
            diff = valid_f1 - valid_praat
            print(f"\nPython vs Praat:")
            print(f"  Mean diff: {np.mean(diff):.2f} Hz")
            print(f"  Max diff: {np.max(np.abs(diff)):.2f} Hz")
            within_1hz = np.sum(np.abs(diff) < 1.0)
            print(f"  Within 1 Hz: {within_1hz}/{len(diff)} ({100*within_1hz/len(diff):.1f}%)")
    except ImportError:
        print("\nParselmouth not available for comparison")

if __name__ == "__main__":
    main()
