#!/usr/bin/env python3
"""
Test HNR CC window size hypothesis: actual_window = (ppw + 1) / min_pitch

The "forward" cross-correlation might mean we need an extra period for the
forward-looking correlation.
"""

import numpy as np
import parselmouth
from parselmouth.praat import call

audio_path = "tests/fixtures/one_two_three_four_five.wav"
snd_pm = parselmouth.Sound(audio_path)
duration = snd_pm.duration

time_step = 0.01
min_pitch = 75.0
silence_threshold = 0.1

print("=" * 70)
print("Hypothesis: actual_window = (periods_per_window + 1) / min_pitch")
print("=" * 70)
print(f"\nDuration: {duration:.6f} s")
print(f"Time step: {time_step} s")
print(f"Min pitch: {min_pitch} Hz")

print(f"\n{'ppw':>5} {'actual':>8} {'n_pm':>6} {'n_exp':>6} {'match':>6} {'t1_pm':>10} {'t1_exp':>10} {'t1_match':>8}")
print("-" * 70)

for ppw in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.5, 6.0]:
    hnr = call(snd_pm, "To Harmonicity (cc)", time_step, min_pitch, silence_threshold, ppw)
    n_frames_pm = call(hnr, "Get number of frames")
    t1_pm = call(hnr, "Get time from frame number", 1)

    # Hypothesis: actual window = (ppw + 1) / min_pitch
    actual_periods = ppw + 1.0
    window_dur = actual_periods / min_pitch

    # Standard frame count formula
    n_expected = int(np.floor((duration - window_dur) / time_step + 1e-9)) + 1

    # Centered t1
    t1_expected = (duration - (n_expected - 1) * time_step) / 2.0

    frames_match = "✓" if n_frames_pm == n_expected else "✗"
    t1_match = "✓" if abs(t1_pm - t1_expected) < 1e-6 else f"Δ{(t1_pm - t1_expected)*1000:.2f}ms"

    print(f"{ppw:>5.1f} {actual_periods:>8.1f} {n_frames_pm:>6} {n_expected:>6} {frames_match:>6} {t1_pm:>10.6f} {t1_expected:>10.6f} {t1_match:>8}")

# Also test if maybe it's (ppw + X) where X != 1.0
print("\n" + "=" * 70)
print("Testing other offsets: window = (ppw + X) / min_pitch")
print("=" * 70)

ppw = 1.0
hnr = call(snd_pm, "To Harmonicity (cc)", time_step, min_pitch, silence_threshold, ppw)
n_frames_pm = call(hnr, "Get number of frames")
t1_pm = call(hnr, "Get time from frame number", 1)

print(f"\nWith ppw={ppw}, parselmouth gives {n_frames_pm} frames, t1={t1_pm:.6f}")
print(f"\nFinding X such that (ppw + X) / min_pitch gives correct frame count:")

for x in np.arange(0.0, 3.0, 0.1):
    actual_periods = ppw + x
    window_dur = actual_periods / min_pitch
    n_expected = int(np.floor((duration - window_dur) / time_step + 1e-9)) + 1
    t1_expected = (duration - (n_expected - 1) * time_step) / 2.0

    if n_expected == n_frames_pm:
        t1_diff = abs(t1_pm - t1_expected)
        print(f"  X = {x:.1f}: window = {actual_periods:.1f} periods = {window_dur*1000:.2f} ms → {n_expected} frames, t1_diff = {t1_diff*1000:.3f} ms")
