#!/usr/bin/env python3
"""
Debug the CC peak detection to understand the 2-3 Hz systematic offset.
"""

import sys
sys.path.insert(0, '/home/urielc/local/decfiles/private/Dev/git/praatfan-core-clean/src')

import numpy as np
import parselmouth
from parselmouth.praat import call
from praatfan.sound import Sound
from praatfan.pitch import sound_to_pitch, _compute_forward_cross_correlation, _find_fcc_peaks

sound_path = "/home/urielc/local/decfiles/private/Dev/git/praatfan-core-clean/tests/fixtures/one_two_three_four_five.wav"
pm_sound = parselmouth.Sound(sound_path)
my_sound = Sound.from_file(sound_path)

sample_rate = my_sound.sample_rate
samples = my_sound.samples

min_pitch = 75.0
max_pitch = 600.0
time_step = 0.01

# Get parselmouth Pitch CC
pm_pitch = call(pm_sound, "To Pitch (cc)", time_step, min_pitch, 15, "no",
                0.03, 0.45, 0.01, 0.35, 0.14, max_pitch)

# Window parameters for CC
window_duration = 2.0 / min_pitch
window_samples = int(round(window_duration * sample_rate))
if window_samples % 2 == 0:
    window_samples += 1
half_window = window_samples // 2

min_lag = int(np.ceil(sample_rate / max_pitch))
max_lag = int(np.floor(sample_rate / min_pitch))

print("Debugging CC peak detection")
print("=" * 70)

# Analyze a frame with ~3 Hz error
test_frames = [
    (20, 0.2060),  # 3.14 Hz error
    (21, 0.2160),  # 3.19 Hz error
    (18, 0.1860),  # 0.01 Hz error (good match)
]

for frame_num, t in test_frames:
    pm_f0 = call(pm_pitch, "Get value at time", t, "Hertz", "linear")
    if not pm_f0 or np.isnan(pm_f0):
        continue

    print(f"\nFrame {frame_num} at t={t:.4f}s:")
    print(f"  Praat F0: {pm_f0:.2f} Hz")

    # Extract frame
    center = int(round(t * sample_rate))
    start = center - half_window
    frame = samples[start:start + window_samples].copy()

    # Compute FCC
    fcc = _compute_forward_cross_correlation(frame, min_lag, max_lag)

    # Find all peaks
    peaks = _find_fcc_peaks(fcc, min_lag, max_lag, sample_rate)

    print(f"  Top 5 candidates: {[(f'{f:.2f}', f'{r:.4f}') for f, r in peaks[:5]]}")

    # Expected lag from Praat
    expected_lag = int(round(sample_rate / pm_f0))

    print(f"\n  FCC values around expected lag ({expected_lag}):")
    for lag in range(expected_lag - 3, expected_lag + 4):
        if lag >= min_lag and lag < len(fcc):
            freq = sample_rate / lag
            marker = " <-- Praat lag" if lag == expected_lag else ""
            print(f"    lag {lag} ({freq:.2f} Hz): r = {fcc[lag]:.6f}{marker}")

    # Check what my implementation picked
    my_pitch = sound_to_pitch(
        my_sound,
        time_step=time_step,
        pitch_floor=min_pitch,
        pitch_ceiling=max_pitch,
        method="cc"
    )
    my_f0 = my_pitch.frames[frame_num - 1].frequency
    my_lag = int(round(sample_rate / my_f0)) if my_f0 > 0 else 0

    print(f"\n  My selected F0: {my_f0:.2f} Hz (lag ~{my_lag})")
    print(f"  Error: {abs(pm_f0 - my_f0):.2f} Hz")

# Check if it's a Viterbi issue or a peak finding issue
print("\n" + "=" * 70)
print("Checking if best peak matches Praat (before Viterbi)")
print("=" * 70)

# Get all best peaks without Viterbi
matches_before_viterbi = 0
total = 0

pm_n_frames = call(pm_pitch, "Get number of frames")
for i in range(1, pm_n_frames + 1):
    t = call(pm_pitch, "Get time from frame number", i)
    pm_f0 = call(pm_pitch, "Get value in frame", i, "Hertz")

    if not pm_f0 or np.isnan(pm_f0) or pm_f0 <= 0:
        continue

    total += 1

    # Extract frame
    center = int(round(t * sample_rate))
    start = center - half_window

    if start < 0 or start + window_samples > len(samples):
        continue

    frame = samples[start:start + window_samples].copy()

    # Compute FCC and find peaks
    fcc = _compute_forward_cross_correlation(frame, min_lag, max_lag)
    peaks = _find_fcc_peaks(fcc, min_lag, max_lag, sample_rate)

    if peaks:
        best_f0 = peaks[0][0]
        if abs(best_f0 - pm_f0) < 1.0:
            matches_before_viterbi += 1

print(f"Best peak matches Praat (before Viterbi): {matches_before_viterbi}/{total} ({100*matches_before_viterbi/total:.1f}%)")

# Check if ANY peak matches Praat
matches_any = 0
for i in range(1, pm_n_frames + 1):
    t = call(pm_pitch, "Get time from frame number", i)
    pm_f0 = call(pm_pitch, "Get value in frame", i, "Hertz")

    if not pm_f0 or np.isnan(pm_f0) or pm_f0 <= 0:
        continue

    center = int(round(t * sample_rate))
    start = center - half_window

    if start < 0 or start + window_samples > len(samples):
        continue

    frame = samples[start:start + window_samples].copy()

    fcc = _compute_forward_cross_correlation(frame, min_lag, max_lag)
    peaks = _find_fcc_peaks(fcc, min_lag, max_lag, sample_rate)

    for f0, r in peaks:
        if abs(f0 - pm_f0) < 1.0:
            matches_any += 1
            break

print(f"Any peak matches Praat: {matches_any}/{total} ({100*matches_any/total:.1f}%)")
