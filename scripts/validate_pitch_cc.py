#!/usr/bin/env python3
"""
Validate Pitch CC implementation against parselmouth.
"""

import sys
sys.path.insert(0, '/home/urielc/local/decfiles/private/Dev/git/praatfan-core-clean/src')

import numpy as np
import parselmouth
from parselmouth.praat import call
from praatfan.sound import Sound
from praatfan.pitch import sound_to_pitch

sound_path = "/home/urielc/local/decfiles/private/Dev/git/praatfan-core-clean/tests/fixtures/one_two_three_four_five.wav"
pm_sound = parselmouth.Sound(sound_path)
my_sound = Sound.from_file(sound_path)

min_pitch = 75.0
max_pitch = 600.0
time_step = 0.01

print("=" * 70)
print("Validating Pitch CC Implementation")
print("=" * 70)

# Get parselmouth Pitch CC
pm_pitch = call(pm_sound, "To Pitch (cc)", time_step, min_pitch, 15, "no",
                0.03, 0.45, 0.01, 0.35, 0.14, max_pitch)
pm_n_frames = call(pm_pitch, "Get number of frames")

# Get my Pitch CC
my_pitch = sound_to_pitch(
    my_sound,
    time_step=time_step,
    pitch_floor=min_pitch,
    pitch_ceiling=max_pitch,
    method="cc"
)

print(f"\nFrame count:")
print(f"  Parselmouth: {pm_n_frames}")
print(f"  My impl:     {my_pitch.n_frames}")
print(f"  Match: {'✓' if pm_n_frames == my_pitch.n_frames else '✗'}")

# Compare frame times
pm_times = [call(pm_pitch, "Get time from frame number", i) for i in range(1, pm_n_frames + 1)]
my_times = my_pitch.times()

print(f"\nFirst frame time (t1):")
print(f"  Parselmouth: {pm_times[0]:.6f}s")
print(f"  My impl:     {my_times[0]:.6f}s")
print(f"  Match: {'✓' if abs(pm_times[0] - my_times[0]) < 1e-6 else '✗'}")

# Compare F0 values frame by frame
pm_f0s = []
my_f0s = []
for i in range(min(pm_n_frames, my_pitch.n_frames)):
    pm_f0 = call(pm_pitch, "Get value in frame", i + 1, "Hertz")
    pm_f0s.append(pm_f0 if pm_f0 and not np.isnan(pm_f0) else 0.0)
    my_f0s.append(my_pitch.frames[i].frequency)

# Count matches
voiced_matches = 0
total_voiced = 0
f0_errors = []

for i, (pm_f0, my_f0) in enumerate(zip(pm_f0s, my_f0s)):
    pm_voiced = pm_f0 > 0
    my_voiced = my_f0 > 0

    if pm_voiced:
        total_voiced += 1
        if my_voiced and abs(pm_f0 - my_f0) < 1.0:  # Within 1 Hz
            voiced_matches += 1
            f0_errors.append(abs(pm_f0 - my_f0))

print(f"\nVoicing agreement:")
print(f"  Parselmouth voiced frames: {total_voiced}")
print(f"  Matching within 1 Hz: {voiced_matches}/{total_voiced} ({100*voiced_matches/total_voiced:.1f}%)")

if f0_errors:
    print(f"\nF0 accuracy (for matching frames):")
    print(f"  Mean error: {np.mean(f0_errors):.4f} Hz")
    print(f"  Std error:  {np.std(f0_errors):.4f} Hz")
    print(f"  Max error:  {np.max(f0_errors):.4f} Hz")

# Detailed comparison
print(f"\nFrame-by-frame comparison (first 20 voiced):")
print(f"{'Frame':>6} {'Time':>8} {'PM F0':>10} {'My F0':>10} {'Error':>8} {'Match':>6}")
print("-" * 55)

count = 0
for i, (pm_f0, my_f0) in enumerate(zip(pm_f0s, my_f0s)):
    if pm_f0 > 0:
        t = pm_times[i]
        err = abs(pm_f0 - my_f0) if my_f0 > 0 else float('inf')
        match = '✓' if err < 1.0 else ('octave' if err > 50 else '✗')
        print(f"{i+1:>6} {t:>8.4f} {pm_f0:>10.2f} {my_f0:>10.2f} {err:>8.2f} {match:>6}")
        count += 1
        if count >= 20:
            break

# Error distribution
print(f"\nF0 error distribution (all voiced):")
all_errors = []
octave_errors = 0
for pm_f0, my_f0 in zip(pm_f0s, my_f0s):
    if pm_f0 > 0 and my_f0 > 0:
        err = abs(pm_f0 - my_f0)
        all_errors.append(err)
        if err > 50:
            octave_errors += 1

if all_errors:
    bins = [0, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 1000.0]
    hist, _ = np.histogram(all_errors, bins=bins)
    for i in range(len(bins) - 1):
        pct = 100 * hist[i] / len(all_errors)
        bar = '█' * int(pct / 2)
        print(f"  {bins[i]:>6.1f} - {bins[i+1]:>6.1f} Hz: {hist[i]:4d} ({pct:5.1f}%) {bar}")

    print(f"\nOctave errors (>50 Hz): {octave_errors}/{len(all_errors)} ({100*octave_errors/len(all_errors):.1f}%)")
