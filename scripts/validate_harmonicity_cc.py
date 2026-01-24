#!/usr/bin/env python3
"""
Validate Harmonicity CC implementation against parselmouth.
"""

import sys
sys.path.insert(0, '/home/urielc/local/decfiles/private/Dev/git/praatfan-core-clean/src')

import numpy as np
import parselmouth
from parselmouth.praat import call
from praatfan.sound import Sound
from praatfan.harmonicity import sound_to_harmonicity_cc

sound_path = "/home/urielc/local/decfiles/private/Dev/git/praatfan-core-clean/tests/fixtures/one_two_three_four_five.wav"
pm_sound = parselmouth.Sound(sound_path)
my_sound = Sound.from_file(sound_path)

min_pitch = 75.0
time_step = 0.01

print("=" * 70)
print("Validating Harmonicity CC Implementation")
print("=" * 70)

# Get parselmouth Harmonicity CC
pm_harm = call(pm_sound, "To Harmonicity (cc)", time_step, min_pitch, 0.1, 1.0)
pm_n_frames = call(pm_harm, "Get number of frames")

# Get my Harmonicity CC
my_harm = sound_to_harmonicity_cc(
    my_sound,
    time_step=time_step,
    min_pitch=min_pitch,
    periods_per_window=1.0
)

print(f"\nFrame count:")
print(f"  Parselmouth: {pm_n_frames}")
print(f"  My impl:     {my_harm.n_frames}")
print(f"  Match: {'✓' if pm_n_frames == my_harm.n_frames else '✗'}")

# Compare frame times
pm_times = [call(pm_harm, "Get time from frame number", i) for i in range(1, pm_n_frames + 1)]
my_times = my_harm.times

print(f"\nFirst frame time (t1):")
print(f"  Parselmouth: {pm_times[0]:.6f}s")
print(f"  My impl:     {my_times[0]:.6f}s")
print(f"  Match: {'✓' if abs(pm_times[0] - my_times[0]) < 1e-6 else '✗'}")

# Compare HNR values frame by frame
pm_hnrs = []
my_hnrs = []
for i in range(min(pm_n_frames, my_harm.n_frames)):
    pm_hnr = call(pm_harm, "Get value in frame", i + 1)
    pm_hnrs.append(pm_hnr if pm_hnr and not np.isnan(pm_hnr) else -200.0)
    my_hnrs.append(my_harm.values[i])

# Count matching frames for high-HNR regions
hnr_errors_high = []  # HNR > 15 dB
hnr_errors_all = []

for pm_hnr, my_hnr in zip(pm_hnrs, my_hnrs):
    if pm_hnr > -100 and my_hnr > -100:  # Both voiced
        err = abs(pm_hnr - my_hnr)
        hnr_errors_all.append(err)
        if pm_hnr > 15:
            hnr_errors_high.append(err)

print(f"\nHNR accuracy (all voiced frames):")
print(f"  Frames compared: {len(hnr_errors_all)}")
if hnr_errors_all:
    print(f"  Mean error: {np.mean(hnr_errors_all):.2f} dB")
    print(f"  Std error:  {np.std(hnr_errors_all):.2f} dB")
    print(f"  Max error:  {np.max(hnr_errors_all):.2f} dB")

print(f"\nHNR accuracy (high-HNR frames, > 15 dB):")
print(f"  Frames compared: {len(hnr_errors_high)}")
if hnr_errors_high:
    print(f"  Mean error: {np.mean(hnr_errors_high):.2f} dB")
    print(f"  Std error:  {np.std(hnr_errors_high):.2f} dB")
    print(f"  Max error:  {np.max(hnr_errors_high):.2f} dB")

# Detailed comparison
print(f"\nFrame-by-frame comparison (first 20 voiced):")
print(f"{'Frame':>6} {'Time':>8} {'PM HNR':>10} {'My HNR':>10} {'Error':>10}")
print("-" * 50)

count = 0
for i, (pm_hnr, my_hnr) in enumerate(zip(pm_hnrs, my_hnrs)):
    if pm_hnr > -100:  # Voiced
        t = pm_times[i]
        err = abs(pm_hnr - my_hnr) if my_hnr > -100 else float('inf')
        print(f"{i+1:>6} {t:>8.4f} {pm_hnr:>10.2f} {my_hnr:>10.2f} {err:>10.2f}")
        count += 1
        if count >= 20:
            break

# Error distribution
if hnr_errors_all:
    print(f"\nHNR error distribution:")
    bins = [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
    hist, _ = np.histogram(hnr_errors_all, bins=bins)
    for i in range(len(bins) - 1):
        pct = 100 * hist[i] / len(hnr_errors_all)
        bar = '█' * int(pct / 2)
        print(f"  {bins[i]:>5.1f} - {bins[i+1]:>5.1f} dB: {hist[i]:4d} ({pct:5.1f}%) {bar}")

    # Accuracy summary
    within_1db = sum(1 for e in hnr_errors_all if e < 1.0)
    within_2db = sum(1 for e in hnr_errors_all if e < 2.0)
    within_5db = sum(1 for e in hnr_errors_all if e < 5.0)
    print(f"\nAccuracy summary:")
    print(f"  Within 1 dB: {within_1db}/{len(hnr_errors_all)} ({100*within_1db/len(hnr_errors_all):.1f}%)")
    print(f"  Within 2 dB: {within_2db}/{len(hnr_errors_all)} ({100*within_2db/len(hnr_errors_all):.1f}%)")
    print(f"  Within 5 dB: {within_5db}/{len(hnr_errors_all)} ({100*within_5db/len(hnr_errors_all):.1f}%)")
