#!/usr/bin/env python3
"""
Test if tuning Viterbi costs improves CC accuracy.
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

# Get parselmouth reference
pm_pitch = call(pm_sound, "To Pitch (cc)", time_step, min_pitch, 15, "no",
                0.03, 0.45, 0.01, 0.35, 0.14, max_pitch)

pm_n_frames = call(pm_pitch, "Get number of frames")
pm_f0s = []
for i in range(1, pm_n_frames + 1):
    f0 = call(pm_pitch, "Get value in frame", i, "Hertz")
    pm_f0s.append(f0 if f0 and not np.isnan(f0) else 0.0)

def evaluate_accuracy(my_pitch, tolerance=1.0):
    """Evaluate accuracy against parselmouth."""
    matches = 0
    total = 0
    errors = []

    for i, pm_f0 in enumerate(pm_f0s):
        if pm_f0 > 0 and i < my_pitch.n_frames:
            total += 1
            my_f0 = my_pitch.frames[i].frequency
            if my_f0 > 0:
                err = abs(pm_f0 - my_f0)
                errors.append(err)
                if err < tolerance:
                    matches += 1

    mean_err = np.mean(errors) if errors else float('inf')
    return matches, total, mean_err

print("Testing different Viterbi cost parameters")
print("=" * 70)

# Test different octave_jump_cost values
print("\nOctave jump cost variations (tolerance=5 Hz):")
print(f"{'octave_jump':>12} {'matches':>10} {'%':>8} {'mean_err':>10}")
print("-" * 45)

for ojc in [0.1, 0.2, 0.35, 0.5, 0.7, 1.0]:
    my_pitch = sound_to_pitch(
        my_sound,
        time_step=time_step,
        pitch_floor=min_pitch,
        pitch_ceiling=max_pitch,
        method="cc",
        octave_jump_cost=ojc
    )
    matches, total, mean_err = evaluate_accuracy(my_pitch, tolerance=5.0)
    print(f"{ojc:>12.2f} {matches:>10}/{total:<3} {100*matches/total:>7.1f}% {mean_err:>10.2f}")

# Test different voiced_unvoiced_cost values
print("\nVoiced/unvoiced cost variations (tolerance=5 Hz):")
print(f"{'vuv_cost':>12} {'matches':>10} {'%':>8} {'mean_err':>10}")
print("-" * 45)

for vuc in [0.05, 0.1, 0.14, 0.2, 0.3]:
    my_pitch = sound_to_pitch(
        my_sound,
        time_step=time_step,
        pitch_floor=min_pitch,
        pitch_ceiling=max_pitch,
        method="cc",
        voiced_unvoiced_cost=vuc
    )
    matches, total, mean_err = evaluate_accuracy(my_pitch, tolerance=5.0)
    print(f"{vuc:>12.2f} {matches:>10}/{total:<3} {100*matches/total:>7.1f}% {mean_err:>10.2f}")

# Test with relaxed tolerances
print("\nAccuracy at different tolerances (default Viterbi costs):")
my_pitch = sound_to_pitch(
    my_sound,
    time_step=time_step,
    pitch_floor=min_pitch,
    pitch_ceiling=max_pitch,
    method="cc"
)

for tol in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
    matches, total, _ = evaluate_accuracy(my_pitch, tolerance=tol)
    print(f"  Within {tol:>5.1f} Hz: {matches}/{total} ({100*matches/total:.1f}%)")

# What percentage have different voicing?
print("\nVoicing agreement:")
voiced_both = 0
voiced_pm_only = 0
voiced_my_only = 0
unvoiced_both = 0

for i, pm_f0 in enumerate(pm_f0s):
    if i >= my_pitch.n_frames:
        break
    my_f0 = my_pitch.frames[i].frequency
    pm_voiced = pm_f0 > 0
    my_voiced = my_f0 > 0

    if pm_voiced and my_voiced:
        voiced_both += 1
    elif pm_voiced and not my_voiced:
        voiced_pm_only += 1
    elif not pm_voiced and my_voiced:
        voiced_my_only += 1
    else:
        unvoiced_both += 1

total = voiced_both + voiced_pm_only + voiced_my_only + unvoiced_both
print(f"  Both voiced:      {voiced_both} ({100*voiced_both/total:.1f}%)")
print(f"  PM voiced only:   {voiced_pm_only} ({100*voiced_pm_only/total:.1f}%)")
print(f"  My voiced only:   {voiced_my_only} ({100*voiced_my_only/total:.1f}%)")
print(f"  Both unvoiced:    {unvoiced_both} ({100*unvoiced_both/total:.1f}%)")
