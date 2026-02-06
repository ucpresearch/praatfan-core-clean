#!/usr/bin/env python3
"""
Compare window sizes and frame counts between Pitch CC and Harmonicity CC.
They might use different parameters!
"""

import sys
sys.path.insert(0, '/home/urielc/local/decfiles/private/Dev/git/praatfan-core-clean/src')

import numpy as np
import parselmouth
from parselmouth.praat import call
from praatfan.sound import Sound

sound_path = "/home/urielc/local/decfiles/private/Dev/git/praatfan-core-clean/tests/fixtures/one_two_three_four_five.wav"
pm_sound = parselmouth.Sound(sound_path)

# Test with different min_pitch values to understand window behavior
for min_pitch in [75.0, 100.0, 150.0]:
    print(f"\n{'='*70}")
    print(f"min_pitch = {min_pitch} Hz")
    print(f"{'='*70}")

    time_step = 0.01

    # Pitch CC
    pm_pitch = call(pm_sound, "To Pitch (cc)", time_step, min_pitch, 15, "no",
                    0.03, 0.45, 0.01, 0.35, 0.14, 600)
    pitch_n_frames = call(pm_pitch, "Get number of frames")
    pitch_t1 = call(pm_pitch, "Get time from frame number", 1)

    # Harmonicity CC with periods_per_window = 1.0 (default)
    pm_harm_1 = call(pm_sound, "To Harmonicity (cc)", time_step, min_pitch, 0.1, 1.0)
    harm1_n_frames = call(pm_harm_1, "Get number of frames")
    harm1_t1 = call(pm_harm_1, "Get time from frame number", 1)

    # Harmonicity CC with periods_per_window = 4.5 (default for AC)
    pm_harm_45 = call(pm_sound, "To Harmonicity (cc)", time_step, min_pitch, 0.1, 4.5)
    harm45_n_frames = call(pm_harm_45, "Get number of frames")
    harm45_t1 = call(pm_harm_45, "Get time from frame number", 1)

    print(f"Pitch CC:           frames={pitch_n_frames}, t1={pitch_t1:.6f}s")
    print(f"Harmonicity CC(1.0): frames={harm1_n_frames}, t1={harm1_t1:.6f}s")
    print(f"Harmonicity CC(4.5): frames={harm45_n_frames}, t1={harm45_t1:.6f}s")

    # Infer window duration from frame count and t1
    duration = pm_sound.get_total_duration()

    # For centered frames: t1 = (duration - (n-1) * dt) / 2
    # So window_duration doesn't directly appear, but we can check
    # whether the counts match our formulas

    print()
    print("Expected window durations:")
    for periods in [1.0, 2.0, 3.0, 4.5]:
        win = periods / min_pitch
        # Centered formula: n = floor((dur - win) / dt) + 1
        n = int(np.floor((duration - win) / time_step + 1e-9)) + 1
        t1 = (duration - (n - 1) * time_step) / 2.0
        print(f"  {periods} periods (win={win:.4f}s): n={n}, t1={t1:.6f}s")

# Now let's check what algorithm Harmonicity CC actually uses
print(f"\n{'='*70}")
print("Checking if Pitch CC F0 and Harmonicity CC HNR are consistent")
print(f"{'='*70}")

min_pitch = 75.0
time_step = 0.01

pm_pitch = call(pm_sound, "To Pitch (cc)", time_step, min_pitch, 15, "no",
                0.03, 0.45, 0.01, 0.35, 0.14, 600)
pm_harm = call(pm_sound, "To Harmonicity (cc)", time_step, min_pitch, 0.1, 1.0)

pitch_n = call(pm_pitch, "Get number of frames")
harm_n = call(pm_harm, "Get number of frames")

print(f"Pitch CC frames: {pitch_n}")
print(f"Harmonicity CC frames: {harm_n}")

# Compare voiced frames
print("\nComparing first 10 voiced frames:")
count = 0
for i in range(1, pitch_n + 1):
    f0 = call(pm_pitch, "Get value in frame", i, "Hertz")
    if f0 and f0 > 0:
        t = call(pm_pitch, "Get time from frame number", i)
        # Get harmonicity at same time
        hnr = call(pm_harm, "Get value at time", t, "cubic")

        if hnr:
            # Convert HNR to r
            ratio = 10 ** (hnr / 10)
            r = ratio / (1 + ratio)
            print(f"Frame {i}: t={t:.4f}s, F0={f0:.2f} Hz, HNR={hnr:.2f} dB, r={r:.6f}")
            count += 1
            if count >= 10:
                break

# Now test if sinc interpolation makes a difference
print(f"\n{'='*70}")
print("Testing different interpolation methods")
print(f"{'='*70}")

# Extract one frame and test
my_sound = Sound.from_file(sound_path)
sample_rate = my_sound.sample_rate
samples = my_sound.samples

# Use a well-behaved voiced frame
test_t = 0.2060  # Frame 20 from earlier
center = int(round(test_t * sample_rate))

# Try different window sizes
for periods in [1.0, 2.0]:
    win_dur = periods / min_pitch
    win_samples = int(round(win_dur * sample_rate))
    if win_samples % 2 == 0:
        win_samples += 1
    half_win = win_samples // 2

    start = center - half_win
    frame = samples[start:start + win_samples].copy()

    print(f"\nWindow = {periods} periods ({win_samples} samples):")

    min_lag = int(np.ceil(sample_rate / 600))
    max_lag = int(np.floor(sample_rate / min_pitch))

    # Compute FCC
    fcc = np.zeros(max_lag + 1)
    for lag in range(min_lag, min(max_lag + 1, len(frame) // 2)):
        first = frame[:lag]
        second = frame[lag:2*lag]
        if len(second) == lag:
            corr = np.sum(first * second)
            e1 = np.sum(first * first)
            e2 = np.sum(second * second)
            if e1 > 0 and e2 > 0:
                fcc[lag] = corr / np.sqrt(e1 * e2)

    # Find best peak
    best_lag = min_lag
    for lag in range(min_lag + 1, min(max_lag, len(fcc) - 1)):
        if fcc[lag] > fcc[lag-1] and fcc[lag] > fcc[lag+1]:
            if fcc[lag] > fcc[best_lag]:
                best_lag = lag

    if best_lag > min_lag and best_lag < len(fcc) - 1:
        # Parabolic interpolation
        r_prev = fcc[best_lag - 1]
        r_curr = fcc[best_lag]
        r_next = fcc[best_lag + 1]

        denom = r_prev - 2*r_curr + r_next
        if abs(denom) > 1e-10:
            delta = 0.5 * (r_prev - r_next) / denom
            ref_lag = best_lag + delta
            ref_r = r_curr - 0.25 * (r_prev - r_next) * delta
            f0 = sample_rate / ref_lag
            print(f"  F0 = {f0:.2f} Hz, r = {ref_r:.6f}")
