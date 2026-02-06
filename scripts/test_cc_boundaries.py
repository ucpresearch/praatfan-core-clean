#!/usr/bin/env python3
"""
Test if period extraction boundaries affect the correlation values.
Also test different starting points for the periods.
"""

import sys
sys.path.insert(0, '/home/urielc/local/decfiles/private/Dev/git/praatfan-core-clean/src')

import numpy as np
import parselmouth
from parselmouth.praat import call
from praatfan.sound import Sound

sound_path = "/home/urielc/local/decfiles/private/Dev/git/praatfan-core-clean/tests/fixtures/one_two_three_four_five.wav"
pm_sound = parselmouth.Sound(sound_path)
my_sound = Sound.from_file(sound_path)

sample_rate = my_sound.sample_rate
samples = my_sound.samples

min_pitch = 75.0
max_pitch = 600.0
time_step = 0.01

# Get Praat results
pm_pitch = call(pm_sound, "To Pitch (cc)", time_step, min_pitch, 15, "no",
                0.03, 0.45, 0.01, 0.35, 0.14, max_pitch)
pm_harm = call(pm_sound, "To Harmonicity (cc)", time_step, min_pitch, 0.1, 1.0)

# Window parameters
window_duration = 2.0 / min_pitch
window_samples = int(round(window_duration * sample_rate))
if window_samples % 2 == 0:
    window_samples += 1
half_window = window_samples // 2

min_lag = int(np.ceil(sample_rate / max_pitch))
max_lag = int(np.floor(sample_rate / min_pitch))

# Test one frame in detail
test_t = 0.2060
pm_f0 = call(pm_pitch, "Get value at time", test_t, "Hertz", "linear")
pm_hnr = call(pm_harm, "Get value at time", test_t, "cubic")
pm_ratio = 10 ** (pm_hnr / 10)
pm_r = pm_ratio / (1 + pm_ratio)

expected_lag = int(round(sample_rate / pm_f0))

print(f"Test frame at t={test_t}s")
print(f"Praat F0: {pm_f0:.2f} Hz (expected lag: {expected_lag})")
print(f"Praat r: {pm_r:.6f}")
print()

center = int(round(test_t * sample_rate))
start = center - half_window
frame = samples[start:start + window_samples].copy()

print("Testing different FCC formulas at expected lag:")
print("=" * 70)

# Formula 1: frame[0:lag] vs frame[lag:2*lag] (current)
first = frame[:expected_lag]
second = frame[expected_lag:2*expected_lag]
corr = np.sum(first * second)
e1 = np.sum(first * first)
e2 = np.sum(second * second)
r1 = corr / np.sqrt(e1 * e2)
print(f"1. [0:lag] vs [lag:2*lag]:        r = {r1:.6f}")

# Formula 2: Center the periods
half_lag = expected_lag // 2
mid = len(frame) // 2
first = frame[mid-expected_lag:mid]
second = frame[mid:mid+expected_lag]
corr = np.sum(first * second)
e1 = np.sum(first * first)
e2 = np.sum(second * second)
r2 = corr / np.sqrt(e1 * e2)
print(f"2. Centered periods:              r = {r2:.6f}")

# Formula 3: Overlap by half period
offset = expected_lag // 2
first = frame[offset:offset+expected_lag]
second = frame[offset+expected_lag:offset+2*expected_lag]
if len(second) == expected_lag:
    corr = np.sum(first * second)
    e1 = np.sum(first * first)
    e2 = np.sum(second * second)
    r3 = corr / np.sqrt(e1 * e2)
    print(f"3. Offset by half period:         r = {r3:.6f}")

# Formula 4: Different energy normalization (biased)
first = frame[:expected_lag]
second = frame[expected_lag:2*expected_lag]
corr = np.sum(first * second)
# Use mean of energies instead of geometric mean
e_mean = (np.sum(first * first) + np.sum(second * second)) / 2
r4 = corr / e_mean
print(f"4. Arithmetic mean energy:        r = {r4:.6f}")

# Formula 5: Normalize by single combined segment energy
combined = frame[:2*expected_lag]
e_combined = np.sum(combined * combined)
r5 = 2 * corr / e_combined  # Factor of 2 to keep same scale
print(f"5. Combined segment energy:       r = {r5:.6f}")

# Formula 6: Cross-correlation at lag tau (standard formula)
# r(tau) = sum(x[t] * x[t+tau]) / sqrt(sum(x[t]^2) * sum(x[t+tau]^2))
# where the sums are over the overlapping region
n = len(frame)
x1 = frame[:n-expected_lag]
x2 = frame[expected_lag:]
corr = np.sum(x1 * x2)
e1 = np.sum(x1 * x1)
e2 = np.sum(x2 * x2)
r6 = corr / np.sqrt(e1 * e2)
print(f"6. Full frame xcorr at lag:       r = {r6:.6f}")

print()
print(f"Target (Praat r): {pm_r:.6f}")
print()

# Test many lags around expected lag
print("Correlation values around expected lag:")
print("-" * 50)

for lag in range(expected_lag - 3, expected_lag + 4):
    # Formula 1 (current)
    first = frame[:lag]
    second = frame[lag:2*lag]
    if len(second) == lag:
        corr = np.sum(first * second)
        e1 = np.sum(first * first)
        e2 = np.sum(second * second)
        r = corr / np.sqrt(e1 * e2)
        freq = sample_rate / lag
        marker = " <-- expected" if lag == expected_lag else ""
        print(f"lag {lag} ({freq:.2f} Hz): r = {r:.6f}{marker}")

# Now test with Viterbi to see if octave errors are resolved
print("\n" + "=" * 70)
print("Testing if Viterbi resolves octave errors")
print("=" * 70)

# Get all frames
test_times = []
for i in range(1, call(pm_pitch, "Get number of frames") + 1):
    t = call(pm_pitch, "Get time from frame number", i)
    test_times.append(t)

# Compute FCC for all frames and find all peaks (candidates)
all_frames_data = []

for t in test_times:
    pm_f0 = call(pm_pitch, "Get value at time", t, "Hertz", "linear")
    pm_hnr = call(pm_harm, "Get value at time", t, "cubic")

    center = int(round(t * sample_rate))
    start = center - half_window

    if start < 0 or start + window_samples > len(samples):
        all_frames_data.append({'t': t, 'pm_f0': pm_f0, 'candidates': []})
        continue

    frame = samples[start:start + window_samples].copy()

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

    # Find all peaks
    candidates = []
    for lag in range(min_lag + 1, min(max_lag, len(fcc) - 1)):
        if fcc[lag] > fcc[lag-1] and fcc[lag] > fcc[lag+1]:
            # Parabolic interpolation
            r_prev, r_curr, r_next = fcc[lag-1], fcc[lag], fcc[lag+1]
            denom = r_prev - 2*r_curr + r_next
            if abs(denom) > 1e-10:
                delta = 0.5 * (r_prev - r_next) / denom
                if abs(delta) < 1:
                    ref_lag = lag + delta
                    ref_r = r_curr - 0.25 * (r_prev - r_next) * delta
                    candidates.append((sample_rate / ref_lag, ref_r))

    # Sort by strength
    candidates.sort(key=lambda x: x[1], reverse=True)

    all_frames_data.append({
        't': t,
        'pm_f0': pm_f0 if pm_f0 else 0,
        'candidates': candidates[:5]  # Top 5
    })

# Count frames where best candidate matches Praat
matches_best = 0
matches_any = 0
total_voiced = 0

for fd in all_frames_data:
    if fd['pm_f0'] > 0 and fd['candidates']:
        total_voiced += 1
        best_f0 = fd['candidates'][0][0]
        if abs(best_f0 - fd['pm_f0']) < 5:  # Within 5 Hz
            matches_best += 1

        # Check if any candidate matches
        for f0, r in fd['candidates']:
            if abs(f0 - fd['pm_f0']) < 5:
                matches_any += 1
                break

print(f"Voiced frames: {total_voiced}")
print(f"Best candidate matches Praat: {matches_best} ({100*matches_best/total_voiced:.1f}%)")
print(f"Any candidate matches Praat: {matches_any} ({100*matches_any/total_voiced:.1f}%)")

# Show some examples of octave errors
print("\nFrames where best candidate doesn't match:")
count = 0
for fd in all_frames_data:
    if fd['pm_f0'] > 0 and fd['candidates']:
        best_f0 = fd['candidates'][0][0]
        if abs(best_f0 - fd['pm_f0']) >= 5:
            print(f"  t={fd['t']:.4f}: Praat={fd['pm_f0']:.1f}, best={best_f0:.1f}")
            if fd['candidates']:
                print(f"    All candidates: {[(f'{f:.1f}', f'{r:.4f}') for f, r in fd['candidates'][:4]]}")
            count += 1
            if count >= 5:
                break
