#!/usr/bin/env python3
"""
Test CC with full xcorr AND Viterbi path optimization.
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

print("Testing CC with Viterbi using current implementation")
print("=" * 70)

# Get parselmouth reference
pm_pitch = call(pm_sound, "To Pitch (cc)", time_step, min_pitch, 15, "no",
                0.03, 0.45, 0.01, 0.35, 0.14, max_pitch)

# Get my implementation (uses forward CC, not full xcorr)
my_pitch = sound_to_pitch(
    my_sound,
    time_step=time_step,
    pitch_floor=min_pitch,
    pitch_ceiling=max_pitch,
    method="cc"
)

pm_n_frames = call(pm_pitch, "Get number of frames")

print(f"\nFrame count: PM={pm_n_frames}, My={my_pitch.n_frames}")

# Compare
errors_current = []
for i in range(min(pm_n_frames, my_pitch.n_frames)):
    pm_f0 = call(pm_pitch, "Get value in frame", i + 1, "Hertz")
    my_f0 = my_pitch.frames[i].frequency

    if pm_f0 and not np.isnan(pm_f0) and pm_f0 > 0:
        if my_f0 > 0:
            errors_current.append(abs(pm_f0 - my_f0))

print(f"\nCurrent implementation (forward CC + Viterbi):")
print(f"  Mean error: {np.mean(errors_current):.2f} Hz")
print(f"  Within 1 Hz: {100*sum(1 for e in errors_current if e < 1)/len(errors_current):.1f}%")
print(f"  Within 2 Hz: {100*sum(1 for e in errors_current if e < 2)/len(errors_current):.1f}%")

print("\n" + "=" * 70)
print("Now testing what full xcorr with Viterbi would give")
print("=" * 70)

# I need to modify the pitch implementation to use full xcorr
# Let me do this inline for now

sample_rate = my_sound.sample_rate
samples = my_sound.samples
duration = my_sound.duration

window_duration = 2.0 / min_pitch
window_samples = int(round(window_duration * sample_rate))
if window_samples % 2 == 0:
    window_samples += 1
half_window = window_samples // 2

min_lag = int(np.ceil(sample_rate / max_pitch))
max_lag = int(np.floor(sample_rate / min_pitch))

# Frame timing
n_frames = int(np.floor((duration - window_duration) / time_step + 1e-9)) + 1
t1 = (duration - (n_frames - 1) * time_step) / 2.0

# Voicing parameters (from Boersma 1993)
voicing_threshold = 0.45
silence_threshold = 0.03
octave_cost = 0.01

global_peak = np.max(np.abs(samples))

# Collect all candidates for all frames
all_frame_candidates = []

for i in range(n_frames):
    t = t1 + i * time_step

    center = int(round(t * sample_rate))
    start = center - half_window
    end = start + window_samples

    if start < 0 or end > len(samples):
        frame_samples = np.zeros(window_samples)
        src_start = max(0, start)
        src_end = min(len(samples), end)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        frame_samples[dst_start:dst_end] = samples[src_start:src_end]
    else:
        frame_samples = samples[start:end].copy()

    # Compute full xcorr
    n = len(frame_samples)
    fcc = np.zeros(max_lag + 1)
    for lag in range(min_lag, min(max_lag + 1, n)):
        x1 = frame_samples[:n-lag]
        x2 = frame_samples[lag:]
        corr = np.sum(x1 * x2)
        e1 = np.sum(x1 * x1)
        e2 = np.sum(x2 * x2)
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
                    freq = sample_rate / ref_lag
                    # Apply octave cost
                    strength = r_curr - octave_cost * np.log2(min_pitch / freq + 1e-30)
                    candidates.append((freq, strength, r_curr))

    # Local intensity for voicing
    local_peak = np.max(np.abs(frame_samples))
    local_intensity = local_peak / (global_peak + 1e-30)

    # Unvoiced candidate
    unvoiced_strength = (voicing_threshold +
                        max(0, 2 - local_intensity/silence_threshold) *
                        (1 + voicing_threshold))
    candidates.append((0.0, unvoiced_strength, 0.0))  # freq, strength, raw_r

    # Sort by strength
    candidates.sort(key=lambda x: x[1], reverse=True)

    all_frame_candidates.append(candidates[:15])  # Keep top 15

# Viterbi path finding
octave_jump_cost = 0.35
voiced_unvoiced_cost = 0.14
time_correction = 0.01 / time_step

n_cands = [len(c) for c in all_frame_candidates]
best_cost = [np.full(n_cands[i], np.inf) for i in range(n_frames)]
best_prev = [np.zeros(n_cands[i], dtype=int) for i in range(n_frames)]

# Initialize first frame
for j, (freq, strength, r) in enumerate(all_frame_candidates[0]):
    best_cost[0][j] = -strength

# Forward pass
for i in range(1, n_frames):
    for j, (freq_j, strength_j, _) in enumerate(all_frame_candidates[i]):
        for k, (freq_k, strength_k, _) in enumerate(all_frame_candidates[i-1]):
            if freq_k == 0 and freq_j == 0:
                trans_cost = 0
            elif freq_k == 0 or freq_j == 0:
                trans_cost = voiced_unvoiced_cost
            else:
                trans_cost = octave_jump_cost * abs(np.log2(freq_j / freq_k))

            trans_cost *= time_correction
            total_cost = best_cost[i-1][k] + trans_cost - strength_j

            if total_cost < best_cost[i][j]:
                best_cost[i][j] = total_cost
                best_prev[i][j] = k

# Backward pass
path = np.zeros(n_frames, dtype=int)
path[-1] = np.argmin(best_cost[-1])
for i in range(n_frames - 2, -1, -1):
    path[i] = best_prev[i+1][path[i+1]]

# Extract results
my_f0s_viterbi = []
for i in range(n_frames):
    freq, strength, r = all_frame_candidates[i][path[i]]
    my_f0s_viterbi.append(freq)

# Compare
errors_viterbi = []
pm_f0s = []
for i in range(pm_n_frames):
    pm_f0 = call(pm_pitch, "Get value in frame", i + 1, "Hertz")
    pm_f0s.append(pm_f0 if pm_f0 and not np.isnan(pm_f0) else 0.0)

for i in range(min(pm_n_frames, n_frames)):
    pm_f0 = pm_f0s[i]
    my_f0 = my_f0s_viterbi[i]
    if pm_f0 > 0 and my_f0 > 0:
        errors_viterbi.append(abs(pm_f0 - my_f0))

print(f"\nFull xcorr + Viterbi:")
print(f"  Mean error: {np.mean(errors_viterbi):.2f} Hz")
print(f"  Within 0.5 Hz: {100*sum(1 for e in errors_viterbi if e < 0.5)/len(errors_viterbi):.1f}%")
print(f"  Within 1 Hz: {100*sum(1 for e in errors_viterbi if e < 1)/len(errors_viterbi):.1f}%")
print(f"  Within 2 Hz: {100*sum(1 for e in errors_viterbi if e < 2)/len(errors_viterbi):.1f}%")
print(f"  Within 5 Hz: {100*sum(1 for e in errors_viterbi if e < 5)/len(errors_viterbi):.1f}%")

# Detailed comparison
print(f"\nFrame-by-frame (first 20 voiced):")
print(f"{'Frame':>6} {'PM_F0':>8} {'My_F0':>8} {'Error':>8}")
print("-" * 35)

count = 0
for i in range(min(pm_n_frames, n_frames)):
    if pm_f0s[i] > 0:
        my_f0 = my_f0s_viterbi[i]
        err = abs(pm_f0s[i] - my_f0) if my_f0 > 0 else float('inf')
        print(f"{i+1:>6} {pm_f0s[i]:>8.2f} {my_f0:>8.2f} {err:>8.2f}")
        count += 1
        if count >= 20:
            break

# Percentiles
print(f"\nPercentiles:")
print(f"  50th: {np.percentile(errors_viterbi, 50):.2f} Hz")
print(f"  90th: {np.percentile(errors_viterbi, 90):.2f} Hz")
print(f"  95th: {np.percentile(errors_viterbi, 95):.2f} Hz")
