#!/usr/bin/env python3
"""
Verify the full cross-correlation approach for CC method.
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
duration = my_sound.duration

min_pitch = 75.0
max_pitch = 600.0
time_step = 0.01

# Window = 2 periods (confirmed)
window_duration = 2.0 / min_pitch
window_samples = int(round(window_duration * sample_rate))
if window_samples % 2 == 0:
    window_samples += 1
half_window = window_samples // 2

min_lag = int(np.ceil(sample_rate / max_pitch))
max_lag = int(np.floor(sample_rate / min_pitch))

pm_pitch = call(pm_sound, "To Pitch (cc)", time_step, min_pitch, 15, "no",
                0.03, 0.45, 0.01, 0.35, 0.14, max_pitch)

def compute_full_xcorr(frame, min_lag, max_lag):
    """Full frame cross-correlation at each lag."""
    fcc = np.zeros(max_lag + 1)
    n = len(frame)
    for lag in range(min_lag, min(max_lag + 1, n)):
        x1 = frame[:n-lag]
        x2 = frame[lag:]
        corr = np.sum(x1 * x2)
        e1 = np.sum(x1 * x1)
        e2 = np.sum(x2 * x2)
        if e1 > 0 and e2 > 0:
            fcc[lag] = corr / np.sqrt(e1 * e2)
    return fcc

def find_best_peak(fcc, min_lag, max_lag, sample_rate):
    """Find best peak with parabolic interpolation."""
    best_lag = min_lag
    best_r = 0
    for lag in range(min_lag + 1, min(max_lag, len(fcc) - 1)):
        if fcc[lag] > fcc[lag-1] and fcc[lag] > fcc[lag+1]:
            if fcc[lag] > best_r:
                best_r = fcc[lag]
                best_lag = lag

    if best_lag <= min_lag or best_lag >= len(fcc) - 1:
        return sample_rate / best_lag, fcc[best_lag] if best_lag < len(fcc) else 0

    r_prev, r_curr, r_next = fcc[best_lag-1], fcc[best_lag], fcc[best_lag+1]
    denom = r_prev - 2*r_curr + r_next
    if abs(denom) > 1e-10:
        delta = 0.5 * (r_prev - r_next) / denom
        if abs(delta) < 1:
            ref_lag = best_lag + delta
            ref_r = r_curr - 0.25 * (r_prev - r_next) * delta
            return sample_rate / ref_lag, ref_r
    return sample_rate / best_lag, r_curr

print("Verifying Full Cross-Correlation for CC Method")
print("=" * 70)

# Frame timing check
pm_n_frames = call(pm_pitch, "Get number of frames")
pm_t1 = call(pm_pitch, "Get time from frame number", 1)

n_frames = int(np.floor((duration - window_duration) / time_step + 1e-9)) + 1
t1 = (duration - (n_frames - 1) * time_step) / 2.0

print(f"\nFrame timing:")
print(f"  Praat:    n_frames={pm_n_frames}, t1={pm_t1:.6f}s")
print(f"  My calc:  n_frames={n_frames}, t1={t1:.6f}s")
print(f"  Match: {'✓' if pm_n_frames == n_frames and abs(pm_t1 - t1) < 1e-6 else '✗'}")

# Detailed frame-by-frame comparison
print(f"\nFrame-by-frame comparison (first 25 voiced):")
print(f"{'Frame':>6} {'Time':>8} {'PM_F0':>8} {'My_F0':>8} {'Error':>8}")
print("-" * 45)

errors = []
count = 0

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
    fcc = compute_full_xcorr(frame, min_lag, max_lag)
    my_f0, r = find_best_peak(fcc, min_lag, max_lag, sample_rate)

    err = abs(my_f0 - pm_f0)
    errors.append(err)

    if count < 25:
        print(f"{i:>6} {t:>8.4f} {pm_f0:>8.2f} {my_f0:>8.2f} {err:>8.2f}")
        count += 1

print(f"\nAccuracy summary ({len(errors)} voiced frames):")
print(f"  Mean error: {np.mean(errors):.2f} Hz")
print(f"  Std error:  {np.std(errors):.2f} Hz")
print(f"  Within 0.5 Hz: {100*sum(1 for e in errors if e < 0.5)/len(errors):.1f}%")
print(f"  Within 1 Hz:   {100*sum(1 for e in errors if e < 1)/len(errors):.1f}%")
print(f"  Within 2 Hz:   {100*sum(1 for e in errors if e < 2)/len(errors):.1f}%")
print(f"  Within 5 Hz:   {100*sum(1 for e in errors if e < 5)/len(errors):.1f}%")

print(f"\nPercentiles:")
print(f"  50th: {np.percentile(errors, 50):.2f} Hz")
print(f"  90th: {np.percentile(errors, 90):.2f} Hz")
print(f"  95th: {np.percentile(errors, 95):.2f} Hz")
print(f"  99th: {np.percentile(errors, 99):.2f} Hz")

# Check if errors are systematic
print(f"\nError analysis:")
signed_errors = []
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
    fcc = compute_full_xcorr(frame, min_lag, max_lag)
    my_f0, r = find_best_peak(fcc, min_lag, max_lag, sample_rate)
    signed_errors.append(my_f0 - pm_f0)

print(f"  Mean signed error: {np.mean(signed_errors):+.2f} Hz (positive = my F0 is higher)")

# Count octave errors
octave_errors = sum(1 for e in errors if e > 50)
print(f"  Octave errors (>50 Hz): {octave_errors} ({100*octave_errors/len(errors):.1f}%)")
