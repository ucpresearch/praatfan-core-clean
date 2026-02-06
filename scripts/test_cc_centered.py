#!/usr/bin/env python3
"""
Test centered FCC formula across multiple frames.
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

pm_pitch = call(pm_sound, "To Pitch (cc)", time_step, min_pitch, 15, "no",
                0.03, 0.45, 0.01, 0.35, 0.14, max_pitch)
pm_harm = call(pm_sound, "To Harmonicity (cc)", time_step, min_pitch, 0.1, 1.0)

window_duration = 2.0 / min_pitch
window_samples = int(round(window_duration * sample_rate))
if window_samples % 2 == 0:
    window_samples += 1
half_window = window_samples // 2

min_lag = int(np.ceil(sample_rate / max_pitch))
max_lag = int(np.floor(sample_rate / min_pitch))

def compute_centered_fcc(frame, min_lag, max_lag):
    """
    Compute FCC using centered periods.
    For each lag, extract two consecutive periods centered in the frame.
    """
    fcc = np.zeros(max_lag + 1)
    mid = len(frame) // 2

    for lag in range(min_lag, max_lag + 1):
        # Centered periods: frame[mid-lag:mid] and frame[mid:mid+lag]
        if mid - lag >= 0 and mid + lag <= len(frame):
            first = frame[mid - lag:mid]
            second = frame[mid:mid + lag]

            corr = np.sum(first * second)
            e1 = np.sum(first * first)
            e2 = np.sum(second * second)
            if e1 > 0 and e2 > 0:
                fcc[lag] = corr / np.sqrt(e1 * e2)

    return fcc

def compute_original_fcc(frame, min_lag, max_lag):
    """Original FCC: frame[0:lag] vs frame[lag:2*lag]"""
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

    return fcc

def find_best_peak(fcc, min_lag, max_lag, sample_rate):
    """Find best peak with parabolic interpolation"""
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

    return sample_rate / best_lag, fcc[best_lag]

# Test all frames
print("Comparing original vs centered FCC")
print("=" * 80)

results = []

for i in range(1, call(pm_pitch, "Get number of frames") + 1):
    t = call(pm_pitch, "Get time from frame number", i)
    pm_f0 = call(pm_pitch, "Get value in frame", i, "Hertz")
    pm_hnr = call(pm_harm, "Get value at time", t, "cubic")

    if not pm_f0 or pm_f0 <= 0:
        continue

    pm_ratio = 10 ** (pm_hnr / 10) if pm_hnr else 0
    pm_r = pm_ratio / (1 + pm_ratio) if pm_ratio > 0 else 0

    center = int(round(t * sample_rate))
    start = center - half_window

    if start < 0 or start + window_samples > len(samples):
        continue

    frame = samples[start:start + window_samples].copy()

    # Original FCC
    fcc_orig = compute_original_fcc(frame, min_lag, max_lag)
    f0_orig, r_orig = find_best_peak(fcc_orig, min_lag, max_lag, sample_rate)

    # Centered FCC
    fcc_cent = compute_centered_fcc(frame, min_lag, max_lag)
    f0_cent, r_cent = find_best_peak(fcc_cent, min_lag, max_lag, sample_rate)

    results.append({
        't': t,
        'pm_f0': pm_f0,
        'pm_r': pm_r,
        'f0_orig': f0_orig,
        'r_orig': r_orig,
        'f0_cent': f0_cent,
        'r_cent': r_cent
    })

# Summary statistics
print(f"Tested {len(results)} voiced frames")
print()

# Filter out octave errors for r comparison
good_frames = [r for r in results if abs(r['f0_orig'] - r['pm_f0']) < 20]

print(f"Frames without octave errors: {len(good_frames)}")
print()

# Original FCC
f0_err_orig = [abs(r['f0_orig'] - r['pm_f0']) for r in good_frames]
r_err_orig = [abs(r['r_orig'] - r['pm_r']) for r in good_frames]

print("Original FCC [0:lag] vs [lag:2*lag]:")
print(f"  F0 error: mean={np.mean(f0_err_orig):.4f} Hz, max={np.max(f0_err_orig):.4f} Hz")
print(f"  r error:  mean={np.mean(r_err_orig):.6f}, max={np.max(r_err_orig):.6f}")
print()

# Centered FCC
good_frames_cent = [r for r in results if abs(r['f0_cent'] - r['pm_f0']) < 20]

f0_err_cent = [abs(r['f0_cent'] - r['pm_f0']) for r in good_frames_cent]
r_err_cent = [abs(r['r_cent'] - r['pm_r']) for r in good_frames_cent]

print("Centered FCC [mid-lag:mid] vs [mid:mid+lag]:")
print(f"  F0 error: mean={np.mean(f0_err_cent):.4f} Hz, max={np.max(f0_err_cent):.4f} Hz")
print(f"  r error:  mean={np.mean(r_err_cent):.6f}, max={np.max(r_err_cent):.6f}")
print()

# Detailed comparison for first 15 frames
print("Frame-by-frame (first 15):")
print(f"{'t':>8} {'pm_f0':>8} {'pm_r':>8} | {'f0_orig':>8} {'r_orig':>8} | {'f0_cent':>8} {'r_cent':>8}")
print("-" * 90)

for r in results[:15]:
    print(f"{r['t']:>8.4f} {r['pm_f0']:>8.2f} {r['pm_r']:>8.6f} | "
          f"{r['f0_orig']:>8.2f} {r['r_orig']:>8.6f} | "
          f"{r['f0_cent']:>8.2f} {r['r_cent']:>8.6f}")

# Distribution of r errors
print("\n" + "=" * 80)
print("Distribution of r errors (centered FCC)")
print("=" * 80)

r_errors = [abs(r['r_cent'] - r['pm_r']) for r in good_frames_cent]
bins = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 1.0]
hist, _ = np.histogram(r_errors, bins=bins)

for i in range(len(bins) - 1):
    pct = 100 * hist[i] / len(r_errors)
    bar = '█' * int(pct / 2)
    print(f"  {bins[i]:.4f} - {bins[i+1]:.4f}: {hist[i]:4d} ({pct:5.1f}%) {bar}")

# Calculate overall accuracy
within_0001 = sum(1 for e in r_errors if e < 0.0001)
within_0005 = sum(1 for e in r_errors if e < 0.0005)
within_001 = sum(1 for e in r_errors if e < 0.001)

print(f"\nAccuracy:")
print(f"  Within 0.0001: {within_0001}/{len(r_errors)} ({100*within_0001/len(r_errors):.1f}%)")
print(f"  Within 0.0005: {within_0005}/{len(r_errors)} ({100*within_0005/len(r_errors):.1f}%)")
print(f"  Within 0.001:  {within_001}/{len(r_errors)} ({100*within_001/len(r_errors):.1f}%)")
