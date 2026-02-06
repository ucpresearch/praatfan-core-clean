#!/usr/bin/env python3
"""
Test: compute r at the expected lag (from Praat's F0) using different formulas.
This removes the peak selection problem and focuses on the r computation itself.
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
time_step = 0.01

pm_pitch = call(pm_sound, "To Pitch (cc)", time_step, min_pitch, 15, "no",
                0.03, 0.45, 0.01, 0.35, 0.14, 600)
pm_harm = call(pm_sound, "To Harmonicity (cc)", time_step, min_pitch, 0.1, 1.0)

window_duration = 2.0 / min_pitch
window_samples = int(round(window_duration * sample_rate))
if window_samples % 2 == 0:
    window_samples += 1
half_window = window_samples // 2

print("Computing r at Praat's expected lag using different formulas")
print("=" * 80)

results = []

for i in range(1, call(pm_pitch, "Get number of frames") + 1):
    t = call(pm_pitch, "Get time from frame number", i)
    pm_f0 = call(pm_pitch, "Get value in frame", i, "Hertz")
    pm_hnr = call(pm_harm, "Get value at time", t, "cubic")

    if not pm_f0 or np.isnan(pm_f0) or pm_f0 <= 0 or not pm_hnr or np.isnan(pm_hnr):
        continue

    pm_ratio = 10 ** (pm_hnr / 10)
    pm_r = pm_ratio / (1 + pm_ratio)

    center = int(round(t * sample_rate))
    start = center - half_window

    if start < 0 or start + window_samples > len(samples):
        continue

    frame = samples[start:start + window_samples].copy()

    # Expected lag from Praat's F0
    expected_lag = int(round(sample_rate / pm_f0))

    if expected_lag >= len(frame) // 2:
        continue

    mid = len(frame) // 2

    # Formula 1: Original [0:lag] vs [lag:2*lag]
    first = frame[:expected_lag]
    second = frame[expected_lag:2*expected_lag]
    if len(second) == expected_lag:
        corr = np.sum(first * second)
        e1 = np.sum(first * first)
        e2 = np.sum(second * second)
        r_orig = corr / np.sqrt(e1 * e2) if e1 > 0 and e2 > 0 else 0
    else:
        r_orig = 0

    # Formula 2: Centered [mid-lag:mid] vs [mid:mid+lag]
    if mid - expected_lag >= 0 and mid + expected_lag <= len(frame):
        first = frame[mid - expected_lag:mid]
        second = frame[mid:mid + expected_lag]
        corr = np.sum(first * second)
        e1 = np.sum(first * first)
        e2 = np.sum(second * second)
        r_cent = corr / np.sqrt(e1 * e2) if e1 > 0 and e2 > 0 else 0
    else:
        r_cent = 0

    # Formula 3: Full frame xcorr at lag
    n = len(frame)
    if expected_lag < n:
        x1 = frame[:n-expected_lag]
        x2 = frame[expected_lag:]
        corr = np.sum(x1 * x2)
        e1 = np.sum(x1 * x1)
        e2 = np.sum(x2 * x2)
        r_full = corr / np.sqrt(e1 * e2) if e1 > 0 and e2 > 0 else 0
    else:
        r_full = 0

    results.append({
        't': t,
        'pm_f0': pm_f0,
        'pm_r': pm_r,
        'lag': expected_lag,
        'r_orig': r_orig,
        'r_cent': r_cent,
        'r_full': r_full
    })

print(f"Tested {len(results)} voiced frames with valid expected lag")
print()

# Compare errors
r_err_orig = [abs(r['r_orig'] - r['pm_r']) for r in results]
r_err_cent = [abs(r['r_cent'] - r['pm_r']) for r in results]
r_err_full = [abs(r['r_full'] - r['pm_r']) for r in results]

print("R errors at expected lag (Praat's F0):")
print(f"  Original [0:lag]:     mean={np.mean(r_err_orig):.6f}, std={np.std(r_err_orig):.6f}, max={np.max(r_err_orig):.6f}")
print(f"  Centered [mid-lag]:   mean={np.mean(r_err_cent):.6f}, std={np.std(r_err_cent):.6f}, max={np.max(r_err_cent):.6f}")
print(f"  Full frame xcorr:     mean={np.mean(r_err_full):.6f}, std={np.std(r_err_full):.6f}, max={np.max(r_err_full):.6f}")
print()

# Show best frames (lowest error) for each method
print("Frame-by-frame (first 20 voiced):")
print(f"{'t':>8} {'pm_r':>10} | {'r_orig':>10} {'err':>8} | {'r_cent':>10} {'err':>8} | {'r_full':>10} {'err':>8}")
print("-" * 100)

for r in results[:20]:
    err_o = abs(r['r_orig'] - r['pm_r'])
    err_c = abs(r['r_cent'] - r['pm_r'])
    err_f = abs(r['r_full'] - r['pm_r'])
    print(f"{r['t']:>8.4f} {r['pm_r']:>10.6f} | "
          f"{r['r_orig']:>10.6f} {err_o:>8.6f} | "
          f"{r['r_cent']:>10.6f} {err_c:>8.6f} | "
          f"{r['r_full']:>10.6f} {err_f:>8.6f}")

# Histogram
print("\nR error distribution (centered formula):")
bins = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 1.0]
hist, _ = np.histogram(r_err_cent, bins=bins)
for i in range(len(bins) - 1):
    pct = 100 * hist[i] / len(r_err_cent)
    bar = '█' * int(pct / 2)
    print(f"  {bins[i]:.4f} - {bins[i+1]:.4f}: {hist[i]:4d} ({pct:5.1f}%) {bar}")

# Check systematic offset
r_diff_orig = [r['r_orig'] - r['pm_r'] for r in results]
r_diff_cent = [r['r_cent'] - r['pm_r'] for r in results]
r_diff_full = [r['r_full'] - r['pm_r'] for r in results]

print("\nSystematic offset (my_r - pm_r):")
print(f"  Original: mean={np.mean(r_diff_orig):+.6f} (we are {'higher' if np.mean(r_diff_orig) > 0 else 'lower'})")
print(f"  Centered: mean={np.mean(r_diff_cent):+.6f} (we are {'higher' if np.mean(r_diff_cent) > 0 else 'lower'})")
print(f"  Full:     mean={np.mean(r_diff_full):+.6f} (we are {'higher' if np.mean(r_diff_full) > 0 else 'lower'})")

# Test with high-HNR frames only
high_hnr = [r for r in results if r['pm_r'] > 0.95]
print(f"\nHigh-HNR frames only (r > 0.95): {len(high_hnr)} frames")

if high_hnr:
    r_err_orig_h = [abs(r['r_orig'] - r['pm_r']) for r in high_hnr]
    r_err_cent_h = [abs(r['r_cent'] - r['pm_r']) for r in high_hnr]
    r_err_full_h = [abs(r['r_full'] - r['pm_r']) for r in high_hnr]

    print(f"  Original: mean={np.mean(r_err_orig_h):.6f}, max={np.max(r_err_orig_h):.6f}")
    print(f"  Centered: mean={np.mean(r_err_cent_h):.6f}, max={np.max(r_err_cent_h):.6f}")
    print(f"  Full:     mean={np.mean(r_err_full_h):.6f}, max={np.max(r_err_full_h):.6f}")
