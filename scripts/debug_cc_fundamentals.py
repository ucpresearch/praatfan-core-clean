#!/usr/bin/env python3
"""
Go back to basics: verify CC window duration, window function, and frame extraction.
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

time_step = 0.01

print("=" * 70)
print("FUNDAMENTAL VERIFICATION: Pitch CC Parameters")
print("=" * 70)

# Test different min_pitch values to understand window behavior
print("\n1. WINDOW DURATION INVESTIGATION")
print("-" * 70)

for min_pitch in [50.0, 75.0, 100.0, 150.0, 200.0]:
    pm_pitch = call(pm_sound, "To Pitch (cc)", time_step, min_pitch, 15, "no",
                    0.03, 0.45, 0.01, 0.35, 0.14, 600)
    n_frames = call(pm_pitch, "Get number of frames")
    t1 = call(pm_pitch, "Get time from frame number", 1)

    # Try different window durations
    print(f"\nmin_pitch = {min_pitch} Hz:")
    print(f"  Praat: n_frames={n_frames}, t1={t1:.6f}s")

    for periods in [1.0, 1.5, 2.0, 2.5, 3.0]:
        win_dur = periods / min_pitch
        # Centered formula
        n_calc = int(np.floor((duration - win_dur) / time_step + 1e-9)) + 1
        t1_calc = (duration - (n_calc - 1) * time_step) / 2.0
        match = "✓" if n_calc == n_frames and abs(t1_calc - t1) < 1e-6 else ""
        print(f"    {periods} periods (win={win_dur:.4f}s): n={n_calc}, t1={t1_calc:.6f}s {match}")

# Verify window function by checking if windowing affects results
print("\n\n2. WINDOW FUNCTION INVESTIGATION")
print("-" * 70)

min_pitch = 75.0
pm_pitch = call(pm_sound, "To Pitch (cc)", time_step, min_pitch, 15, "no",
                0.03, 0.45, 0.01, 0.35, 0.14, 600)

# Use the correct window duration (appears to be 2 periods based on frame count)
window_duration = 2.0 / min_pitch
window_samples = int(round(window_duration * sample_rate))
if window_samples % 2 == 0:
    window_samples += 1
half_window = window_samples // 2

min_lag = int(np.ceil(sample_rate / 600))
max_lag = int(np.floor(sample_rate / min_pitch))

def hanning_window(n):
    if n <= 1:
        return np.array([1.0])
    i = np.arange(n)
    return 0.5 - 0.5 * np.cos(2 * np.pi * i / (n - 1))

# Test a well-behaved voiced frame
test_t = 0.3360  # This frame had 0.15 Hz error earlier
pm_f0 = call(pm_pitch, "Get value at time", test_t, "Hertz", "linear")
print(f"\nTest frame at t={test_t}s, Praat F0={pm_f0:.2f} Hz")

center = int(round(test_t * sample_rate))
start = center - half_window
frame_raw = samples[start:start + window_samples].copy()

# Test different approaches
def compute_fcc(frame, min_lag, max_lag, use_window=False):
    """Compute forward cross-correlation with optional windowing."""
    if use_window:
        w = hanning_window(len(frame))
        frame = frame * w

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
            return sample_rate / (best_lag + delta), r_curr
    return sample_rate / best_lag, r_curr

# Test without window
fcc_raw = compute_fcc(frame_raw, min_lag, max_lag, use_window=False)
f0_raw, r_raw = find_best_peak(fcc_raw, min_lag, max_lag, sample_rate)
print(f"  No window:      F0={f0_raw:.2f} Hz, r={r_raw:.6f}, error={abs(f0_raw-pm_f0):.2f} Hz")

# Test with Hanning window on full frame
fcc_win = compute_fcc(frame_raw, min_lag, max_lag, use_window=True)
f0_win, r_win = find_best_peak(fcc_win, min_lag, max_lag, sample_rate)
print(f"  Hanning window: F0={f0_win:.2f} Hz, r={r_win:.6f}, error={abs(f0_win-pm_f0):.2f} Hz")

# Test AC-style autocorrelation (for comparison)
def compute_ac_normalized(frame, min_lag, max_lag):
    """AC-style autocorrelation with window normalization."""
    window = hanning_window(len(frame))
    windowed = frame * window

    r_a = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag < len(windowed):
            r_a[lag] = np.sum(windowed[:len(windowed)-lag] * windowed[lag:])

    r_w = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag < len(window):
            r_w[lag] = np.sum(window[:len(window)-lag] * window[lag:])

    r_norm = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if r_a[0] > 0 and r_w[0] > 0 and r_w[lag] > 0:
            r_norm[lag] = (r_a[lag] / r_a[0]) / (r_w[lag] / r_w[0])

    return r_norm

ac_norm = compute_ac_normalized(frame_raw, min_lag, max_lag)
f0_ac, r_ac = find_best_peak(ac_norm, min_lag, max_lag, sample_rate)
print(f"  AC-style:       F0={f0_ac:.2f} Hz, r={r_ac:.6f}, error={abs(f0_ac-pm_f0):.2f} Hz")

# Test with windowed periods (window each period separately)
def compute_fcc_windowed_periods(frame, min_lag, max_lag):
    """FCC with Hanning window applied to each period separately."""
    fcc = np.zeros(max_lag + 1)
    for lag in range(min_lag, min(max_lag + 1, len(frame) // 2)):
        first = frame[:lag].copy()
        second = frame[lag:2*lag].copy()
        if len(second) == lag:
            w = hanning_window(lag)
            first = first * w
            second = second * w
            corr = np.sum(first * second)
            e1 = np.sum(first * first)
            e2 = np.sum(second * second)
            if e1 > 0 and e2 > 0:
                fcc[lag] = corr / np.sqrt(e1 * e2)
    return fcc

fcc_wp = compute_fcc_windowed_periods(frame_raw, min_lag, max_lag)
f0_wp, r_wp = find_best_peak(fcc_wp, min_lag, max_lag, sample_rate)
print(f"  Windowed periods: F0={f0_wp:.2f} Hz, r={r_wp:.6f}, error={abs(f0_wp-pm_f0):.2f} Hz")

# Test full-frame cross-correlation (not just 2 periods)
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

fcc_full = compute_full_xcorr(frame_raw, min_lag, max_lag)
f0_full, r_full = find_best_peak(fcc_full, min_lag, max_lag, sample_rate)
print(f"  Full xcorr:     F0={f0_full:.2f} Hz, r={r_full:.6f}, error={abs(f0_full-pm_f0):.2f} Hz")

# Test full xcorr with window
frame_win = frame_raw * hanning_window(len(frame_raw))
fcc_full_win = compute_full_xcorr(frame_win, min_lag, max_lag)
f0_full_win, r_full_win = find_best_peak(fcc_full_win, min_lag, max_lag, sample_rate)
print(f"  Full xcorr+win: F0={f0_full_win:.2f} Hz, r={r_full_win:.6f}, error={abs(f0_full_win-pm_f0):.2f} Hz")

# Now test across many frames to find which method works best
print("\n\n3. COMPREHENSIVE TEST ACROSS ALL VOICED FRAMES")
print("-" * 70)

pm_n_frames = call(pm_pitch, "Get number of frames")

methods = {
    "FCC raw": lambda f: compute_fcc(f, min_lag, max_lag, False),
    "FCC Hanning": lambda f: compute_fcc(f, min_lag, max_lag, True),
    "AC-style": lambda f: compute_ac_normalized(f, min_lag, max_lag),
    "FCC win periods": lambda f: compute_fcc_windowed_periods(f, min_lag, max_lag),
    "Full xcorr": lambda f: compute_full_xcorr(f, min_lag, max_lag),
    "Full xcorr+win": lambda f: compute_full_xcorr(f * hanning_window(len(f)), min_lag, max_lag),
}

results = {name: [] for name in methods}

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

    for name, method in methods.items():
        fcc = method(frame)
        f0, r = find_best_peak(fcc, min_lag, max_lag, sample_rate)
        err = abs(f0 - pm_f0)
        results[name].append(err)

print(f"\n{'Method':<20} {'Mean':>8} {'<1Hz':>8} {'<2Hz':>8} {'<5Hz':>8}")
print("-" * 55)

for name, errors in results.items():
    if errors:
        mean_err = np.mean(errors)
        within_1 = 100 * sum(1 for e in errors if e < 1) / len(errors)
        within_2 = 100 * sum(1 for e in errors if e < 2) / len(errors)
        within_5 = 100 * sum(1 for e in errors if e < 5) / len(errors)
        print(f"{name:<20} {mean_err:>8.2f} {within_1:>7.1f}% {within_2:>7.1f}% {within_5:>7.1f}%")
