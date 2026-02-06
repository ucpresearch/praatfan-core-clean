#!/usr/bin/env python3
"""
Test a hybrid approach: maybe Praat uses one algorithm for finding the peak
and a different computation for the strength value.

Also test whether the CC method uses Viterbi smoothing.
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

def hanning_window(n):
    if n <= 1:
        return np.array([1.0])
    i = np.arange(n)
    return 0.5 - 0.5 * np.cos(2 * np.pi * i / (n - 1))

# Hypothesis: Maybe Praat uses full xcorr windowed to find the LAG,
# but then computes the STRENGTH using the raw FCC at that lag

print("Testing hypothesis: Use windowed xcorr for F0, raw FCC for strength")
print("=" * 70)

test_times = []
for i in range(1, call(pm_pitch, "Get number of frames") + 1):
    f0 = call(pm_pitch, "Get value in frame", i, "Hertz")
    if f0 and f0 > 150 and f0 < 300:
        t = call(pm_pitch, "Get time from frame number", i)
        hnr = call(pm_harm, "Get value at time", t, "cubic")
        if hnr and hnr > 15:
            test_times.append(t)
            if len(test_times) >= 15:
                break

print(f"Testing {len(test_times)} frames")
print()

results = []

for t in test_times:
    # Get Praat values
    pm_f0 = call(pm_pitch, "Get value at time", t, "Hertz", "linear")
    pm_hnr = call(pm_harm, "Get value at time", t, "cubic")
    if not pm_f0 or not pm_hnr:
        continue
    pm_ratio = 10 ** (pm_hnr / 10)
    pm_r = pm_ratio / (1 + pm_ratio)

    # Extract frame
    center = int(round(t * sample_rate))
    start = center - half_window
    if start < 0 or start + window_samples > len(samples):
        continue
    frame = samples[start:start + window_samples].copy()
    window = hanning_window(len(frame))
    windowed = frame * window

    # Method 1: Raw FCC for both F0 and r
    fcc_raw = np.zeros(max_lag + 1)
    for lag in range(min_lag, min(max_lag + 1, len(frame) // 2)):
        first = frame[:lag]
        second = frame[lag:2*lag]
        if len(second) == lag:
            corr = np.sum(first * second)
            e1 = np.sum(first * first)
            e2 = np.sum(second * second)
            if e1 > 0 and e2 > 0:
                fcc_raw[lag] = corr / np.sqrt(e1 * e2)

    # Method 2: Windowed full xcorr for F0
    fcc_win = np.zeros(max_lag + 1)
    for lag in range(min_lag, min(max_lag + 1, len(windowed))):
        x1 = windowed[:len(windowed)-lag]
        x2 = windowed[lag:]
        corr = np.sum(x1 * x2)
        e1 = np.sum(x1 * x1)
        e2 = np.sum(x2 * x2)
        if e1 > 0 and e2 > 0:
            fcc_win[lag] = corr / np.sqrt(e1 * e2)

    # Find best lag using windowed method
    best_lag_win = min_lag
    for lag in range(min_lag + 1, min(max_lag, len(fcc_win) - 1)):
        if fcc_win[lag] > fcc_win[lag-1] and fcc_win[lag] > fcc_win[lag+1]:
            if fcc_win[lag] > fcc_win[best_lag_win]:
                best_lag_win = lag

    # Find best lag using raw method
    best_lag_raw = min_lag
    for lag in range(min_lag + 1, min(max_lag, len(fcc_raw) - 1)):
        if fcc_raw[lag] > fcc_raw[lag-1] and fcc_raw[lag] > fcc_raw[lag+1]:
            if fcc_raw[lag] > fcc_raw[best_lag_raw]:
                best_lag_raw = lag

    # Interpolate lags
    def interp_lag(arr, lag):
        if lag <= min_lag or lag >= len(arr) - 1:
            return lag, arr[lag] if lag < len(arr) else 0
        r_prev, r_curr, r_next = arr[lag-1], arr[lag], arr[lag+1]
        denom = r_prev - 2*r_curr + r_next
        if abs(denom) > 1e-10:
            delta = 0.5 * (r_prev - r_next) / denom
            if abs(delta) < 1:
                return lag + delta, r_curr - 0.25 * (r_prev - r_next) * delta
        return lag, r_curr

    lag_win_interp, r_win = interp_lag(fcc_win, best_lag_win)
    lag_raw_interp, r_raw = interp_lag(fcc_raw, best_lag_raw)

    # Hybrid: use windowed lag, but raw r at that lag
    # Find the closest integer lag in the raw array
    hybrid_lag = int(round(lag_win_interp))
    if hybrid_lag < min_lag:
        hybrid_lag = min_lag
    if hybrid_lag >= len(fcc_raw) - 1:
        hybrid_lag = len(fcc_raw) - 2
    _, r_hybrid = interp_lag(fcc_raw, hybrid_lag)

    f0_win = sample_rate / lag_win_interp
    f0_raw = sample_rate / lag_raw_interp

    results.append({
        't': t,
        'pm_f0': pm_f0,
        'pm_r': pm_r,
        'f0_win': f0_win,
        'r_win': r_win,
        'f0_raw': f0_raw,
        'r_raw': r_raw,
        'r_hybrid': r_hybrid
    })

# Analyze results
print(f"{'t':>8} {'pm_f0':>8} {'pm_r':>8} | {'f0_win':>8} {'r_win':>8} | {'f0_raw':>8} {'r_raw':>8}")
print("-" * 90)

for r in results[:10]:
    print(f"{r['t']:>8.4f} {r['pm_f0']:>8.2f} {r['pm_r']:>8.6f} | "
          f"{r['f0_win']:>8.2f} {r['r_win']:>8.6f} | "
          f"{r['f0_raw']:>8.2f} {r['r_raw']:>8.6f}")

print()
print("Error summary:")
print(f"{'Method':<25} {'F0 mean':>10} {'F0 max':>10} {'r mean':>10} {'r max':>10}")
print("-" * 65)

# Windowed
f0_err_win = [abs(r['f0_win'] - r['pm_f0']) for r in results]
r_err_win = [abs(r['r_win'] - r['pm_r']) for r in results]
print(f"{'Windowed xcorr':<25} {np.mean(f0_err_win):>10.4f} {np.max(f0_err_win):>10.4f} "
      f"{np.mean(r_err_win):>10.6f} {np.max(r_err_win):>10.6f}")

# Raw FCC
f0_err_raw = [abs(r['f0_raw'] - r['pm_f0']) for r in results]
r_err_raw = [abs(r['r_raw'] - r['pm_r']) for r in results]
print(f"{'Raw FCC':<25} {np.mean(f0_err_raw):>10.4f} {np.max(f0_err_raw):>10.4f} "
      f"{np.mean(r_err_raw):>10.6f} {np.max(r_err_raw):>10.6f}")

# Hybrid: windowed F0, raw r
r_err_hybrid = [abs(r['r_hybrid'] - r['pm_r']) for r in results]
print(f"{'Hybrid (win F0, raw r)':<25} {np.mean(f0_err_win):>10.4f} {np.max(f0_err_win):>10.4f} "
      f"{np.mean(r_err_hybrid):>10.6f} {np.max(r_err_hybrid):>10.6f}")

# Let's also check if there's a systematic offset in r values
print("\n" + "=" * 70)
print("Checking for systematic offset in r values")
print("=" * 70)

r_diff_raw = [r['r_raw'] - r['pm_r'] for r in results]
print(f"Raw FCC r - Praat r: mean={np.mean(r_diff_raw):.6f}, std={np.std(r_diff_raw):.6f}")

# Maybe Praat's CC uses some attenuation factor?
print("\nTesting if Praat applies a scaling factor to r:")
for scale in [0.95, 0.96, 0.97, 0.98, 0.99, 1.0]:
    scaled_r_err = [abs(r['r_raw'] * scale - r['pm_r']) for r in results]
    print(f"  scale={scale}: mean r error = {np.mean(scaled_r_err):.6f}")

# Also check F0 offset
f0_diff = [r['f0_win'] - r['pm_f0'] for r in results if abs(r['f0_win'] - r['pm_f0']) < 10]
print(f"\nWindowed F0 - Praat F0 (excluding outliers): mean={np.mean(f0_diff):.4f} Hz, std={np.std(f0_diff):.4f}")
