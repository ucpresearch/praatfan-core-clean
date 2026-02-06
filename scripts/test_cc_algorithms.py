#!/usr/bin/env python3
"""
Test different variations of the cross-correlation algorithm to find
which one matches Praat's CC method.
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

# Select a test frame
test_t = 0.2060

# Get Praat's values
pm_f0 = call(pm_pitch, "Get value at time", test_t, "Hertz", "linear")
pm_hnr = call(pm_harm, "Get value at time", test_t, "cubic")
pm_ratio = 10 ** (pm_hnr / 10)
pm_r = pm_ratio / (1 + pm_ratio)

print(f"Test frame at t={test_t}s")
print(f"Praat F0: {pm_f0:.2f} Hz")
print(f"Praat HNR: {pm_hnr:.2f} dB")
print(f"Praat r: {pm_r:.6f}")
print()

# Extract frame
center = int(round(test_t * sample_rate))
start = center - half_window
frame = samples[start:start + window_samples].copy()

# Algorithm variants to test
algorithms = {}

# Algorithm 1: Basic forward CC (my current implementation)
def algo_basic_fcc(frame, min_lag, max_lag):
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

# Algorithm 2: FCC with windowed periods
def algo_fcc_windowed(frame, min_lag, max_lag):
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

# Algorithm 3: Standard autocorrelation (AC-style)
def algo_ac_style(frame, min_lag, max_lag):
    window = hanning_window(len(frame))
    windowed = frame * window

    # Compute autocorrelation
    r_a = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag < len(windowed):
            r_a[lag] = np.sum(windowed[:len(windowed)-lag] * windowed[lag:])

    # Window autocorrelation
    r_w = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag < len(window):
            r_w[lag] = np.sum(window[:len(window)-lag] * window[lag:])

    # Normalize
    r_norm = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if r_a[0] > 0 and r_w[0] > 0 and r_w[lag] > 0:
            r_norm[lag] = (r_a[lag] / r_a[0]) / (r_w[lag] / r_w[0])

    return r_norm

# Algorithm 4: FCC with sqrt normalization (different formula)
def algo_fcc_sqrt(frame, min_lag, max_lag):
    fcc = np.zeros(max_lag + 1)
    for lag in range(min_lag, min(max_lag + 1, len(frame) // 2)):
        first = frame[:lag]
        second = frame[lag:2*lag]
        if len(second) == lag:
            corr = np.sum(first * second)
            # Alternative: normalize by total energy
            total_energy = np.sum(first * first) + np.sum(second * second)
            if total_energy > 0:
                fcc[lag] = 2 * corr / total_energy
    return fcc

# Algorithm 5: FCC with Boersma-style normalization
# (pretend periods are windowed samples and use window autocorrelation)
def algo_fcc_boersma(frame, min_lag, max_lag):
    fcc = np.zeros(max_lag + 1)
    for lag in range(min_lag, min(max_lag + 1, len(frame) // 2)):
        first = frame[:lag]
        second = frame[lag:2*lag]
        if len(second) == lag:
            # Use Hanning window on each period
            w = hanning_window(lag)
            first_w = first * w
            second_w = second * w

            # Cross-correlation
            corr = np.sum(first_w * second_w)

            # Energy normalization like Boersma: r = r_ab / sqrt(r_aa(0) * r_bb(0))
            e_first = np.sum(first_w * first_w)
            e_second = np.sum(second_w * second_w)

            # Window energy
            w_energy = np.sum(w * w)

            if e_first > 0 and e_second > 0:
                fcc[lag] = corr / np.sqrt(e_first * e_second)
    return fcc

# Algorithm 6: Full cross-correlation sliding
def algo_full_xcorr(frame, min_lag, max_lag):
    """
    For each lag, compute full cross-correlation between
    signal and shifted signal, then pick peak.
    """
    fcc = np.zeros(max_lag + 1)
    n = len(frame)

    for lag in range(min_lag, min(max_lag + 1, n)):
        # Cross-correlation at this lag
        x1 = frame[:n-lag]
        x2 = frame[lag:]

        corr = np.sum(x1 * x2)
        e1 = np.sum(x1 * x1)
        e2 = np.sum(x2 * x2)

        if e1 > 0 and e2 > 0:
            fcc[lag] = corr / np.sqrt(e1 * e2)

    return fcc

# Algorithm 7: Full cross-correlation with window
def algo_full_xcorr_windowed(frame, min_lag, max_lag):
    window = hanning_window(len(frame))
    windowed = frame * window

    fcc = np.zeros(max_lag + 1)
    n = len(windowed)

    for lag in range(min_lag, min(max_lag + 1, n)):
        x1 = windowed[:n-lag]
        x2 = windowed[lag:]

        corr = np.sum(x1 * x2)
        e1 = np.sum(x1 * x1)
        e2 = np.sum(x2 * x2)

        if e1 > 0 and e2 > 0:
            fcc[lag] = corr / np.sqrt(e1 * e2)

    return fcc

# Algorithm 8: Using only first two periods of the frame
def algo_two_periods_only(frame, min_lag, max_lag):
    """
    For each lag, use only the first 2*lag samples
    """
    fcc = np.zeros(max_lag + 1)

    for lag in range(min_lag, min(max_lag + 1, len(frame) // 2)):
        # Use exactly 2 periods
        segment = frame[:2*lag]

        x1 = segment[:lag]
        x2 = segment[lag:]

        corr = np.sum(x1 * x2)
        e1 = np.sum(x1 * x1)
        e2 = np.sum(x2 * x2)

        if e1 > 0 and e2 > 0:
            fcc[lag] = corr / np.sqrt(e1 * e2)

    return fcc

# Test all algorithms
algorithms = {
    "1. Basic FCC": algo_basic_fcc,
    "2. FCC windowed periods": algo_fcc_windowed,
    "3. AC-style (Boersma)": algo_ac_style,
    "4. FCC sqrt norm": algo_fcc_sqrt,
    "5. FCC Boersma-style": algo_fcc_boersma,
    "6. Full xcorr": algo_full_xcorr,
    "7. Full xcorr windowed": algo_full_xcorr_windowed,
    "8. Two periods only": algo_two_periods_only,
}

print(f"{'Algorithm':<30} {'F0 (Hz)':>10} {'F0 err':>10} {'r':>10} {'r err':>10}")
print("-" * 70)

for name, algo_func in algorithms.items():
    fcc = algo_func(frame, min_lag, max_lag)

    # Find best peak
    best_lag = min_lag
    for lag in range(min_lag + 1, min(max_lag, len(fcc) - 1)):
        if fcc[lag] > fcc[lag-1] and fcc[lag] > fcc[lag+1]:
            if fcc[lag] > fcc[best_lag]:
                best_lag = lag

    # Parabolic interpolation
    if best_lag > min_lag and best_lag < len(fcc) - 1:
        r_prev = fcc[best_lag - 1]
        r_curr = fcc[best_lag]
        r_next = fcc[best_lag + 1]

        denom = r_prev - 2*r_curr + r_next
        if abs(denom) > 1e-10:
            delta = 0.5 * (r_prev - r_next) / denom
            if abs(delta) < 1:
                refined_lag = best_lag + delta
                refined_r = r_curr - 0.25 * (r_prev - r_next) * delta
            else:
                refined_lag = best_lag
                refined_r = r_curr
        else:
            refined_lag = best_lag
            refined_r = r_curr
    else:
        refined_lag = best_lag
        refined_r = fcc[best_lag] if best_lag < len(fcc) else 0

    my_f0 = sample_rate / refined_lag if refined_lag > 0 else 0
    f0_err = abs(my_f0 - pm_f0)
    r_err = abs(refined_r - pm_r)

    print(f"{name:<30} {my_f0:>10.2f} {f0_err:>10.2f} {refined_r:>10.6f} {r_err:>10.6f}")

print()
print(f"{'Target (Praat)':<30} {pm_f0:>10.2f} {'-':>10} {pm_r:>10.6f} {'-':>10}")

# Now test across multiple frames to see consistency
print("\n" + "=" * 70)
print("Testing across multiple frames")
print("=" * 70)

# Get several voiced frames
test_times = []
for i in range(1, call(pm_pitch, "Get number of frames") + 1):
    f0 = call(pm_pitch, "Get value in frame", i, "Hertz")
    if f0 and f0 > 150 and f0 < 300:  # Well-defined pitch range
        t = call(pm_pitch, "Get time from frame number", i)
        hnr = call(pm_harm, "Get value at time", t, "cubic")
        if hnr and hnr > 15:  # High HNR
            test_times.append(t)
            if len(test_times) >= 10:
                break

print(f"Testing {len(test_times)} high-quality voiced frames")
print()

# Test a few promising algorithms across all frames
test_algos = {
    "1. Basic FCC": algo_basic_fcc,
    "3. AC-style": algo_ac_style,
    "6. Full xcorr": algo_full_xcorr,
    "7. Full xcorr win": algo_full_xcorr_windowed,
}

for name, algo_func in test_algos.items():
    f0_errors = []
    r_errors = []

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

        # Run algorithm
        fcc = algo_func(frame, min_lag, max_lag)

        # Find best peak
        best_lag = min_lag
        for lag in range(min_lag + 1, min(max_lag, len(fcc) - 1)):
            if fcc[lag] > fcc[lag-1] and fcc[lag] > fcc[lag+1]:
                if fcc[lag] > fcc[best_lag]:
                    best_lag = lag

        if best_lag <= min_lag or best_lag >= len(fcc) - 1:
            continue

        # Parabolic interpolation
        r_prev = fcc[best_lag - 1]
        r_curr = fcc[best_lag]
        r_next = fcc[best_lag + 1]

        denom = r_prev - 2*r_curr + r_next
        if abs(denom) > 1e-10:
            delta = 0.5 * (r_prev - r_next) / denom
            if abs(delta) < 1:
                refined_lag = best_lag + delta
                refined_r = r_curr - 0.25 * (r_prev - r_next) * delta
            else:
                refined_lag = best_lag
                refined_r = r_curr
        else:
            refined_lag = best_lag
            refined_r = r_curr

        my_f0 = sample_rate / refined_lag
        f0_errors.append(abs(my_f0 - pm_f0))
        r_errors.append(abs(refined_r - pm_r))

    if f0_errors:
        print(f"{name}:")
        print(f"  F0: mean={np.mean(f0_errors):.3f} Hz, max={np.max(f0_errors):.3f} Hz")
        print(f"  r:  mean={np.mean(r_errors):.6f}, max={np.max(r_errors):.6f}")
        print()
