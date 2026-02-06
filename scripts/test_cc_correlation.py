#!/usr/bin/env python3
"""
Test script to investigate the Cross-Correlation (CC) method for Pitch detection.

Goal: Understand what "forward cross-correlation" means algorithmically and
match Praat's correlation values frame-by-frame.
"""

import sys
sys.path.insert(0, '/home/urielc/local/decfiles/private/Dev/git/praatfan-core-clean/src')

import numpy as np
import parselmouth
from parselmouth.praat import call

# Load test sound
sound_path = "/home/urielc/local/decfiles/private/Dev/git/praatfan-core-clean/tests/fixtures/one_two_three_four_five.wav"
pm_sound = parselmouth.Sound(sound_path)
from praatfan.sound import Sound
my_sound = Sound.from_file(sound_path)

# Parameters for CC method
min_pitch = 75.0
time_step = 0.01

# Get parselmouth Pitch CC
pm_pitch = call(pm_sound, "To Pitch (cc)", time_step, min_pitch, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 600)

pm_n_frames = call(pm_pitch, "Get number of frames")
pm_times = []
pm_values = []
pm_strengths = []

print(f"Parselmouth Pitch CC: {pm_n_frames} frames")
print(f"Time step: {time_step}s")
print(f"Min pitch: {min_pitch} Hz")
print()

# Get frame times and values
for i in range(1, pm_n_frames + 1):
    t = call(pm_pitch, "Get time from frame number", i)
    f0 = call(pm_pitch, "Get value in frame", i, "Hertz")
    strength = call(pm_pitch, "Get strength in frame", i)
    pm_times.append(t)
    pm_values.append(f0 if f0 else 0.0)
    pm_strengths.append(strength if strength else 0.0)

# Find first voiced frame with good F0
voiced_frames = [(i, pm_times[i], pm_values[i], pm_strengths[i])
                 for i in range(len(pm_times)) if pm_values[i] > 0]

print(f"Found {len(voiced_frames)} voiced frames")
print()

if voiced_frames:
    # Analyze first few voiced frames
    print("First 5 voiced frames:")
    print("-" * 70)
    for idx, t, f0, strength in voiced_frames[:5]:
        print(f"Frame {idx+1}: t={t:.4f}s, F0={f0:.2f} Hz, strength={strength:.6f}")
    print()

# Now investigate what "forward cross-correlation" means
# The signal is the audio frame, and we're looking for periodicity

# Get frame timing info
pm_t1 = pm_times[0] if pm_times else 0
pm_dt = time_step

# CC uses window = 2/min_pitch (2 periods)
window_duration = 2.0 / min_pitch
window_samples = int(round(window_duration * my_sound.sample_rate))
if window_samples % 2 == 0:
    window_samples += 1
half_window = window_samples // 2

print(f"Window duration: {window_duration:.4f}s ({window_samples} samples)")
print(f"t1 (first frame time): {pm_t1:.6f}s")
print()

# Test one frame in detail
if voiced_frames:
    test_idx, test_t, test_f0, test_strength = voiced_frames[0]
    print(f"Testing frame {test_idx+1} at t={test_t:.4f}s")
    print(f"  Parselmouth: F0={test_f0:.2f} Hz, strength={test_strength:.6f}")
    print()

    # Extract frame samples
    center_sample = int(round(test_t * my_sound.sample_rate))
    start_sample = center_sample - half_window
    end_sample = start_sample + window_samples

    if start_sample >= 0 and end_sample <= len(my_sound.samples):
        frame = my_sound.samples[start_sample:end_sample].copy()
    else:
        frame = np.zeros(window_samples)
        src_start = max(0, start_sample)
        src_end = min(len(my_sound.samples), end_sample)
        dst_start = src_start - start_sample
        dst_end = dst_start + (src_end - src_start)
        frame[dst_start:dst_end] = my_sound.samples[src_start:src_end]

    # Expected lag for this F0
    expected_lag = int(round(my_sound.sample_rate / test_f0))
    print(f"  Expected lag for {test_f0:.2f} Hz: {expected_lag} samples")
    print()

    # Test different cross-correlation methods
    print("Testing correlation methods:")
    print("-" * 70)

    def hanning_window(n):
        """Hanning window"""
        if n <= 1:
            return np.array([1.0])
        i = np.arange(n)
        return 0.5 - 0.5 * np.cos(2 * np.pi * i / (n - 1))

    # Method 1: Standard autocorrelation (AC style)
    min_lag = int(np.ceil(my_sound.sample_rate / 600))  # max pitch
    max_lag = int(np.floor(my_sound.sample_rate / min_pitch))  # min pitch

    window = hanning_window(len(frame))
    windowed = frame * window

    # Standard autocorrelation
    r_ac = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag < len(windowed):
            r_ac[lag] = np.sum(windowed[:len(windowed)-lag] * windowed[lag:])

    # Window autocorrelation for normalization
    r_w = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag < len(window):
            r_w[lag] = np.sum(window[:len(window)-lag] * window[lag:])

    # Find peak in normalized AC
    best_lag_ac = 0
    best_r_ac = 0
    for lag in range(min_lag, min(max_lag + 1, len(r_ac) - 1)):
        if r_ac[0] > 0 and r_w[0] > 0 and r_w[lag] > 0:
            r_norm = (r_ac[lag] / r_ac[0]) / (r_w[lag] / r_w[0])
            if r_norm > best_r_ac:
                best_r_ac = r_norm
                best_lag_ac = lag

    print(f"Method 1 (Standard AC): lag={best_lag_ac} -> {my_sound.sample_rate/best_lag_ac:.2f} Hz, r={best_r_ac:.6f}")

    # Method 2: Forward cross-correlation - correlate period 1 with period 2
    # frame[0:lag] with frame[lag:2*lag]
    best_lag_fcc = 0
    best_r_fcc = 0

    for lag in range(min_lag, min(max_lag + 1, len(frame) // 2)):
        first_period = frame[:lag]
        second_period = frame[lag:2*lag]

        if len(second_period) == lag:
            corr = np.sum(first_period * second_period)
            e1 = np.sum(first_period * first_period)
            e2 = np.sum(second_period * second_period)
            if e1 > 0 and e2 > 0:
                r = corr / np.sqrt(e1 * e2)
                if r > best_r_fcc:
                    best_r_fcc = r
                    best_lag_fcc = lag

    print(f"Method 2 (Forward CC - raw): lag={best_lag_fcc} -> {my_sound.sample_rate/best_lag_fcc:.2f} Hz, r={best_r_fcc:.6f}")

    # Method 3: Forward CC with Hanning window on each period
    best_lag_fcc_win = 0
    best_r_fcc_win = 0

    for lag in range(min_lag, min(max_lag + 1, len(frame) // 2)):
        first_period = frame[:lag].copy()
        second_period = frame[lag:2*lag].copy()

        if len(second_period) == lag:
            # Apply Hanning window to each period
            w = hanning_window(lag)
            first_period = first_period * w
            second_period = second_period * w

            corr = np.sum(first_period * second_period)
            e1 = np.sum(first_period * first_period)
            e2 = np.sum(second_period * second_period)
            if e1 > 0 and e2 > 0:
                r = corr / np.sqrt(e1 * e2)
                if r > best_r_fcc_win:
                    best_r_fcc_win = r
                    best_lag_fcc_win = lag

    print(f"Method 3 (Forward CC - windowed): lag={best_lag_fcc_win} -> {my_sound.sample_rate/best_lag_fcc_win:.2f} Hz, r={best_r_fcc_win:.6f}")

    # Method 4: Normalized cross-correlation over full frame
    # CC(lag) = sum(frame[t] * frame[t+lag]) / sqrt(E1 * E2)
    # where E1 = sum(frame[0:N-lag]^2), E2 = sum(frame[lag:N]^2)
    best_lag_ncc = 0
    best_r_ncc = 0

    for lag in range(min_lag, min(max_lag + 1, len(frame) - 1)):
        x1 = frame[:len(frame)-lag]
        x2 = frame[lag:]

        corr = np.sum(x1 * x2)
        e1 = np.sum(x1 * x1)
        e2 = np.sum(x2 * x2)
        if e1 > 0 and e2 > 0:
            r = corr / np.sqrt(e1 * e2)
            if r > best_r_ncc:
                best_r_ncc = r
                best_lag_ncc = lag

    print(f"Method 4 (Normalized CC full): lag={best_lag_ncc} -> {my_sound.sample_rate/best_lag_ncc:.2f} Hz, r={best_r_ncc:.6f}")

    # Method 5: Like Method 4 but with Hanning window first
    best_lag_ncc_win = 0
    best_r_ncc_win = 0

    for lag in range(min_lag, min(max_lag + 1, len(windowed) - 1)):
        x1 = windowed[:len(windowed)-lag]
        x2 = windowed[lag:]

        corr = np.sum(x1 * x2)
        e1 = np.sum(x1 * x1)
        e2 = np.sum(x2 * x2)
        if e1 > 0 and e2 > 0:
            r = corr / np.sqrt(e1 * e2)
            if r > best_r_ncc_win:
                best_r_ncc_win = r
                best_lag_ncc_win = lag

    print(f"Method 5 (Normalized CC windowed): lag={best_lag_ncc_win} -> {my_sound.sample_rate/best_lag_ncc_win:.2f} Hz, r={best_r_ncc_win:.6f}")

    # Method 6: Forward CC on DC-removed signal
    frame_dc = frame - np.mean(frame)
    best_lag_fcc_dc = 0
    best_r_fcc_dc = 0

    for lag in range(min_lag, min(max_lag + 1, len(frame_dc) // 2)):
        first_period = frame_dc[:lag]
        second_period = frame_dc[lag:2*lag]

        if len(second_period) == lag:
            corr = np.sum(first_period * second_period)
            e1 = np.sum(first_period * first_period)
            e2 = np.sum(second_period * second_period)
            if e1 > 0 and e2 > 0:
                r = corr / np.sqrt(e1 * e2)
                if r > best_r_fcc_dc:
                    best_r_fcc_dc = r
                    best_lag_fcc_dc = lag

    print(f"Method 6 (Forward CC DC-removed): lag={best_lag_fcc_dc} -> {my_sound.sample_rate/best_lag_fcc_dc:.2f} Hz, r={best_r_fcc_dc:.6f}")

    print()
    print(f"Target from Praat: strength={test_strength:.6f}")
    print()

    # Method 7: The exact Boersma normalization but for CC
    # r_x(tau) = r_a(tau) / sqrt(r_w(tau) * r_a(0))
    # This is a different normalization...
    best_lag_7 = 0
    best_r_7 = 0

    for lag in range(min_lag, min(max_lag + 1, len(r_ac) - 1)):
        if r_ac[0] > 0 and r_w[lag] > 0:
            # Alternative normalization
            r_norm = r_ac[lag] / np.sqrt(r_w[lag] * r_ac[0])
            if r_norm > best_r_7:
                best_r_7 = r_norm
                best_lag_7 = lag

    print(f"Method 7 (sqrt normalization): lag={best_lag_7} -> {my_sound.sample_rate/best_lag_7:.2f} Hz, r={best_r_7:.6f}")

    # Let's look at individual lags around the expected period
    print()
    print("Detailed look at lags near expected period:")
    print("-" * 70)

    for lag in range(expected_lag - 5, expected_lag + 6):
        if lag >= min_lag and lag < len(frame) // 2:
            first_period = frame[:lag]
            second_period = frame[lag:2*lag]
            if len(second_period) == lag:
                corr = np.sum(first_period * second_period)
                e1 = np.sum(first_period * first_period)
                e2 = np.sum(second_period * second_period)
                if e1 > 0 and e2 > 0:
                    r_fcc = corr / np.sqrt(e1 * e2)
                else:
                    r_fcc = 0
            else:
                r_fcc = 0

            # Also show AC normalized value
            if r_ac[0] > 0 and r_w[0] > 0 and r_w[lag] > 0:
                r_ac_norm = (r_ac[lag] / r_ac[0]) / (r_w[lag] / r_w[0])
            else:
                r_ac_norm = 0

            freq = my_sound.sample_rate / lag
            marker = " <-- expected" if lag == expected_lag else ""
            print(f"lag={lag} ({freq:.2f} Hz): FCC={r_fcc:.6f}, AC_norm={r_ac_norm:.6f}{marker}")
